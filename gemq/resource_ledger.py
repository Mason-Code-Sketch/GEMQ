"""Stage resource accounting for GEMQ protocol runs."""

from __future__ import annotations

from contextlib import contextmanager
import json
import platform
from importlib.metadata import version, PackageNotFoundError
import os
from pathlib import Path
import time
import sys
import uuid
import threading
import psutil
from typing import Iterator, Mapping, Sequence

import torch


_BYTES_PER_GIB = 1024**3


def _visible_gpu_ids() -> list[int]:
    """Map local CUDA indices to physical IDs selected for this command."""
    configured = os.environ.get("CUDA_VISIBLE_DEVICES")
    if configured:
        values = [value.strip() for value in configured.split(",") if value.strip()]
        if all(value.isdigit() for value in values):
            return [int(value) for value in values]
    return list(range(torch.cuda.device_count()))


def _memory_fields(per_gpu_bytes: Mapping[str, int]) -> dict:
    per_gpu_gib = {
        str(device): float(value) / _BYTES_PER_GIB
        for device, value in sorted(per_gpu_bytes.items(), key=lambda item: int(item[0]))
    }
    return {
        "per_gpu_peak_memory_gib": per_gpu_gib,
        "peak_gpu_memory_sum_gib": sum(per_gpu_gib.values()),
    }


def _merge_memory_fields(records: Sequence[Mapping]) -> dict:
    per_gpu: dict[str, float] = {}
    for record in records:
        for device, value in record.get("per_gpu_peak_memory_gib", {}).items():
            per_gpu[str(device)] = max(per_gpu.get(str(device), 0.0), float(value))
    return _memory_fields(
        {device: int(value * _BYTES_PER_GIB) for device, value in per_gpu.items()}
    )


class ResourceLedger:
    """Record wall time and PyTorch allocator peaks for one GEMQ command."""

    def __init__(self, output_path: str | Path | None):
        self.output_path = Path(output_path) if output_path else None
        self.enabled = self.output_path is not None
        self.components: dict[str, dict] = {}
        self._command_wall_seconds = 0.0
        self.run_id = uuid.uuid4().hex
        self.status = "pending"
        self.error = None
        self.environment = None
        self._active_component = None
        self._max_rss = 0
        self._component_rss = 0

    @contextmanager
    def command(self) -> Iterator[None]:
        """Measure the complete Python command, including setup and I/O."""
        if not self.enabled:
            yield
            return
        if self.output_path.exists():
            previous = json.loads(self.output_path.read_text())
            archive = self.output_path.with_name(self.output_path.stem + "." + previous.get("run_id", uuid.uuid4().hex) + ".json")
            if archive.exists():
                raise FileExistsError(archive)
            self.output_path.rename(archive)
        started_at = time.perf_counter()
        self.status = "running"
        self.environment = _environment()
        stop = threading.Event()
        def sample():
            while not stop.wait(1.0):
                self._sample_ram()
        thread = threading.Thread(target=sample, daemon=True)
        self._sample_ram()
        self.write()
        thread.start()
        try:
            yield
        except BaseException as error:
            self.status, self.error = "failed", repr(error)
            raise
        else:
            self.status = "complete"
        finally:
            stop.set()
            thread.join()
            self._sample_ram()
            self._command_wall_seconds = time.perf_counter() - started_at
            self.write()

    def _sample_ram(self):
        try:
            process = psutil.Process()
            members = [process]
            try:
                members.extend(process.children(recursive=True))
            except (psutil.Error, OSError) as error:
                self.ram_sampling_error = repr(error)
            rss = 0
            for member in members:
                try:
                    rss += member.memory_info().rss
                except (psutil.Error, OSError) as error:
                    self.ram_sampling_error = repr(error)
            self._max_rss = max(self._max_rss, rss)
            if self._active_component is not None:
                self._component_rss = max(self._component_rss, rss)
        except (psutil.Error, OSError) as error:
            self.ram_sampling_error = repr(error)

    def _reset_peaks(self) -> None:
        if not torch.cuda.is_initialized():
            return
        for local_device in range(torch.cuda.device_count()):
            with torch.cuda.device(local_device):
                torch.cuda.synchronize(local_device)
                torch.cuda.reset_peak_memory_stats(local_device)

    def _read_peaks(self) -> dict[str, int]:
        if not torch.cuda.is_initialized():
            return {}
        visible_ids = _visible_gpu_ids()
        peaks: dict[str, int] = {}
        for local_device in range(torch.cuda.device_count()):
            with torch.cuda.device(local_device):
                torch.cuda.synchronize(local_device)
                physical_device = (
                    visible_ids[local_device]
                    if local_device < len(visible_ids)
                    else local_device
                )
                peaks[str(physical_device)] = int(
                    torch.cuda.max_memory_reserved(local_device)
                )
        return peaks

    @contextmanager
    def component(self, name: str) -> Iterator[None]:
        """Record a directly executed component without changing its work."""
        if not self.enabled:
            yield
            return
        if self._active_component is not None:
            raise RuntimeError("Resource components must not overlap")
        self._active_component = name
        self._component_rss = 0
        reset_error = None
        try:
            self._reset_peaks()
        except Exception as error:
            reset_error = repr(error)
        self._sample_ram()
        self.write()
        started_at = time.perf_counter()
        status, error_text = "executed", None
        try:
            yield
        except BaseException as error:
            status, error_text = "failed", repr(error)
            raise
        finally:
            peaks, measurement_error = {}, reset_error
            try:
                peaks = self._read_peaks()  # Synchronize before stopping the timer.
            except Exception as error:
                measurement_error = repr(error)
            elapsed = time.perf_counter() - started_at
            self._sample_ram()
            record = {"status": status, "error": error_text,
                      "measurement_error": measurement_error,
                      "wall_seconds": elapsed,
                      "max_sampled_process_tree_rss_bytes": self._component_rss,
                      **_memory_fields(peaks)}
            existing = self.components.get(name)
            if existing is None:
                self.components[name] = record
            else:
                record["wall_seconds"] += existing["wall_seconds"]
                record["max_sampled_process_tree_rss_bytes"] = max(record["max_sampled_process_tree_rss_bytes"], existing.get("max_sampled_process_tree_rss_bytes", 0))
                record.update(_merge_memory_fields((existing, record)))
                if existing["status"] == "failed":
                    record["status"], record["error"] = "failed", existing.get("error")
                self.components[name] = record
            self._active_component = None
            self.write()
            print(f"[resources] {name} status={status} wall={elapsed:.3f}s", flush=True)

    def to_dict(self) -> dict:
        component_wall = sum(
            float(component["wall_seconds"]) for component in self.components.values()
        )
        summary_memory = _merge_memory_fields(tuple(self.components.values()))
        core_names = {"comp_grad", "comp_stats", "solve_lp", "quant_gptq", "ft_routers"}
        core_wall = sum(value["wall_seconds"] for name, value in self.components.items() if name in core_names)
        return {
            "schema_version": 2,
            "run_id": self.run_id, "status": self.status, "error": self.error,
            "pid": os.getpid(), "command": sys.argv,
            "active_component": self._active_component,
            "environment": self.environment,
            "measurement": {
                "wall_seconds": {
                    "unit": "seconds",
                    "source": "time.perf_counter",
                    "aggregation": "Python command wall time and directly measured component wall time",
                },
                "per_gpu_peak_memory_gib": {
                    "unit": "GiB",
                    "source": "torch.cuda.max_memory_reserved",
                    "aggregation": "maximum PyTorch reserved memory for each physical GPU during the component",
                    "scope": "excludes driver, CUDA context, NCCL, and non-PyTorch allocations",
                },
                "peak_gpu_memory_sum_gib": {
                    "unit": "GiB",
                    "source": "sum(per_gpu_peak_memory_gib.values())",
                    "aggregation": "sum of per-GPU component peaks, which may occur at different times",
                },
            },
            "components": self.components,
            "summary": {
                "stage_wall_seconds": self._command_wall_seconds,
                "recorded_component_wall_seconds": component_wall,
                "unattributed_wall_seconds": self._command_wall_seconds - component_wall,
                **summary_memory,
                "max_component_peak_gpu_memory_sum_gib": max((v["peak_gpu_memory_sum_gib"] for v in self.components.values()), default=0.0),
                "core_wall_seconds": core_wall,
                "core_peak_gpu_memory_sum_gib": _merge_memory_fields([v for k, v in self.components.items() if k in core_names])["peak_gpu_memory_sum_gib"],
                "core_time_percent": {name: 100 * value["wall_seconds"] / core_wall for name, value in self.components.items() if name in core_names and core_wall},
                "max_sampled_process_tree_rss_bytes": self._max_rss,
                "ram_sample_interval_seconds": 1.0,
                "ram_sampling_error": getattr(self, "ram_sampling_error", None),
            },
        }

    def write(self) -> None:
        if not self.enabled:
            return
        assert self.output_path is not None
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.output_path.with_suffix(self.output_path.suffix + ".tmp")
        payload = self.to_dict()
        if hasattr(self, "source_records"):
            payload["source_records"] = self.source_records
            payload["missing_sources"] = self.missing_sources
        temporary.write_text(json.dumps(payload, indent=2) + "\n")
        temporary.replace(self.output_path)


def merge_ledgers(output_path: str | Path, input_paths: Sequence[str | Path]) -> None:
    """Combine command records into one protocol-stage resource breakdown."""
    output = Path(output_path)
    aggregate = ResourceLedger(None)
    aggregate.enabled = True
    aggregate.output_path = output
    aggregate.components = {}
    aggregate._command_wall_seconds = 0.0
    records = {}
    missing = []
    for input_path in input_paths:
        path = Path(input_path)
        if not path.exists():
            missing.append(str(path))
            continue
        payload = json.loads(path.read_text())
        missing.extend(payload.get("missing_sources", []))
        for record in payload.get("source_records", [payload]):
            key = record.get("run_id", str(path.resolve()))
            records[key] = record
    aggregate.status = "complete" if records and not missing and all(r.get("status") == "complete" for r in records.values()) else "incomplete"
    aggregate.source_records = list(records.values())
    aggregate.missing_sources = sorted(set(missing))
    for payload in records.values():
        aggregate._command_wall_seconds += float(payload.get("summary", {}).get("stage_wall_seconds", 0.0))
        aggregate._max_rss = max(aggregate._max_rss, payload.get("summary", {}).get("max_sampled_process_tree_rss_bytes", 0))
        for name, record in payload.get("components", {}).items():
            existing = aggregate.components.get(name)
            if existing is None:
                aggregate.components[name] = dict(record)
            else:
                aggregate.components[name] = {
                    "status": "failed" if "failed" in (existing["status"], record["status"]) else "executed",
                    "wall_seconds": float(existing["wall_seconds"]) + float(record["wall_seconds"]),
                    "error": record.get("error") or existing.get("error"),
                    "measurement_error": record.get("measurement_error") or existing.get("measurement_error"),
                    "max_sampled_process_tree_rss_bytes": max(existing.get("max_sampled_process_tree_rss_bytes", 0), record.get("max_sampled_process_tree_rss_bytes", 0)),
                    **_merge_memory_fields((existing, record)),
                }
    aggregate.write()


def _environment():
    result = {"torch": str(torch.__version__), "cuda": torch.version.cuda,
              "python": platform.python_version(),
              "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "gpus": []}
    try:
        result["transformers"] = version("transformers")
    except PackageNotFoundError:
        result["transformers"] = None
    try:
        if torch.cuda.is_available():
            for index in range(torch.cuda.device_count()):
                device = torch.cuda.get_device_properties(index)
                result["gpus"].append({"logical_device": index, "name": device.name,
                                       "total_memory_bytes": device.total_memory})
    except Exception as error:
        result["gpu_query_error"] = repr(error)
    return result
