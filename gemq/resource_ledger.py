"""Stage resource accounting for GEMQ protocol runs."""

from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
import time
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
        if self.enabled and self.output_path.exists():
            existing = json.loads(self.output_path.read_text())
            self.components = dict(existing.get("components", {}))
            self._command_wall_seconds = float(
                existing.get("summary", {}).get("stage_wall_seconds", 0.0)
            )

    @contextmanager
    def command(self) -> Iterator[None]:
        """Measure the complete Python command, including setup and I/O."""
        if not self.enabled:
            yield
            return
        started_at = time.perf_counter()
        try:
            yield
        finally:
            self._command_wall_seconds += time.perf_counter() - started_at
            self.write()

    def _reset_peaks(self) -> None:
        if not torch.cuda.is_available():
            return
        for local_device in range(torch.cuda.device_count()):
            with torch.cuda.device(local_device):
                torch.cuda.synchronize(local_device)
                torch.cuda.reset_peak_memory_stats(local_device)

    def _read_peaks(self) -> dict[str, int]:
        if not torch.cuda.is_available():
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
        self._reset_peaks()
        started_at = time.perf_counter()
        try:
            yield
        finally:
            record = {
                "status": "executed",
                "wall_seconds": time.perf_counter() - started_at,
                **_memory_fields(self._read_peaks()),
            }
            existing = self.components.get(name)
            if existing is None:
                self.components[name] = record
            else:
                self.components[name] = {
                    "status": "executed",
                    "wall_seconds": float(existing["wall_seconds"])
                    + float(record["wall_seconds"]),
                    **_merge_memory_fields((existing, record)),
                }

    def to_dict(self) -> dict:
        component_wall = sum(
            float(component["wall_seconds"]) for component in self.components.values()
        )
        summary_memory = _merge_memory_fields(tuple(self.components.values()))
        return {
            "schema_version": 1,
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
            },
        }

    def write(self) -> None:
        if not self.enabled:
            return
        assert self.output_path is not None
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.output_path.with_suffix(self.output_path.suffix + ".tmp")
        temporary.write_text(json.dumps(self.to_dict(), indent=2) + "\n")
        temporary.replace(self.output_path)


def merge_ledgers(output_path: str | Path, input_paths: Sequence[str | Path]) -> None:
    """Combine command records into one protocol-stage resource breakdown."""
    output = Path(output_path)
    aggregate = ResourceLedger(None)
    aggregate.enabled = True
    aggregate.output_path = output
    aggregate.components = {}
    aggregate._command_wall_seconds = 0.0
    for input_path in input_paths:
        path = Path(input_path)
        if not path.exists():
            continue
        payload = json.loads(path.read_text())
        aggregate._command_wall_seconds += float(
            payload.get("summary", {}).get("stage_wall_seconds", 0.0)
        )
        for name, record in payload.get("components", {}).items():
            existing = aggregate.components.get(name)
            if existing is None:
                aggregate.components[name] = dict(record)
                continue
            aggregate.components[name] = {
                "status": "executed",
                "wall_seconds": float(existing["wall_seconds"])
                + float(record["wall_seconds"]),
                **_merge_memory_fields((existing, record)),
            }
    aggregate.write()
