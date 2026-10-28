import json

from gemq import resource_ledger
from gemq.resource_ledger import ResourceLedger, merge_ledgers


def test_command_records_component_and_writes_json(tmp_path, monkeypatch):
    monkeypatch.setattr(resource_ledger.torch.cuda, "is_available", lambda: False)
    output_path = tmp_path / "resources.json"
    ledger = ResourceLedger(output_path)

    with ledger.command():
        with ledger.component("solve_lp"):
            pass

    payload = json.loads(output_path.read_text())
    component = payload["components"]["solve_lp"]
    assert component["status"] == "executed"
    assert component["wall_seconds"] >= 0.0
    assert component["per_gpu_peak_memory_gib"] == {}
    assert component["peak_gpu_memory_sum_gib"] == 0.0
    assert payload["summary"]["stage_wall_seconds"] >= component["wall_seconds"]


def test_merge_ledgers_accumulates_matching_components(tmp_path, monkeypatch):
    monkeypatch.setattr(resource_ledger.torch.cuda, "is_available", lambda: False)
    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"
    output_path = tmp_path / "merged.json"

    for path in (first_path, second_path):
        ledger = ResourceLedger(path)
        with ledger.command():
            with ledger.component("quant_gptq"):
                pass

    merge_ledgers(output_path, [first_path, second_path])

    payload = json.loads(output_path.read_text())
    assert payload["components"]["quant_gptq"]["status"] == "executed"
    assert payload["components"]["quant_gptq"]["wall_seconds"] >= 0.0
    assert payload["summary"]["stage_wall_seconds"] >= 0.0


def test_failure_archive_and_idempotent_merge(tmp_path):
    import pytest
    path = tmp_path / "resources.json"
    with pytest.raises(ValueError, match="deliberate"):
        ledger = ResourceLedger(path)
        with ledger.command():
            with ledger.component("quant_gptq"):
                raise ValueError("deliberate")
    failed = json.loads(path.read_text())
    assert failed["components"]["quant_gptq"]["status"] == "failed"
    ledger = ResourceLedger(path)
    with ledger.command():
        with ledger.component("evaluate_zeroshot"):
            pass
    fresh = json.loads(path.read_text())
    assert "quant_gptq" not in fresh["components"]
    assert fresh["summary"]["core_wall_seconds"] == 0
    assert json.loads(path.with_name("resources." + failed["run_id"] + ".json").read_text()) == failed
    output = tmp_path / "merged.json"
    merge_ledgers(output, [path])
    first = json.loads(output.read_text())
    merge_ledgers(output, [output, path])
    second = json.loads(output.read_text())
    assert second["components"] == first["components"]
    assert len(second["source_records"]) == 1
    merge_ledgers(output, [path, tmp_path / "missing.json"])
    assert json.loads(output.read_text())["status"] == "incomplete"


def test_component_timer_includes_cuda_synchronization(tmp_path, monkeypatch):
    ledger = ResourceLedger(tmp_path / "resources.json")
    clock = [10.0]
    monkeypatch.setattr(resource_ledger.time, "perf_counter", lambda: clock[0])
    monkeypatch.setattr(ledger, "_reset_peaks", lambda: None)
    def finish():
        clock[0] += 3.0
        return {"0": 100}
    monkeypatch.setattr(ledger, "_read_peaks", finish)
    with ledger.component("quant_gptq"):
        clock[0] += 2.0
    assert ledger.components["quant_gptq"]["wall_seconds"] == 5.0
