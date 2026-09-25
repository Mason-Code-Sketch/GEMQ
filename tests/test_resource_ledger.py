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
