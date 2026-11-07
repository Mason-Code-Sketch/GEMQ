from types import SimpleNamespace
import pytest
from gemq.batch_profiles import auto_batch_cap, resolve_cli_batch_cap

@pytest.mark.parametrize("model_type,caps", [("mixtral", [34, 13, 48]), ("deepseek_v2", [32, 24, 20]), ("qwen2_moe", [32, 24, 56]), ("qwen3_moe", [51, 60, 96]), ("unknown", [51, 17, 96])])
def test_model_profiles_and_explicit_override(model_type, caps):
    model = SimpleNamespace(config=SimpleNamespace(model_type=model_type))
    for profile, cap in zip(["seven", "mmlu_5shot", "gsm8k_5shot"], caps):
        assert auto_batch_cap(model, profile) == cap
        assert auto_batch_cap(model, profile, 7) == 7
        args = SimpleNamespace(max_batch_size=51, _max_batch_size_explicit=False)
        assert resolve_cli_batch_cap(args, model, profile) == cap
        args._max_batch_size_explicit = True
        args.max_batch_size = 51
        assert resolve_cli_batch_cap(args, model, profile) == 51
    assert auto_batch_cap(model, "zeroshot") == caps[0]

def test_manual_namespace_retains_explicit_value():
    model = SimpleNamespace(config=SimpleNamespace(model_type="mixtral"))
    assert resolve_cli_batch_cap(SimpleNamespace(max_batch_size=11), model, "seven") == 11

import json
from unittest.mock import Mock
from gemq import eval_zeroshot as entry
from test_eval_zeroshot import mock_harness, results

@pytest.mark.parametrize("model_type,cap", [("mixtral", 34), ("deepseek_v2", 32), ("qwen2_moe", 32), ("qwen3_moe", 51)])
@pytest.mark.parametrize("explicit", [None, 51])
def test_loaded_model_cap_reaches_harness_and_saved_result(tmp_path, monkeypatch, model_type, cap, explicit):
    checkpoint = tmp_path / "arbitrary-name"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}")
    (checkpoint / "qmodel.pt").write_bytes(b"checkpoint")
    _, wrapper = mock_harness(monkeypatch, results())
    model = Mock()
    model.config.model_type = model_type
    monkeypatch.setattr(entry, "load_model", lambda args: (model, "tokenizer"))
    extra = [] if explicit is None else ["--max_batch_size", str(explicit)]
    entry.main(["--model_path", str(checkpoint), *extra])
    expected = cap if explicit is None else explicit
    assert wrapper.call_args.kwargs["max_batch_size"] == expected
    saved = json.loads((checkpoint / "zeroshot.json").read_text())
    assert saved["experiment"]["max_batch_size"] == expected
