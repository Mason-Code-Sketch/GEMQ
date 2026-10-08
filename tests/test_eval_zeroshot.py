import json
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

from gemq import eval_zeroshot as entry
from gemq.utils import eval_utils


@pytest.fixture
def checkpoint(tmp_path):
    path = tmp_path / "packed"
    path.mkdir()
    (path / "config.json").write_text("{}")
    (path / "qmodel.pt").write_bytes(b"checkpoint")
    return path


def results():
    return {
        "results": {task: {"acc,none": .1, "acc_norm,none": .2} for task in eval_utils.ZEROSHOT_TASKS if task != "mmlu"},
        "groups": {"mmlu": {"acc,none": .9}},
    }


def mock_harness(monkeypatch, output):
    package = ModuleType("lm_eval")
    package.evaluator = SimpleNamespace(simple_evaluate=Mock(return_value=output))
    hf = ModuleType("lm_eval.models.huggingface")
    hf.HFLM = Mock()
    utils = ModuleType("lm_eval.utils")
    utils.handle_non_serializable = str
    for name, module in (("lm_eval", package), ("lm_eval.models.huggingface", hf), ("lm_eval.utils", utils)):
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(eval_utils, "version", lambda name: eval_utils.LM_EVAL_VERSION)
    monkeypatch.setattr(entry, "version", lambda name: eval_utils.LM_EVAL_VERSION)
    return package.evaluator.simple_evaluate, hf.HFLM


def test_default_output_is_inside_checkpoint(checkpoint):
    args = entry.parse_args(["--model_path", str(checkpoint)])
    assert tuple(args.tasks) == eval_utils.ZEROSHOT_TASKS
    assert args.batch_size == 1
    assert args.max_batch_size == 8
    assert args.trust_remote_code
    assert entry.result_path(args) == checkpoint / "zeroshot.json"


@pytest.mark.parametrize("extra", [
    ["--batch_size", "0"], ["--max_batch_size", "0"], ["--limit", "0"], ["--max_length", "0"],
    ["--tasks", "piqa", "piqa"], ["--limit", "4", "--output", "x.json"],
])
def test_invalid_cli_is_rejected(checkpoint, extra):
    with pytest.raises(SystemExit):
        entry.parse_args(["--model_path", str(checkpoint), *extra])


def test_auto_batch_size(checkpoint):
    args = entry.parse_args(["--model_path", str(checkpoint), "--batch_size", "auto", "--max_batch_size", "4"])
    assert args.batch_size == "auto"
    assert args.max_batch_size == 4


def test_pytorch_backend_leaves_model_unchanged(checkpoint):
    args = entry.parse_args(["--model_path", str(checkpoint), "--backend", "pytorch"])
    model = Mock()
    assert entry.prepare_backend(model, "tokenizer", args) == "pytorch"
    model.assert_not_called()


@pytest.mark.parametrize("extra", [["--is_fp"], ["--model_dtype", "bfloat16"], ["--device", "cpu"]])
def test_triton_backend_rejects_unsupported_settings(checkpoint, extra):
    args = entry.parse_args(["--model_path", str(checkpoint), "--backend", "triton", *extra])
    model = Mock()
    model.config.model_type = "qwen3_moe"
    with pytest.raises(ValueError, match="packed Qwen3 FP16"):
        entry.prepare_backend(model, "tokenizer", args)


@pytest.mark.parametrize("offset", [0.0, 1.0, float("nan")])
def test_triton_backend_checks_logits_before_evaluation(checkpoint, monkeypatch, offset):
    import torch

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(16, 8)
            self.config = SimpleNamespace(model_type="qwen3_moe")
            self.offset = 0.0

        def get_input_embeddings(self):
            return self.embedding

        def forward(self, ids, use_cache=False):
            assert not use_cache
            logits = torch.arange(16, dtype=torch.float32).expand(*ids.shape, 16) + self.offset
            return SimpleNamespace(logits=logits)

    model = TinyModel()
    tokenizer = Mock(return_value=SimpleNamespace(input_ids=torch.tensor([[1, 2, 3]])))
    patch = ModuleType("gemq.inference.patch")
    patch.prepare_for_inference = Mock(side_effect=lambda model, *a, **kw: setattr(model, "offset", offset))
    monkeypatch.setitem(sys.modules, "gemq.inference.patch", patch)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    args = entry.parse_args(["--model_path", str(checkpoint)])
    if offset == 0:
        assert entry.prepare_backend(model, tokenizer, args) == "triton"
        assert all(not parameter.requires_grad for parameter in model.parameters())
    else:
        with pytest.raises(RuntimeError, match="numerical check"):
            entry.prepare_backend(model, tokenizer, args)
    patch.prepare_for_inference.assert_called_once_with(model, "Qwen/Qwen3-30B-A3B", is_fp=False)


@pytest.mark.parametrize("name", ["zeroshot.json", "zeroshot.log", "config.json"])
def test_existing_files_are_not_overwritten(checkpoint, name):
    path = checkpoint / name
    path.write_text("old")
    extra = ["--output", str(path)] if name == "config.json" else []
    args = entry.parse_args(["--model_path", str(checkpoint), *extra])
    with pytest.raises(FileExistsError):
        entry.result_path(args)
    assert path.read_text() == "old"


def test_smoke_still_requires_checkpoint_files(checkpoint):
    args = entry.parse_args(["--model_path", str(checkpoint), "--limit", "4"])
    assert entry.result_path(args) is None
    (checkpoint / "qmodel.pt").unlink()
    with pytest.raises(FileNotFoundError):
        entry.result_path(args)


def test_summary_counts_mmlu_once_and_prefers_normalized_accuracy():
    output = results()
    output["results"]["mmlu_subject"] = {"acc,none": .8}
    summary = eval_utils.summarize_lm_eval(output, eval_utils.ZEROSHOT_TASKS)
    assert summary["seven_task_average"] == pytest.approx((6 * .2 + .9) / 7)
    assert summary["tasks"]["piqa"] == {"metric": "acc_norm,none", "value": .2}
    assert "seven_task_average" not in eval_utils.summarize_lm_eval(output, ["piqa"])


def test_gsm8k_does_not_enter_seven_task_average():
    output = results()
    output["results"]["gsm8k"] = {"exact_match,strict-match": .3, "exact_match,flexible-extract": .4}
    summary = eval_utils.summarize_lm_eval(output, eval_utils.ZEROSHOT_TASKS, True)
    assert summary["seven_task_average"] == pytest.approx((6 * .2 + .9) / 7)
    assert summary["gsm8k"] == output["results"]["gsm8k"]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1, 2])
def test_invalid_accuracy_fails(value):
    output = results()
    output["results"]["piqa"]["acc_norm,none"] = value
    with pytest.raises(ValueError):
        eval_utils.summarize_lm_eval(output, ["piqa"])


def test_harness_receives_batch_limit_seeds_and_zero_shot(monkeypatch):
    evaluate, wrapper = mock_harness(monkeypatch, results())
    output = eval_utils.run_lm_eval(
        "model", "tokenizer", tasks=eval_utils.ZEROSHOT_TASKS, batch_size="auto", max_batch_size=4, limit=4, seed=3,
    )
    assert wrapper.call_args.kwargs["pretrained"] == "model"
    assert wrapper.call_args.kwargs["batch_size"] == "auto"
    assert wrapper.call_args.kwargs["max_batch_size"] == 4
    options = evaluate.call_args.kwargs
    assert options["num_fewshot"] == 0
    assert options["limit"] == 4
    assert not options["apply_chat_template"]
    assert not options["log_samples"]
    assert options["random_seed"] == options["fewshot_random_seed"] == 3
    assert "seven_task_average" in output["summary"]


def test_evaluation_errors_propagate(monkeypatch):
    evaluate, _ = mock_harness(monkeypatch, results())
    evaluate.side_effect = RuntimeError("evaluation failed")
    with pytest.raises(RuntimeError, match="evaluation failed"):
        eval_utils.run_lm_eval("model", "tokenizer")


def test_packed_loader_is_reused_and_weights_are_frozen(checkpoint, monkeypatch):
    args = entry.parse_args(["--model_path", str(checkpoint)])
    model = Mock()
    parameter = Mock()
    model.parameters.return_value = [parameter]
    loader = Mock(return_value=model)
    fp_loader = Mock()
    monkeypatch.setattr(entry, "load_quantized_model", loader)
    monkeypatch.setattr(entry.AutoModelForCausalLM, "from_pretrained", fp_loader)
    monkeypatch.setattr(entry.AutoTokenizer, "from_pretrained", Mock(return_value="tokenizer"))
    monkeypatch.setattr(entry, "align_deepseek_softmax_scale", Mock())
    assert entry.load_model(args) == (model, "tokenizer")
    assert loader.call_args.args == (str(checkpoint),)
    assert loader.call_args.kwargs["trust_remote_code"]
    model.eval.assert_called_once()
    parameter.requires_grad_.assert_called_once_with(False)
    fp_loader.assert_not_called()


def test_full_precision_loader_does_not_use_packed_loader(checkpoint, monkeypatch):
    args = entry.parse_args(["--model_path", str(checkpoint), "--is_fp"])
    model = Mock()
    model.parameters.return_value = []
    packed = Mock()
    monkeypatch.setattr(entry, "load_quantized_model", packed)
    monkeypatch.setattr(entry.AutoModelForCausalLM, "from_pretrained", Mock(return_value=model))
    monkeypatch.setattr(entry.AutoTokenizer, "from_pretrained", Mock(return_value="tokenizer"))
    monkeypatch.setattr(entry, "align_deepseek_softmax_scale", Mock())
    entry.load_model(args)
    packed.assert_not_called()


def test_formal_run_saves_results_and_logs_without_changing_model(checkpoint, monkeypatch):
    mock_harness(monkeypatch, results())
    before = {path: path.read_bytes() for path in checkpoint.iterdir()}
    monkeypatch.setattr(entry, "load_model", lambda args: (Mock(), "tokenizer"))
    entry.main(["--model_path", str(checkpoint)])
    saved = json.loads((checkpoint / "zeroshot.json").read_text())
    assert saved["experiment"]["num_fewshot"] == 0
    assert saved["experiment"]["model_path"] == str(checkpoint)
    assert "seven_task_average" in saved["summary"]
    assert "Avg (7 tasks)" in (checkpoint / "zeroshot.log").read_text()
    for path, content in before.items():
        assert path.read_bytes() == content


def test_smoke_only_prints_to_terminal(checkpoint, monkeypatch, capsys):
    mock_harness(monkeypatch, results())
    before = {path: path.read_bytes() for path in checkpoint.iterdir()}
    monkeypatch.setattr(entry, "load_model", lambda args: (Mock(), "tokenizer"))
    entry.main(["--model_path", str(checkpoint), "--tasks", "piqa", "--limit", "4"])
    assert "piqa" in capsys.readouterr().out
    assert {path: path.read_bytes() for path in checkpoint.iterdir()} == before


def test_failed_smoke_does_not_create_output(checkpoint, monkeypatch):
    mock_harness(monkeypatch, results())
    monkeypatch.setattr(entry, "load_model", Mock(side_effect=RuntimeError("loading failed")))
    with pytest.raises(RuntimeError, match="loading failed"):
        entry.main(["--model_path", str(checkpoint), "--limit", "4"])
    assert not list(checkpoint.glob("zeroshot*"))
