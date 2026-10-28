"""Native command controls are applied before expensive execution."""
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from gemq import quantize
from gemq.quantizers.gptq import GPTQWeightQuantizer, MCMoeGPTQWeightQuantizer


def parse(extra):
    with patch("sys.argv", ["quantize", "--model", "unused", "--model_name", "Qwen/Qwen3-30B-A3B", *extra]):
        return quantize.parse_args()


@pytest.mark.parametrize("mcmoe", [False, True])
def test_mse_and_sequence_cache_settings_reach_execution(mcmoe):
    for mse in (False, True):
        args = parse(["--mse" if mse else "--no-mse", "--seqlen", "64", "--disable_cache",
                      "--groupsize", "8", "--blocksize", "4"]
                     + (["--reproduce_mcmoe"] if mcmoe else []))
        model = SimpleNamespace(config=SimpleNamespace(use_cache=True))
        quantize.configure_quantization_model(model, args)
        assert model.seqlen == 64
        assert not model.config.use_cache
        solver = quantize.make_gptq_quantizer(torch.randn(3, 16).half(), "projection", 2, args)
        assert solver.mse is mse
        assert solver.groupsize == 8 and solver.blocksize == 4


@pytest.mark.parametrize("flag", ["--actorder", "--static_groups"])
def test_unsupported_cli_controls_fail_during_parsing(flag):
    with pytest.raises(SystemExit):
        parse([flag])


@pytest.mark.parametrize("cls", [GPTQWeightQuantizer, MCMoeGPTQWeightQuantizer])
@pytest.mark.parametrize("option", ["actorder", "static_groups"])
def test_unsupported_constructor_controls_cannot_be_silently_ignored(cls, option):
    with pytest.raises(ValueError, match="does not implement"):
        cls(torch.ones(2, 8), groupsize=4, **{option: True})


def test_saved_evaluation_uses_the_actual_model_precision():
    args = SimpleNamespace(save_path="final", save_dtype="float16", model_dtype="auto",
        downstream_tasks="piqa", lm_eval_batchsize=8, seed=0)
    with patch("gemq.eval_zeroshot.main") as evaluate:
        quantize.evaluate_saved_downstream(args, model_dtype="bfloat16")
    argv = evaluate.call_args.args[0]
    assert argv[argv.index("--model_dtype") + 1] == "bfloat16"
    assert argv[argv.index("--backend") + 1] == "grouped_cached"


def test_packed_checkpoint_does_not_silently_ignore_save_dtype():
    args = SimpleNamespace(real_quant=True, eval_real_quant=False,
        fake_save_path="", save_dtype="bfloat16")
    with pytest.raises(ValueError, match="save_dtype applies"):
        quantize.validate_checkpoint_paths(args)
