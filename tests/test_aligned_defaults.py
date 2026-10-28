"""Default GPTQ grid, native precision and saved seven-task entry."""
from types import SimpleNamespace
from unittest.mock import patch
import torch
import pytest
from gemq.quantizers.gptq import GPTQWeightQuantizer, MCMoeGPTQWeightQuantizer

@pytest.mark.parametrize("cls", [GPTQWeightQuantizer, MCMoeGPTQWeightQuantizer])
def test_floating_grid_search_and_dead_columns(cls):
    weight = torch.tensor([[-3., -2., 1., 8.], [1., 2., 3., 4.]], dtype=torch.float16)
    quantizer = cls(weight, nbits=2, groupsize=4, blocksize=4)
    scales, zeros, _ = quantizer.find_params(weight.float())
    assert quantizer.mse
    assert bool((zeros != zeros.round()).any())
    assert scales.dtype == (torch.float16 if cls is MCMoeGPTQWeightQuantizer else torch.float32)
    assert .21 <= float(scales[0] / (11 / 3)) <= 1
    codes, scales, zeros = quantizer.quantize()
    assert torch.count_nonzero(quantizer.dequantize(codes, scales, zeros)).item() > 0
    assert torch.equal(quantizer.W, weight)

def test_combined_seven_task_entry_loads_saved_router_checkpoint():
    from gemq.quantize import evaluate_saved_downstream
    args = SimpleNamespace(save_path="final-router-ft", save_dtype="float16", model_dtype="float16",
        downstream_tasks="piqa,arc_easy,arc_challenge,hellaswag,winogrande,mathqa,mmlu",
        lm_eval_batchsize=8, seed=0)
    with patch("gemq.eval_zeroshot.main") as evaluate:
        evaluate_saved_downstream(args)
    argv = evaluate.call_args.args[0]
    assert argv[argv.index("--model_path") + 1] == "final-router-ft"
    assert argv[argv.index("--backend") + 1] == "grouped_cached"


@pytest.mark.parametrize("cls", [GPTQWeightQuantizer, MCMoeGPTQWeightQuantizer])
@pytest.mark.parametrize("bits", [1, 2, 3])
def test_low_bit_compensation_changes_the_next_column(cls, bits):
    weight = torch.tensor([[-.25, .25]], dtype=torch.float16)
    scale = torch.ones(1, 1, dtype=torch.float16)
    zero = torch.full((1, 1), 2 ** (bits - 1) - .5, dtype=torch.float16)
    factor = torch.tensor([[1., 2.], [0., 1.]])
    quantizer = cls(weight, nbits=bits, groupsize=2, blocksize=2)
    with patch.object(quantizer, "find_params", return_value=(scale, zero, 2**bits-1)), \
         patch("gemq.quantizers.gptq.inverse_hessian_factor", return_value=factor):
        result = quantizer.dequantize(*quantizer.quantize()).reshape_as(weight)
    assert torch.equal(result, torch.tensor([[-.5, -.5]], dtype=torch.float16))
