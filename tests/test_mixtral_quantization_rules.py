"""Mixtral-only range and compensation regressions, including attention."""
import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from gemq.quantizers.gptq import GPTQWeightQuantizer, MCMoeGPTQWeightQuantizer
from test_gptq_low_bit_block import _columnwise


@pytest.mark.parametrize("cls", [GPTQWeightQuantizer, MCMoeGPTQWeightQuantizer])
@pytest.mark.parametrize("bits", [1, 2, 3, 4])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_mixtral_columnwise_reference(cls, bits, dtype):
    torch.manual_seed(82)
    w = torch.randn(4, 32).to(dtype)
    x = torch.randn(48, 32)
    solver = cls(w, nbits=bits, groupsize=8, blocksize=8, mcmoe_rules=True)
    solver.H = x.T @ x + torch.eye(32)
    expected = _columnwise(solver)
    actual = solver.quantize()
    for a, e in zip(actual, expected):
        # The FP32 solver retains its existing cast to resident storage dtype.
        torch.testing.assert_close(a, e.to(dtype) if cls is GPTQWeightQuantizer else e, rtol=0, atol=0)


@pytest.mark.parametrize("cls", [GPTQWeightQuantizer, MCMoeGPTQWeightQuantizer])
@pytest.mark.parametrize("bits", [2, 3, 4])
def test_search_covers_original_range_and_preserves_float_zero(cls, bits):
    w = torch.tensor([[-3., -2., 1., 8.], [-2., .1, .3, .5]])
    solver = cls(w, nbits=bits, groupsize=4, mcmoe_rules=True)
    seen = []
    original = solver.quantize_vector
    def capture(x, scales, zeros, maximum):
        seen.append(scales.clone())
        return original(x, scales, zeros, maximum)
    with patch.object(solver, "quantize_vector", side_effect=capture):
        scales, zeros, maximum = solver.find_params(w)
    base = (w.amax(1) - w.amin(1)) / maximum
    factors = torch.stack(seen)[:, :, 0] / base
    assert len(seen) == 101
    torch.testing.assert_close(factors[0], torch.ones(2))
    torch.testing.assert_close(factors.max(0).values, torch.full((2,), 1.1))
    torch.testing.assert_close(factors.min(0).values, torch.full((2,), .9))
    assert torch.any(zeros != zeros.round())


def test_model_dispatch_covers_uniform_and_mixed_projections():
    # Execute the actual lightweight factory without loading optional GPU libraries.
    tree = ast.parse((Path(__file__).parents[1] / "gemq/quantize.py").read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "make_gptq_quantizer")
    ns = dict(GPTQWeightQuantizer=GPTQWeightQuantizer, MCMoeGPTQWeightQuantizer=MCMoeGPTQWeightQuantizer,
              NAME_TO_MODEL={"mixtral": "mixtral", "qwen3": "qwen3"}, ModelType=SimpleNamespace(MIXTRAL="mixtral"))
    exec(compile(ast.Module(body=[node], type_ignores=[]), "factory", "exec"), ns)
    for model in ("mixtral", "qwen3"):
        for mcmoe in (True, False):
            args = SimpleNamespace(model_name=model, groupsize=8, blocksize=8, percdamp=.01,
                actorder=False, static_groups=False, mse=True, reproduce_mcmoe=mcmoe)
            for bit in (1, 2, 3, 4):
                solver = ns["make_gptq_quantizer"](torch.randn(4, 16), "projection", bit, args)
                assert solver.mcmoe_rules == (model == "mixtral")
                assert type(solver) is (MCMoeGPTQWeightQuantizer if mcmoe else GPTQWeightQuantizer)
