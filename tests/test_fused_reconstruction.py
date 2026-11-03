"""Compare streamed scoring against the unchanged native grouped-MM scorer."""
from unittest.mock import patch

import pytest
import torch
from transformers import Qwen3MoeConfig
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeSparseMoeBlock

from gemq.compute_model_stats import compute_fused_moe_layer_reconstruction_errors
from gemq.quantizers.rtn import MCMoeRTNWeightQuantizer
from gemq.utils.fused_reconstruction import (
    GroupedBatchContributions, streamed_fused_reconstruction_scores,
)


def fixture(dtype=torch.float32, normalize=True):
    torch.manual_seed(47)
    config = Qwen3MoeConfig(hidden_size=128, moe_intermediate_size=128,
        num_experts=4, num_experts_per_tok=2, norm_topk_prob=normalize)
    config._experts_implementation = "grouped_mm"
    block = Qwen3MoeSparseMoeBlock(config).to(dtype=dtype).eval()
    with torch.no_grad():
        for parameter in block.parameters():
            parameter.normal_(0, .02)
        inputs = torch.randn(3, 16, 128, dtype=dtype)
        reference = torch.cat([block(inputs[i:i + 1]) for i in range(3)])
    gradients = torch.randn_like(inputs)
    return block, inputs, reference, gradients


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("normalize", [False, True])
@torch.inference_mode()
def test_scores_and_weights_match_native(dtype, normalize):
    block, x, y, gradients = fixture(dtype, normalize)
    before = {k: t.clone() for k, t in block.state_dict().items()}
    cache = GroupedBatchContributions(block, x[:1], y[:1])
    assert cache.exact
    calls = []
    def quantizer(weight, nbits):
        calls.append(nbits)
        return MCMoeRTNWeightQuantizer(weight, nbits=nbits)
    expected = compute_fused_moe_layer_reconstruction_errors(
        block, x, y, gradients.double().pow(2), [1, 2, 3], 1)
    actual = streamed_fused_reconstruction_scores(block, x, y, gradients,
                                                [1, 2, 3], 1, quantizer)
    assert actual == expected
    # Two projections for each expert and bit, independent of sample count.
    assert len(calls) == 2 * 4 * 3
    assert all(torch.equal(t, block.state_dict()[k]) for k, t in before.items())


@torch.inference_mode()
def test_mismatched_reference_falls_back_to_native():
    block, x, y, gradients = fixture()
    y = y + .01
    expected = compute_fused_moe_layer_reconstruction_errors(
        block, x, y, gradients.double().pow(2), [2], 1)
    assert streamed_fused_reconstruction_scores(block, x, y, gradients,
        [2], 1, MCMoeRTNWeightQuantizer) == expected


@torch.inference_mode()
def test_error_restores_quantized_expert():
    block, x, y, gradients = fixture()
    before = {k: t.clone() for k, t in block.state_dict().items()}
    with patch.object(GroupedBatchContributions, "candidate_output",
                      side_effect=RuntimeError("probe error")):
        with pytest.raises(RuntimeError, match="probe error"):
            streamed_fused_reconstruction_scores(block, x, y, gradients,
                [2], 1, MCMoeRTNWeightQuantizer)
    assert all(torch.equal(t, block.state_dict()[k]) for k, t in before.items())
