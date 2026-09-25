"""GPU regression coverage for Qwen1.5 fused-expert HQQ checkpoints."""

from __future__ import annotations

import tempfile
import unittest

import torch
from hqq.models.hf.base import AutoHQQHFModel
from transformers import Qwen2MoeConfig, Qwen2MoeForCausalLM
from transformers.models.qwen2_moe.modeling_qwen2_moe import Qwen2MoeSparseMoeBlock

from gemq.inference.qwen2_moe import (
    HQQQwen2MoeExperts,
    hqq_state_from_quantized_weight,
    replace_qwen2_moe_experts,
)
from gemq.utils.hf_loading import load_quantized_model


DEVICE = "cuda"
GROUP_SIZE = 8


def _make_config():
    return Qwen2MoeConfig(
        vocab_size=128,
        hidden_size=16,
        intermediate_size=32,
        moe_intermediate_size=8,
        shared_expert_intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        num_experts=3,
        num_experts_per_tok=2,
        max_position_embeddings=32,
    )


@torch.no_grad()
def _initialize(module):
    for parameter in module.parameters():
        parameter.normal_(mean=0.0, std=0.02)


@torch.no_grad()
def _pack_projection(weight, nbits):
    if weight.shape[1] % GROUP_SIZE:
        raise ValueError("Tiny Qwen test weight must divide evenly into HQQ groups.")

    quantized = torch.randint(
        0,
        2**nbits,
        (weight.numel() // GROUP_SIZE, GROUP_SIZE),
        device=weight.device,
        dtype=torch.int32,
    ).float()
    scales = torch.rand(
        quantized.shape[0], 1, device=weight.device, dtype=torch.float32
    ).add_(0.01)
    zeros = torch.rand_like(scales).mul_(2**nbits - 1)
    dequantized = ((quantized - zeros) * scales).reshape_as(weight)
    weight.copy_(dequantized.to(dtype=weight.dtype))
    state = hqq_state_from_quantized_weight(
        quantized,
        scales,
        zeros,
        tuple(weight.shape),
        nbits,
        GROUP_SIZE,
        weight.device,
    )
    return state, dequantized


@torch.no_grad()
def _pack_qwen_experts(moe_block):
    packed_states = {}
    for expert_id, nbits in enumerate((1, 2, 3)):
        expert_states = {}
        for projection_name in ("gate_up_proj", "down_proj"):
            weight = getattr(moe_block.experts, projection_name)[expert_id]
            expert_states[projection_name], _ = _pack_projection(weight, nbits)
        packed_states[expert_id] = expert_states
    return packed_states


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA for HQQ packing")
class Qwen15RealQuantTest(unittest.TestCase):
    def test_hqq_fused_experts_match_fake_quantized_forward(self):
        config = _make_config()
        moe_block = Qwen2MoeSparseMoeBlock(config).to(DEVICE, dtype=torch.float16)
        _initialize(moe_block)
        packed_states = _pack_qwen_experts(moe_block)
        reference = moe_block.experts
        packed = HQQQwen2MoeExperts.from_hf(config, reference, packed_states)

        hidden_states = torch.randn(6, config.hidden_size, device=DEVICE, dtype=torch.float16)
        top_k_index = torch.tensor(
            [[0, 1], [1, 2], [2, 0], [0, 2], [1, 0], [2, 1]],
            device=DEVICE,
        )
        top_k_weights = torch.rand(6, config.num_experts_per_tok, device=DEVICE)
        top_k_weights.div_(top_k_weights.sum(dim=-1, keepdim=True))

        expected = reference(hidden_states, top_k_index, top_k_weights)
        actual = packed(hidden_states, top_k_index, top_k_weights)

        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)

    def test_hqq_checkpoint_reloads_with_fused_experts(self):
        config = _make_config()
        model = Qwen2MoeForCausalLM(config).to(DEVICE, dtype=torch.float16).eval()
        _initialize(model)
        packed_states = _pack_qwen_experts(model.model.layers[0].mlp)
        input_ids = torch.tensor([[1, 2, 3, 4, 5, 6]], device=DEVICE)

        expected_logits = model(input_ids=input_ids).logits
        replace_qwen2_moe_experts(model, {0: packed_states})
        packed_logits = model(input_ids=input_ids).logits
        torch.testing.assert_close(packed_logits, expected_logits, rtol=2e-2, atol=2e-2)

        with tempfile.TemporaryDirectory() as checkpoint_dir:
            AutoHQQHFModel.save_quantized(model, checkpoint_dir)
            loaded = load_quantized_model(
                checkpoint_dir,
                compute_dtype=torch.float16,
                device=DEVICE,
                trust_remote_code=False,
            ).eval()
            self.assertIsInstance(
                loaded.model.layers[0].mlp.experts, HQQQwen2MoeExperts
            )
            actual_logits = loaded(input_ids=input_ids).logits
            torch.testing.assert_close(actual_logits, packed_logits, rtol=2e-2, atol=2e-2)


if __name__ == "__main__":
    unittest.main()
