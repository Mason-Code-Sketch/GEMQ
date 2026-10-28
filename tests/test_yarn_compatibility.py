"""CPU regressions for YaRN scaling across supported native MoE models."""

import math
import unittest

import torch
from transformers import (
    AutoModelForCausalLM, DeepseekV2Config, MixtralConfig,
    Qwen2MoeConfig, Qwen3MoeConfig,
)

from gemq.utils.hf_loading import align_deepseek_softmax_scale as align


def tiny_config(kind):
    common = dict(vocab_size=32, hidden_size=16, intermediate_size=32,
                  num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
                  max_position_embeddings=64)
    if kind == "deepseek_v2":
        return DeepseekV2Config(**common, moe_intermediate_size=8,
            n_routed_experts=4, n_shared_experts=1, num_experts_per_tok=2,
            q_lora_rank=8, kv_lora_rank=4, qk_nope_head_dim=4,
            qk_rope_head_dim=4, v_head_dim=4, first_k_dense_replace=1,
            rope_parameters=dict(rope_type="yarn", factor=40,
                mscale=.707, mscale_all_dim=.707,
                original_max_position_embeddings=64))
    if kind == "qwen2_moe":
        return Qwen2MoeConfig(**common, moe_intermediate_size=8,
            shared_expert_intermediate_size=8, num_experts=4, num_experts_per_tok=2)
    if kind == "qwen3_moe":
        return Qwen3MoeConfig(**common, head_dim=8, moe_intermediate_size=8,
            num_experts=4, num_experts_per_tok=2)
    return MixtralConfig(**common, num_local_experts=4, num_experts_per_tok=2)


class YarnCompatibilityTests(unittest.TestCase):
    def test_current_native_models_keep_their_scales_and_logits(self):
        for kind in ("deepseek_v2", "qwen2_moe", "qwen3_moe", "mixtral"):
            with self.subTest(model=kind):
                torch.manual_seed(17)
                model = AutoModelForCausalLM.from_config(
                    tiny_config(kind), attn_implementation="eager",
                    experts_implementation="eager",
                ).eval()
                tokens = torch.tensor([[1, 7, 3, 9, 5]])
                with torch.no_grad():
                    expected = model(tokens, use_cache=False).logits
                scales = [layer.self_attn.scaling for layer in model.model.layers]
                if kind == "deepseek_v2":
                    target = 8 ** -.5 * (1 + .1 * .707 * math.log(40)) ** 2
                    self.assertAlmostEqual(scales[0], target)
                align(model)
                align(model)
                self.assertEqual(scales, [layer.self_attn.scaling for layer in model.model.layers])
                with torch.no_grad():
                    actual = model(tokens, use_cache=False).logits
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_legacy_missing_scale_is_restored_once(self):
        model = AutoModelForCausalLM.from_config(
            tiny_config("deepseek_v2"), attn_implementation="eager",
            experts_implementation="eager",
        )
        for layer in model.model.layers:
            layer.self_attn.scaling = layer.self_attn.qk_head_dim ** -.5
        align(model)
        scales = [layer.self_attn.scaling for layer in model.model.layers]
        align(model)
        for layer, scale in zip(model.model.layers, scales):
            target = layer.self_attn.qk_head_dim ** -.5 * (1 + .1 * .707 * math.log(40)) ** 2
            self.assertAlmostEqual(scale, target)
            self.assertEqual(layer.self_attn.scaling, scale)
