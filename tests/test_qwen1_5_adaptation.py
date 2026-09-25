"""CPU regression coverage for the Qwen1.5 fused-expert adapter."""

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from transformers import Qwen2MoeConfig
from transformers.models.qwen2_moe.modeling_qwen2_moe import Qwen2MoeSparseMoeBlock

from gemq import compute_model_stats
from gemq.quantize import (
    capture_router_finetune_state,
    get_qwen2_expert_bits,
    quantize_qwen2_expert_weights,
    validate_router_finetune_state,
)
from gemq.utils.model_utils import (
    compute_gate_stats_hook_qwen2moe,
    get_decoder_hidden_states,
    get_qwen2_num_routed_experts,
    get_router_params,
    validate_qwen2_moe_model,
)


def _make_qwen_moe_block():
    config = Qwen2MoeConfig(
        hidden_size=16,
        intermediate_size=32,
        moe_intermediate_size=8,
        shared_expert_intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        num_experts=3,
        num_experts_per_tok=2,
    )
    return Qwen2MoeSparseMoeBlock(config)


class _FlatFakeQuantizer:
    nbits = 2
    groupsize = 4

    def __init__(self, target):
        self._dequantized = torch.arange(
            target.numel(), dtype=torch.float32
        ).reshape(-1, self.groupsize)

    def quantize(self):
        scales = torch.ones(self._dequantized.shape[0], 1)
        zeros = torch.zeros_like(scales)
        return self._dequantized, scales, zeros

    def dequantize(self, quantized, scales, zeros):
        return self._dequantized


class _TinyQwenLayer(torch.nn.Module):
    def __init__(self, moe_block):
        super().__init__()
        self.mlp = moe_block


class _TinyQwenModel(torch.nn.Module):
    def __init__(self, moe_block):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([_TinyQwenLayer(moe_block)])


class _LoadedModel:
    def __init__(self):
        self.config = SimpleNamespace()

    def train(self):
        return self

    def eval(self):
        return self


class Qwen15AdaptationTest(unittest.TestCase):
    def test_fused_block_has_tensor_output_and_expert_count_on_experts(self):
        block = _make_qwen_moe_block()
        hidden_states = torch.randn(2, 3, 16)

        output = block(hidden_states)

        self.assertIsInstance(output, torch.Tensor)
        self.assertEqual(tuple(output.shape), tuple(hidden_states.shape))
        self.assertFalse(hasattr(block, "num_experts"))
        self.assertEqual(get_qwen2_num_routed_experts(block), 3)

    def test_decoder_output_extraction_rejects_missing_batch_dimension(self):
        hidden_states = torch.randn(2, 3, 4)

        self.assertIs(get_decoder_hidden_states(hidden_states, hidden_states.shape), hidden_states)
        self.assertIs(
            get_decoder_hidden_states((hidden_states, None), hidden_states.shape),
            hidden_states,
        )
        with self.assertRaisesRegex(RuntimeError, "does not match"):
            get_decoder_hidden_states(hidden_states[0], hidden_states.shape)

    def test_runtime_layout_validation_requires_the_configured_expert_count(self):
        block = _make_qwen_moe_block()
        model = _TinyQwenModel(block)
        with self.assertRaisesRegex(ValueError, "requires 60"):
            validate_qwen2_moe_model(model, "Qwen/Qwen1.5-MoE-A2.7B")

        block.experts.num_experts = 60
        block.experts.gate_up_proj = torch.nn.Parameter(
            torch.empty(60, 16, 16)
        )
        block.experts.down_proj = torch.nn.Parameter(torch.empty(60, 16, 8))
        block.gate.num_experts = 60
        block.gate.weight = torch.nn.Parameter(torch.empty(60, 16))
        validate_qwen2_moe_model(model, "Qwen/Qwen1.5-MoE-A2.7B")

    def test_layer_grads_validates_layout_before_loading_calibration(self):
        model = _LoadedModel()
        args = SimpleNamespace(
            resource_output="",
            model="unused",
            use_fast=False,
            mode="layer_grads",
            model_dtype=torch.float16,
            attn_impl="eager",
            seqlen=16,
            model_name="Qwen/Qwen1.5-MoE-A2.7B",
        )
        calls = []

        def record_validation(*_args):
            calls.append("validate")

        def record_calibration(*_args):
            calls.append("calibration")
            return []

        def record_gradients(*_args):
            calls.append("layer_grads")

        with (
            patch.object(
                compute_model_stats.AutoTokenizer,
                "from_pretrained",
                return_value=object(),
            ),
            patch.object(
                compute_model_stats.AutoModelForCausalLM,
                "from_pretrained",
                return_value=model,
            ),
            patch.object(compute_model_stats, "align_deepseek_softmax_scale"),
            patch.object(
                compute_model_stats,
                "validate_qwen2_moe_model",
                side_effect=record_validation,
            ),
            patch.object(
                compute_model_stats,
                "get_calib_loader",
                side_effect=record_calibration,
            ),
            patch.object(
                compute_model_stats,
                "compute_layer_grads",
                side_effect=record_gradients,
            ),
        ):
            compute_model_stats.main(args)

        self.assertEqual(calls, ["validate", "calibration", "layer_grads"])

    def test_gate_stats_uses_fused_weight_expert_count(self):
        block = _make_qwen_moe_block()
        hidden_states = torch.randn(2, 3, 16)
        output = block(hidden_states)
        inputs, outputs, weights, counts = [], [], [], []

        compute_gate_stats_hook_qwen2moe(
            block,
            (hidden_states,),
            output,
            inputs,
            outputs,
            weights,
            counts,
        )

        self.assertEqual(tuple(weights[0].shape), (4,))
        self.assertEqual(tuple(counts[0].shape), (4,))
        self.assertIs(outputs[0], output)

    def test_qwen_gptq_writeback_restores_fused_weight_shape(self):
        block = _make_qwen_moe_block()
        target = block.experts.gate_up_proj[0]
        quantizer = _FlatFakeQuantizer(target)

        quantize_qwen2_expert_weights(
            block,
            {0: {"gate_up_proj": quantizer}},
            SimpleNamespace(verbose=False),
        )

        self.assertEqual(tuple(target.shape), (16, 16))
        self.assertTrue(torch.equal(target.float(), quantizer._dequantized.reshape_as(target)))

    def test_qwen_bit_allocation_is_checked_against_runtime_expert_count(self):
        allocation = {0: {0: 1, 1: 2, 2: 3, 3: 3}}

        self.assertEqual(get_qwen2_expert_bits(SimpleNamespace(mixed=False, expert_wbits=2), None, 0, 3), [2, 2, 2, 2])
        self.assertEqual(get_qwen2_expert_bits(SimpleNamespace(mixed=True), allocation, 0, 3), [1, 2, 3, 3])
        with self.assertRaisesRegex(ValueError, "expected"):
            get_qwen2_expert_bits(
                SimpleNamespace(mixed=True), {0: {0: 1, 1: 2, 2: 3}}, 0, 3
            )

    def test_qwen_router_sanity_uses_parameter_identity(self):
        block = _make_qwen_moe_block()
        model = _TinyQwenModel(block)
        router_params = get_router_params(model, "Qwen/Qwen1.5-MoE-A2.7B")
        self.assertEqual([id(param) for param in router_params], [id(block.gate.weight)])

        state = capture_router_finetune_state(model, router_params)
        with torch.no_grad():
            block.gate.weight.add_(0.25)

        self.assertGreater(
            validate_router_finetune_state(model, router_params, state), 0.0
        )

        state = capture_router_finetune_state(model, router_params)
        with torch.no_grad():
            block.experts.gate_up_proj.add_(0.25)
        with self.assertRaisesRegex(RuntimeError, "Frozen parameters changed"):
            validate_router_finetune_state(model, router_params, state)


if __name__ == "__main__":
    unittest.main()
