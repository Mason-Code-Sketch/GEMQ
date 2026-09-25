"""HQQ-backed routed experts for Qwen1.5-MoE real quantization."""

from __future__ import annotations

import torch
import torch.nn as nn
from hqq.core.quantize import HQQLinear
from hqq.models.hf.base import AutoHQQHFModel
from transformers.activations import ACT2FN

from gemq.utils.quant_utils import create_hqq_linear_from_quantized_weights


QWEN2_HQQ_EXPERTS_CONFIG_KEY = "gemq_qwen2_hqq_experts"


def hqq_state_to_cpu(module):
    """Detach an HQQLinear state so it can be retained during router fine-tuning."""
    return {
        key: value.detach().cpu().clone() if isinstance(value, torch.Tensor) else value
        for key, value in module.state_dict().items()
    }


def hqq_state_from_quantized_weight(
    quantized, scales, zeros, shape, nbits, groupsize, device
):
    """Pack GPTQ codes into an HQQ state dictionary without retaining a GPU module."""
    hqq_linear = create_hqq_linear_from_quantized_weights(
        quantized,
        scales,
        zeros,
        shape,
        nbits,
        groupsize,
        device=device,
    )
    try:
        return hqq_state_to_cpu(hqq_linear)
    finally:
        del hqq_linear


def hqq_linear_from_state(state, device, compute_dtype):
    """Restore one HQQLinear from a state dictionary produced by HQQ serialization."""
    hqq_linear = HQQLinear(
        linear_layer=None,
        quant_config=None,
        compute_dtype=compute_dtype,
        device=device,
    )
    hqq_linear.load_state_dict(state)
    return hqq_linear


class HQQQwen2MoeExperts(nn.Module):
    """Qwen2MoeExperts with one HQQLinear pair per routed expert."""

    def __init__(self, config, device="meta"):
        super().__init__()
        self.num_experts = int(config.num_experts)
        self.act_fn = ACT2FN[config.hidden_act]
        self.gate_up_proj = nn.ModuleList(
            [
                nn.Linear(
                    config.hidden_size,
                    2 * config.moe_intermediate_size,
                    bias=False,
                    device=device,
                )
                for _ in range(self.num_experts)
            ]
        )
        self.down_proj = nn.ModuleList(
            [
                nn.Linear(
                    config.moe_intermediate_size,
                    config.hidden_size,
                    bias=False,
                    device=device,
                )
                for _ in range(self.num_experts)
            ]
        )

    @classmethod
    @torch.no_grad()
    def from_hf(cls, config, hf_experts, packed_states):
        """Build packed experts from Qwen's fused tensors and retained HQQ states."""
        packed_experts = cls(config)
        expected_ids = set(range(packed_experts.num_experts))
        unexpected_ids = set(packed_states).difference(expected_ids)
        if unexpected_ids:
            raise ValueError(
                f"Qwen1.5 packed expert states contain invalid ids: {sorted(unexpected_ids)}."
            )

        for expert_id in range(packed_experts.num_experts):
            for projection_name in ("gate_up_proj", "down_proj"):
                source_weight = getattr(hf_experts, projection_name)[expert_id]
                state = packed_states.get(expert_id, {}).get(projection_name)
                if state is None:
                    packed_linear = nn.Linear(
                        source_weight.shape[1],
                        source_weight.shape[0],
                        bias=False,
                        device=source_weight.device,
                        dtype=source_weight.dtype,
                    )
                    packed_linear.weight.copy_(source_weight)
                else:
                    device = source_weight.device if source_weight.is_cuda else "cuda"
                    packed_linear = hqq_linear_from_state(
                        state,
                        device=device,
                        compute_dtype=source_weight.dtype,
                    )
                getattr(packed_experts, projection_name)[expert_id] = packed_linear

        return packed_experts

    def forward(self, hidden_states, top_k_index, top_k_weights):
        """Match transformers' Qwen2MoeExperts routing and aggregation semantics."""
        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(
                top_k_index, num_classes=self.num_experts
            ).permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_index in expert_hit:
            expert_id = int(expert_index.item())
            top_k_pos, token_index = torch.where(expert_mask[expert_id])
            current_state = hidden_states[token_index]
            gate, up = self.gate_up_proj[expert_id](current_state).chunk(2, dim=-1)
            current_hidden_states = self.act_fn(gate) * up
            current_hidden_states = self.down_proj[expert_id](current_hidden_states)
            current_hidden_states = (
                current_hidden_states * top_k_weights[token_index, top_k_pos, None]
            )
            final_hidden_states.index_add_(
                0, token_index, current_hidden_states.to(final_hidden_states.dtype)
            )

        return final_hidden_states


def _max_qwen2_expert_reconstruction_error(hf_experts, packed_experts):
    max_error = 0.0
    for expert_id in range(packed_experts.num_experts):
        for projection_name in ("gate_up_proj", "down_proj"):
            packed_linear = getattr(packed_experts, projection_name)[expert_id]
            if not isinstance(packed_linear, HQQLinear):
                continue
            source_weight = getattr(hf_experts, projection_name)[expert_id]
            restored_weight = packed_linear.dequantize().reshape_as(source_weight)
            avg_error = (source_weight - restored_weight).abs().mean().item()
            max_error = max(max_error, avg_error)
    return max_error


def replace_qwen2_moe_experts(model, packed_states):
    """Replace each Qwen fused expert tensor with HQQ-backed routed experts."""
    if not packed_states:
        return 0.0

    layers = model.model.layers
    invalid_layer_ids = set(packed_states).difference(range(len(layers)))
    if invalid_layer_ids:
        raise ValueError(
            f"Qwen1.5 packed expert states contain invalid layers: {sorted(invalid_layer_ids)}."
        )

    max_error = 0.0
    for layer_id, layer_states in packed_states.items():
        moe_block = layers[layer_id].mlp
        hf_experts = moe_block.experts
        packed_experts = HQQQwen2MoeExperts.from_hf(
            model.config, hf_experts, layer_states
        )
        max_error = max(
            max_error,
            _max_qwen2_expert_reconstruction_error(hf_experts, packed_experts),
        )
        moe_block.experts = packed_experts

    setattr(model.config, QWEN2_HQQ_EXPERTS_CONFIG_KEY, True)
    print(f"Max Qwen routed-expert packing reconstruction error: {max_error}")
    return max_error


def install_qwen2_moe_hqq_placeholders(model):
    """Install meta-device Qwen HQQ expert placeholders before HQQ state loading."""
    for layer in model.model.layers:
        moe_block = layer.mlp
        if not hasattr(moe_block, "experts"):
            raise TypeError("Qwen1.5 decoder layer is missing its routed-expert module.")
        moe_block.experts = HQQQwen2MoeExperts(model.config)


class Qwen2MoeHQQModel(AutoHQQHFModel):
    """HQQ loader that reconstructs Qwen packed routed-expert module names."""

    @classmethod
    def create_model(cls, save_dir, kwargs):
        model = super().create_model(save_dir, kwargs)
        install_qwen2_moe_hqq_placeholders(model)
        return model
