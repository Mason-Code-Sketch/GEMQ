"""Evaluation-only grouped experts preserving GEMQ Qwen3 FP16 aggregation."""

from types import MethodType

import torch
from torch import nn

from gemq.inference.cached_linear import CachedQuantLinear


def _linear(inputs, weights, offsets):
    return torch._grouped_mm(inputs, weights.transpose(1, 2), offs=offsets)


class GroupedCachedExperts(nn.Module):
    def __init__(self, experts):
        super().__init__()
        for name in ("gate_proj", "up_proj", "down_proj"):
            projections = [getattr(expert, name) for expert in experts]
            if any(not isinstance(proj, CachedQuantLinear) or proj.bias is not None
                   or proj.weight.dtype != torch.float16 for proj in projections):
                raise ValueError("Expected cached FP16 bias-free Qwen3 projections")
            self.register_buffer(name, torch.stack([proj.weight for proj in projections]), persistent=False)
        self.act_fn = experts[0].act_fn
        self.num_experts = len(experts)

    def forward(self, hidden, indices, weights):
        if self.training or torch.is_grad_enabled():
            raise RuntimeError("Grouped cached experts are restricted to gradient-free evaluation")
        if hidden.dtype != torch.float16 or weights.dtype != torch.float16:
            raise ValueError("Grouped cached experts require original FP16 arithmetic")
        tokens, top_k = indices.shape
        # Match torch.where(mask[expert]) enumeration: top-k slot, then token.
        ids = indices.T.contiguous().reshape(-1)
        permutation = ids.argsort(stable=True)
        token_ids, slots = permutation % tokens, permutation // tokens
        offsets = torch.bincount(ids, minlength=self.num_experts).cumsum(0, dtype=torch.int32)
        selected = hidden[token_ids]
        gate = _linear(selected, self.gate_proj, offsets)
        up = _linear(selected, self.up_proj, offsets)
        values = _linear(self.act_fn(gate) * up, self.down_proj, offsets)
        values = values * weights[token_ids, slots, None]
        restored = torch.empty_like(values)
        restored[permutation] = values
        contributions = restored.view(top_k, tokens, -1).transpose(0, 1)
        order = indices.argsort(dim=1)
        contributions = contributions.gather(1, order.unsqueeze(-1).expand_as(contributions))
        output = torch.zeros_like(hidden)
        for slot in range(top_k):
            output.add_(contributions[:, slot])
        return output


def grouped_qwen3_forward(self, hidden_states):
    shape = hidden_states.shape
    hidden = hidden_states.reshape(-1, shape[-1])
    router_logits = self.gate(hidden)
    weights = torch.softmax(router_logits, dim=1, dtype=torch.float32)
    weights, indices = torch.topk(weights, self.top_k, dim=-1)
    if self.norm_topk_prob:
        weights = weights / weights.sum(dim=-1, keepdim=True)
    output = self.experts(hidden, indices, weights.to(hidden.dtype))
    return output.reshape(shape), router_logits


def install_grouped_cached(model):
    """Replace only Qwen3 evaluation expert modules, leaving routers untouched."""
    if model.config.model_type != "qwen3_moe":
        raise ValueError("Grouped cached evaluation currently requires Qwen3 MoE")
    blocks = [layer.mlp for layer in model.model.layers]
    if any(not isinstance(block.experts, nn.ModuleList) for block in blocks):
        raise ValueError("Expected GEMQ ModuleList experts before grouped conversion")
    # Validate every layer before modifying any layer.
    for block in blocks:
        for expert in block.experts:
            for name in ("gate_proj", "up_proj", "down_proj"):
                proj = getattr(expert, name)
                if not isinstance(proj, CachedQuantLinear) or proj.bias is not None or proj.weight.dtype != torch.float16:
                    raise ValueError("Expected cached FP16 bias-free Qwen3 projections")
    for block in blocks:
        block.experts = GroupedCachedExperts(block.experts).eval()
        attribute = "_old_forward" if hasattr(block, "_hf_hook") else "forward"
        setattr(block, attribute, MethodType(grouped_qwen3_forward, block))
