"""Grouped cached evaluation preserving each model's original arithmetic."""

from types import MethodType

import torch
from torch import nn

from gemq.inference.cached_linear import CachedQuantLinear


def _linear(inputs, weights, offsets):
    return torch._grouped_mm(inputs, weights.transpose(1, 2), offs=offsets)


class GroupedCachedExperts(nn.Module):
    def __init__(self, experts, fp32_topk=False):
        super().__init__()
        self.fp32_topk = fp32_topk
        fused = not isinstance(experts, nn.ModuleList)
        names = (("gate_up_proj", "down_proj") if fused else
                 ("w1", "w3", "w2") if hasattr(experts[0], "w1") else
                 ("gate_proj", "up_proj", "down_proj"))
        for name in names:
            projections = list(getattr(experts, name)) if fused else [getattr(expert, name) for expert in experts]
            if any(not isinstance(proj, CachedQuantLinear) or proj.bias is not None
                   or proj.weight.dtype != torch.float16 for proj in projections):
                raise ValueError("Expected cached FP16 bias-free expert projections")
            target = {"w1": "gate_proj", "w3": "up_proj", "w2": "down_proj"}.get(name, name)
            self.register_buffer(target, torch.stack([proj.weight for proj in projections]), persistent=False)
        self.act_fn = experts.act_fn if fused else experts[0].act_fn
        self.num_experts = experts.num_experts if fused else len(experts)

    def __len__(self):
        return self.num_experts

    def forward(self, hidden, indices, weights):
        if self.training or torch.is_grad_enabled():
            raise RuntimeError("Grouped cached experts are restricted to gradient-free evaluation")
        if hidden.dtype != torch.float16:
            raise ValueError("Grouped cached experts require original FP16 arithmetic")
        tokens, top_k = indices.shape
        # Match torch.where(mask[expert]) enumeration: top-k slot, then token.
        ids = indices.reshape(-1) if self.fp32_topk else indices.T.contiguous().reshape(-1)
        permutation = ids.argsort() if self.fp32_topk else ids.argsort(stable=True)
        token_ids, slots = ((permutation // top_k, permutation % top_k) if self.fp32_topk else
                           (permutation % tokens, permutation // tokens))
        offsets = torch.bincount(ids, minlength=self.num_experts).cumsum(0, dtype=torch.int32)
        selected = hidden[token_ids]
        if hasattr(self, "gate_up_proj"):
            gate, up = _linear(selected, self.gate_up_proj, offsets).chunk(2, dim=-1)
        else:
            gate = _linear(selected, self.gate_proj, offsets)
            up = _linear(selected, self.up_proj, offsets)
        values = _linear(self.act_fn(gate) * up, self.down_proj, offsets)
        if self.fp32_topk:
            restored = torch.empty_like(values)
            restored[permutation] = values
            return (restored.view(tokens, top_k, -1).to(weights.dtype)
                    * weights.unsqueeze(-1)).sum(dim=1).to(hidden.dtype)
        values = values * weights[token_ids, slots, None]
        restored = torch.empty_like(values)
        restored[permutation] = values
        contributions = restored.view(top_k, tokens, -1).transpose(0, 1).to(hidden.dtype)
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
    if getattr(self, "norm_topk_prob", True):
        weights = weights / weights.sum(dim=-1, keepdim=True)
    output = self.experts(hidden, indices, weights.to(hidden.dtype))
    if hasattr(self, "shared_expert"):
        shared = self.shared_expert(hidden)
        shared = torch.sigmoid(self.shared_expert_gate(hidden)) * shared
        output = output + shared
    return output.reshape(shape), router_logits


def grouped_deepseek_infer(self, hidden, indices, weights):
    return self.experts(hidden, indices, weights)


def install_grouped_cached(model):
    """Replace routed experts only; preserve routers, shared experts and attention."""
    kind = model.config.model_type
    if kind not in ("qwen3_moe", "qwen2_moe", "mixtral", "deepseek_v2"):
        raise ValueError("Unsupported grouped cached model")
    blocks = [getattr(layer, "block_sparse_moe", None) or getattr(layer, "mlp", None)
              for layer in model.model.layers]
    blocks = [block for block in blocks if hasattr(block, "experts")]
    if not blocks:
        raise ValueError("No routed expert blocks found")
    if kind == "deepseek_v2" and any(
        not hasattr(block, "moe_infer") or getattr(block, "ep_size", 1) != 1 for block in blocks
    ):
        raise ValueError("DeepSeek grouped evaluation requires original single-rank moe_infer")
    # Validate every layer before modifying any layer.
    for block in blocks:
        experts = block.experts
        fused = not isinstance(experts, nn.ModuleList)
        names = (("gate_up_proj", "down_proj") if fused else
                 ("w1", "w3", "w2") if hasattr(experts[0], "w1") else
                 ("gate_proj", "up_proj", "down_proj"))
        for name in names:
            projections = list(getattr(experts, name)) if fused else [getattr(expert, name) for expert in experts]
            for proj in projections:
                if not isinstance(proj, CachedQuantLinear) or proj.bias is not None or proj.weight.dtype != torch.float16:
                    raise ValueError("Expected cached FP16 bias-free expert projections")
    devices = set()
    for block in blocks:
        fused = not isinstance(block.experts, nn.ModuleList)
        block.experts = GroupedCachedExperts(block.experts, fp32_topk=kind == "deepseek_v2").eval()
        devices.add(block.experts.down_proj.device)
        if kind == "deepseek_v2":
            block.moe_infer = MethodType(grouped_deepseek_infer, block)
        elif not fused:
            attribute = "_old_forward" if hasattr(block, "_hf_hook") else "forward"
            setattr(block, attribute, MethodType(grouped_qwen3_forward, block))
    # Release conversion-time allocator leftovers once, never during forwards.
    for device in devices:
        if device.type == "cuda":
            with torch.cuda.device(device):
                torch.cuda.empty_cache()
