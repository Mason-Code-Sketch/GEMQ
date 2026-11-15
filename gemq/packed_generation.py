"""Lossless FP16/BF16 expert storage and grouped CUDA generation for DGX evaluation."""
from types import FunctionType, MethodType

import torch


def pack_weight(weight):
    """Encode existing FP16/BF16 bits, with at most eight values per 128-column group."""
    if weight.dtype not in (torch.float16, torch.bfloat16) or weight.ndim != 3 or weight.shape[-1] % 128:
        raise ValueError("packed generation requires 3-D FP16/BF16 weights and 128-column groups")
    experts, outputs, inputs = weight.shape
    codes = torch.empty((experts, inputs // 2, outputs), device=weight.device, dtype=torch.uint8)
    levels = torch.empty((experts, inputs // 128, 8, outputs), device=weight.device, dtype=weight.dtype)
    for expert in range(experts):
        for start in range(0, outputs, 1024):
            rows = weight[expert, start:start + 1024]
            groups = rows.reshape(-1, 128)
            bit_values, order = groups.view(torch.int16).sort(dim=-1)
            values = bit_values.view(weight.dtype)
            changed = torch.ones_like(values, dtype=torch.int64)
            changed[:, 1:] = (values[:, 1:].view(torch.int16) != values[:, :-1].view(torch.int16)).long()
            indices = changed.cumsum(-1) - 1
            if int(indices.max()) >= 8:
                raise ValueError("expert weights exceed eight distinct FP16/BF16 values per group")
            table = torch.zeros((groups.shape[0], 8), device=weight.device, dtype=weight.dtype)
            table.scatter_(1, indices, values)
            original_indices = torch.empty_like(indices)
            original_indices.scatter_(1, order, indices)
            if not torch.equal(table.gather(1, original_indices).view(torch.int16), groups.view(torch.int16)):
                raise RuntimeError("FP16/BF16 weight bit verification failed")
            ids = original_indices.reshape(rows.shape).to(torch.uint8)
            codes[expert, :, start:start + len(rows)] = (ids[:, 0::2] | (ids[:, 1::2] << 4)).T
            levels[expert, :, :, start:start + len(rows)] = table.reshape(len(rows), inputs // 128, 8).permute(1, 2, 0)
    return codes, levels


from gemq.inference.grouped_cached import GroupedCachedExperts

def _bind_packed_forward(module, records, stats, top_k, grouped_mm, unpack_weight):
    """Keep native routing/accumulation; specialize only projection execution.

    Qwen3 and Mixtral routers use topk without replacement. Consequently an
    expert receives at most one row per input token, including skewed routing.
    The configured top-k must match the actual routed tensor on every call.
    """
    native = GroupedCachedExperts.forward

    def linear(inputs, weight, offsets):
        codes, levels, width = records[id(weight)]
        if inputs.shape[0] % top_k:
            raise ValueError("Routed row count must be divisible by top-k")
        if inputs.shape[0] <= 2048:
            stats["decode_calls"] += 1
            return grouped_mm(inputs, codes, levels, offsets,
                              max_rows_per_expert=inputs.shape[0] // top_k)
        stats["prefill_calls"] += 1
        dense = unpack_weight(codes, levels, width)
        return torch._grouped_mm(inputs, dense.transpose(1, 2), offs=offsets)

    private_forward = FunctionType(
        native.__code__, {**native.__globals__, "_linear": linear},
        native.__name__, native.__defaults__, native.__closure__,
    )

    def forward(self, hidden, indices, weights):
        if indices.ndim != 2 or indices.shape != (hidden.shape[0], top_k):
            raise ValueError("Router top-k does not match the packed launch bound")
        return private_forward(self, hidden, indices, weights)

    module.forward = MethodType(forward, module)


@torch.no_grad()
def prepare_packed_generation(wrapped):
    """Compress resident grouped experts for Qwen3/Mixtral GSM8K only."""
    model = wrapped.model
    existing = getattr(model, "_packed_generation_stats", None)
    if existing is not None:
        return existing
    stats = dict(enabled=False, backend="grouped_cached", converted=0,
                 original_bytes=0, compressed_bytes=0, decode_calls=0,
                 prefill_calls=0, weight_bits_verified=False)
    if (getattr(getattr(model, "config", None), "model_type", None)
            not in {"qwen3_moe", "mixtral"} or model.training):
        return stats
    modules = [m for m in model.modules() if isinstance(m, GroupedCachedExperts)]
    if not modules:
        return stats
    # The bound relies on the supported routers selecting distinct experts.
    top_k = getattr(model.config, "num_experts_per_tok", None)
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
        raise ValueError("Packed generation requires a positive integer router top-k")
    names_by_module = []
    for module in modules:
        if top_k > module.num_experts or module.fp32_topk:
            raise ValueError("Unsupported router contract for packed generation")
        names = ("gate_up_proj", "down_proj") if hasattr(module, "gate_up_proj") else ("gate_proj", "up_proj", "down_proj")
        for name in names:
            w = getattr(module, name)
            if w.device.type != "cuda":
                return stats
            if w.dtype not in (torch.float16, torch.bfloat16) or w.ndim != 3 or w.shape[-1] % 128:
                raise ValueError("Unsupported grouped expert weight layout")
        names_by_module.append(names)
    from gemq.packed_generation_kernels import grouped_mm, unpack_weight
    for module, names in zip(modules, names_by_module):
        records = {}
        for name in names:
            weight = getattr(module, name)
            codes, levels = pack_weight(weight)
            placeholder = torch.empty(weight.shape, device="meta", dtype=weight.dtype)
            setattr(module, name, placeholder)
            module.register_buffer("_packed_" + name, codes, persistent=False)
            module.register_buffer("_levels_" + name, levels, persistent=False)
            records[id(placeholder)] = (codes, levels, weight.shape[-1])
            stats["original_bytes"] += weight.numel() * weight.element_size()
            stats["compressed_bytes"] += codes.numel() + levels.numel() * levels.element_size()
            stats["converted"] += 1
            del weight
            torch.cuda.empty_cache()
        _bind_packed_forward(module, records, stats, top_k, grouped_mm, unpack_weight)
    stats.update(enabled=True, backend="fp16_bf16_lut_grouped_v1", weight_bits_verified=True)
    model._packed_generation_stats = stats
    return stats
