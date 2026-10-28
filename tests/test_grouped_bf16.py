"""Exercise the real grouped GEMM operator in both resident compute dtypes."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from gemq.inference.cached_linear import CachedQuantLinear
from gemq.inference.grouped_cached import GroupedCachedExperts


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("layout", ["qwen3", "qwen15", "mixtral", "deepseek"])
def test_resident_grouped_gemm_preserves_expert_outputs(dtype, layout):
    torch.manual_seed(19)
    entries = nn.ModuleList()
    calls = []
    for index in range(5):
        entry = nn.Module()
        entry.act_fn = nn.functional.silu
        for name in ("gate_proj", "up_proj", "down_proj"):
            weight = (torch.randn(16, 16) / 8).to(dtype)
            def decode(w=weight):
                calls.append(1)
                return w
            setattr(entry, name, CachedQuantLinear(SimpleNamespace(dequantize=decode, bias=None)))
        entries.append(entry)
    hidden = torch.randn(13, 16).to(dtype)
    indices = torch.tensor([[3, 0, 2], [2, 3, 0], [0, 2, 3], [3, 2, 0], [2, 0, 3],
                            [0, 3, 2], [3, 0, 2], [0, 2, 3], [2, 3, 0], [0, 3, 2],
                            [3, 2, 0], [0, 2, 3], [2, 0, 3]])
    weights = torch.rand(13, 3, dtype=torch.float32 if layout in ("deepseek", "qwen15") else dtype)
    if layout == "qwen15":
        original = nn.Module(); original.act_fn = nn.functional.silu; original.num_experts = len(entries)
        original.gate_up_proj = nn.ModuleList()
        original.down_proj = nn.ModuleList()
        for entry in entries:
            combined = torch.cat([entry.gate_proj.weight, entry.up_proj.weight])
            original.gate_up_proj.append(CachedQuantLinear(SimpleNamespace(dequantize=lambda w=combined: w, bias=None)))
            original.down_proj.append(entry.down_proj)
    else:
        original = entries
        if layout == "mixtral":
            for entry in original:
                entry.w1, entry.w3, entry.w2 = entry.gate_proj, entry.up_proj, entry.down_proj
                del entry.gate_proj, entry.up_proj, entry.down_proj
    before_calls = len(calls)
    grouped = GroupedCachedExperts(original, fp32_topk=layout == "deepseek").eval()
    reference = torch.zeros_like(hidden)
    per_slot = torch.empty(13, 3, 16, dtype=dtype)
    with torch.inference_mode():
        for index in range(5):
            tokens, slots = torch.where(indices == index)
            if layout == "qwen15":
                gate, up = original.gate_up_proj[index](hidden[tokens]).chunk(2, -1)
                value = original.down_proj[index](nn.functional.silu(gate) * up)
            else:
                expert = original[index]
                names = ("w1", "w3", "w2") if layout == "mixtral" else ("gate_proj", "up_proj", "down_proj")
                value = getattr(expert, names[2])(nn.functional.silu(getattr(expert, names[0])(hidden[tokens])) * getattr(expert, names[1])(hidden[tokens]))
            if layout == "deepseek":
                per_slot[tokens, slots] = value
            else:
                reference.index_add_(0, tokens, (value * weights[tokens, slots, None]).to(dtype))
        if layout == "deepseek":
            reference = (per_slot.float() * weights.unsqueeze(-1)).sum(1).to(dtype)
        for _ in range(2):
            actual = grouped(hidden, indices, weights)
            torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    assert len(calls) == before_calls
    assert grouped.compute_dtype == dtype
    assert not grouped.state_dict()


def tiny_model(kind):
    transformers = pytest.importorskip("transformers")
    shared = dict(vocab_size=32, hidden_size=16, num_hidden_layers=2, num_attention_heads=2,
                  num_key_value_heads=2, max_position_embeddings=64)
    if kind == "deepseek_v2":
        cfg = transformers.DeepseekV2Config(**shared, intermediate_size=32, moe_intermediate_size=16,
            n_routed_experts=4, n_shared_experts=1, num_experts_per_tok=2, q_lora_rank=8,
            kv_lora_rank=4, qk_nope_head_dim=4, qk_rope_head_dim=4, v_head_dim=4,
            first_k_dense_replace=0)
    elif kind == "qwen2_moe":
        cfg = transformers.Qwen2MoeConfig(**shared, intermediate_size=32, moe_intermediate_size=16,
            shared_expert_intermediate_size=16, num_experts=4, num_experts_per_tok=2)
    elif kind == "qwen3_moe":
        cfg = transformers.Qwen3MoeConfig(**shared, intermediate_size=32, moe_intermediate_size=16,
            num_experts=4, num_experts_per_tok=2, head_dim=8)
    else:
        cfg = transformers.MixtralConfig(**shared, intermediate_size=16, num_local_experts=4,
            num_experts_per_tok=2, head_dim=8)
    cfg._attn_implementation = "eager"
    cfg._experts_implementation = "eager"
    torch.manual_seed(23)
    return transformers.AutoModelForCausalLM.from_config(cfg).to(torch.bfloat16).eval()


@pytest.mark.parametrize("kind", ["deepseek_v2", "qwen2_moe", "qwen3_moe", "mixtral"])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_bfloat16_native_fused_logits(kind, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    from gemq.inference.grouped_cached import install_grouped_cached
    model = tiny_model(kind).to(device)
    ids = torch.randint(0, 32, (2, 9), device=device)
    routers = [(name, item, item.weight.detach().clone()) for name, item in model.named_modules()
               if name.endswith(".gate")]
    with torch.inference_mode():
        reference = model(ids, use_cache=False).logits.clone()
        install_grouped_cached(model)
        actual = model(ids, use_cache=False).logits
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    for name, router, values in routers:
        assert model.get_submodule(name) is router
        assert torch.equal(router.weight, values)
