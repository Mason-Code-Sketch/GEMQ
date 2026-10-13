from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch import nn

from gemq.inference.cached_linear import CachedQuantLinear
from gemq.inference import grouped_cached as backend


def experts():
    torch.manual_seed(0)
    result = nn.ModuleList()
    for _ in range(5):
        expert = nn.Module()
        expert.act_fn = nn.functional.silu
        for name, shape in (("gate_proj", (6, 8)), ("up_proj", (6, 8)), ("down_proj", (8, 6))):
            weight = torch.randn(shape).half()
            setattr(expert, name, CachedQuantLinear(SimpleNamespace(dequantize=lambda w=weight: w, bias=None)))
        result.append(expert)
    return result


def linear_groups(inputs, weights, offsets):
    outputs, start = [], 0
    for expert, end in enumerate(offsets.tolist()):
        outputs.append(inputs[start:end] @ weights[expert].T)
        start = end
    return torch.cat(outputs)


def test_fp16_unsorted_routes_and_aggregation_are_exact():
    original = experts()
    grouped = backend.GroupedCachedExperts(original).eval()
    hidden = torch.randn(7, 8).half()
    indices = torch.tensor([[3, 0, 2], [2, 3, 0], [0, 2, 3], [3, 2, 0],
                            [2, 0, 3], [0, 3, 2], [3, 0, 2]])
    weights = torch.rand(7, 3).half()
    reference = torch.zeros_like(hidden)
    with torch.no_grad():
        for expert_id, expert in enumerate(original):
            slots, tokens = torch.where(indices.T == expert_id)
            value = expert.down_proj(expert.act_fn(expert.gate_proj(hidden[tokens])) * expert.up_proj(hidden[tokens]))
            reference.index_add_(0, tokens, value * weights[tokens, slots, None])
        with patch.object(backend, "_linear", side_effect=linear_groups):
            actual = grouped(hidden, indices, weights)
    assert torch.equal(actual, reference)
    assert not grouped.state_dict()
    for name in ("gate_proj", "up_proj", "down_proj"):
        assert torch.equal(getattr(grouped, name), torch.stack([getattr(e, name).weight for e in original]))


@pytest.mark.parametrize("training", [False, True])
def test_gradients_and_training_are_rejected(training):
    grouped = backend.GroupedCachedExperts(experts())
    grouped.train(training)
    with pytest.raises(RuntimeError, match="evaluation"):
        grouped(torch.randn(2, 8).half(), torch.zeros(2, 1, dtype=torch.long), torch.ones(2, 1).half())


def test_router_identity_and_accelerate_wrapper_are_preserved():
    block = nn.Module()
    block.experts = experts()
    block.gate = nn.Linear(8, 5).half()
    block.top_k, block.norm_topk_prob = 3, True
    block._hf_hook = object()
    block._old_forward = block.forward
    wrapper, router = block.forward, block.gate
    model = SimpleNamespace(config=SimpleNamespace(model_type="qwen3_moe"),
                            model=SimpleNamespace(layers=[SimpleNamespace(mlp=block)]))
    backend.install_grouped_cached(model)
    assert block.forward == wrapper
    assert block._old_forward.__func__ == backend.grouped_qwen3_forward
    assert block.gate is router


def test_invalid_later_layer_fails_before_modification():
    first = SimpleNamespace(experts=experts())
    second = SimpleNamespace(experts=experts())
    second.experts[0].gate_proj.weight = second.experts[0].gate_proj.weight.float()
    model = SimpleNamespace(config=SimpleNamespace(model_type="qwen3_moe"),
                            model=SimpleNamespace(layers=[SimpleNamespace(mlp=b) for b in (first, second)]))
    with pytest.raises(ValueError, match="FP16"):
        backend.install_grouped_cached(model)
    assert isinstance(first.experts, nn.ModuleList)


@pytest.mark.parametrize("model_type", ["llama", "qwen2", "qwen3"])
def test_unvalidated_model_types_are_rejected(model_type):
    with pytest.raises(ValueError, match="Unsupported"):
        backend.install_grouped_cached(SimpleNamespace(config=SimpleNamespace(model_type=model_type)))


def test_deepseek_preserves_fp32_topk_sum_and_final_fp16_cast():
    original = experts()
    grouped = backend.GroupedCachedExperts(original, fp32_topk=True).eval()
    hidden = torch.randn(7, 8).half()
    indices = torch.tensor([[3, 0, 2]] * 7)
    weights = torch.rand(7, 3, dtype=torch.float32)
    values = torch.empty(7, 3, 8, dtype=torch.float16)
    with torch.no_grad():
        for expert_id, expert in enumerate(original):
            tokens, slots = torch.where(indices == expert_id)
            values[tokens, slots] = expert.down_proj(expert.act_fn(expert.gate_proj(hidden[tokens])) * expert.up_proj(hidden[tokens]))
        reference = (values.float() * weights.unsqueeze(-1)).sum(1).half()
        with patch.object(backend, "_linear", side_effect=linear_groups):
            assert torch.equal(grouped(hidden, indices, weights), reference)


def test_qwen1_fused_projection_preserves_fp32_weighted_contribution_rounding():
    original = nn.Module()
    original.num_experts, original.act_fn = 5, nn.functional.silu
    for name, shape in (("gate_up_proj", (12, 8)), ("down_proj", (8, 6))):
        projections = nn.ModuleList()
        for _ in range(5):
            weight = torch.randn(shape).half()
            projections.append(CachedQuantLinear(SimpleNamespace(dequantize=lambda w=weight: w, bias=None)))
        setattr(original, name, projections)
    grouped = backend.GroupedCachedExperts(original).eval()
    hidden, indices, weights = torch.randn(7, 8).half(), torch.tensor([[3, 0, 2]] * 7), torch.rand(7, 3)
    reference = torch.zeros_like(hidden)
    with torch.no_grad():
        for expert_id in range(5):
            slots, tokens = torch.where(indices.T == expert_id)
            gate, up = original.gate_up_proj[expert_id](hidden[tokens]).chunk(2, -1)
            value = original.down_proj[expert_id](original.act_fn(gate) * up)
            reference.index_add_(0, tokens, (value * weights[tokens, slots, None]).half())
        with patch.object(backend, "_linear", side_effect=linear_groups):
            assert torch.equal(grouped(hidden, indices, weights), reference)


def test_mixtral_projection_mapping_preserves_values():
    original = experts()
    for expert in original:
        expert.w1, expert.w3, expert.w2 = expert.gate_proj, expert.up_proj, expert.down_proj
        del expert.gate_proj, expert.up_proj, expert.down_proj
    grouped = backend.GroupedCachedExperts(original)
    assert torch.equal(grouped.gate_proj, torch.stack([e.w1.weight for e in original]))
    assert torch.equal(grouped.up_proj, torch.stack([e.w3.weight for e in original]))
    assert torch.equal(grouped.down_proj, torch.stack([e.w2.weight for e in original]))


def test_conversion_does_not_retain_a_gradient_graph_or_old_weights():
    block = nn.Module()
    block.experts = experts()
    for expert in block.experts:
        for name in ("gate_proj", "up_proj", "down_proj"):
            getattr(expert, name).weight.requires_grad_(True)
    model = SimpleNamespace(config=SimpleNamespace(model_type="qwen3_moe"),
                            model=SimpleNamespace(layers=[SimpleNamespace(mlp=block)]))
    backend.install_grouped_cached(model)
    assert all(not tensor.requires_grad and tensor.grad_fn is None for tensor in block.experts.buffers())
