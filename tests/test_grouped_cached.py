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


@pytest.mark.parametrize("model_type", ["deepseek_v2", "qwen2_moe", "mixtral"])
def test_unvalidated_model_types_are_rejected(model_type):
    with pytest.raises(ValueError, match="Qwen3"):
        backend.install_grouped_cached(SimpleNamespace(config=SimpleNamespace(model_type=model_type)))
