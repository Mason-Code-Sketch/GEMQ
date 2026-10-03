import pytest
import torch

from gemq.compute_model_stats import compute_moe_reconstruction_error
from gemq.utils.moe_reconstruction import Qwen3MoeReconstructionCache


class CountingExpert(torch.nn.Linear):
    def __init__(self):
        super().__init__(6, 6, bias=False)
        self.calls = 0

    def forward(self, hidden):
        self.calls += 1
        return super().forward(hidden)


class ToySparseMoe(torch.nn.Module):
    def __init__(self, normalize):
        super().__init__()
        self.gate = torch.nn.Linear(6, 9)
        self.experts = torch.nn.ModuleList([CountingExpert() for _ in range(9)])
        self.top_k = 3
        self.norm_topk_prob = normalize
        with torch.no_grad():
            self.gate.weight[8].zero_()
            self.gate.bias[8] = -100

    def forward(self, inputs):
        hidden = inputs.reshape(-1, 6)
        routing = torch.softmax(self.gate(hidden), dim=1, dtype=torch.float)
        routing, selected = routing.topk(self.top_k, dim=-1)
        if self.norm_topk_prob:
            routing /= routing.sum(dim=-1, keepdim=True)
        routing = routing.to(hidden.dtype)
        output = torch.zeros_like(hidden)
        for expert, module in enumerate(self.experts):
            slot, row = torch.where(selected.T == expert)
            if row.numel():
                output.index_add_(0, row, module(hidden[row]) * routing[row, slot, None])
        return output.reshape_as(inputs), None


@pytest.mark.parametrize("batch_size", [1, 2, 4, 8])
@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@torch.inference_mode()
def test_cached_scores_match_full_forwards(batch_size, normalize, dtype):
    torch.manual_seed(49)
    module = ToySparseMoe(normalize).to(dtype=dtype)
    inputs = torch.randn(5, 7, 6, dtype=dtype)
    outputs = torch.cat([module(inputs[start:start + batch_size])[0]
                         for start in range(0, len(inputs), batch_size)])
    weights = torch.rand_like(inputs, dtype=torch.float64)
    cache = Qwen3MoeReconstructionCache(module, inputs, outputs, weights, batch_size)
    assert cache.is_exact
    for expert, target in enumerate(module.experts):
        original = target.weight.clone()
        try:
            for factor in [0.3, 0.7, 1.0]:
                target.weight.copy_(original * factor)
                expected = compute_moe_reconstruction_error(module, inputs, outputs, weights, batch_size)
                for item in module.experts:
                    item.calls = 0
                assert cache.score(expert) == expected
                assert all(item.calls == 0 for index, item in enumerate(module.experts) if index != expert)
        finally:
            target.weight.copy_(original)
    assert cache.score(8) == 0


@torch.inference_mode()
def test_cache_rejects_inexact_baseline():
    module = ToySparseMoe(True)
    inputs = torch.ones(2, 3, 6)
    outputs = module(inputs)[0] + 1
    cache = Qwen3MoeReconstructionCache(module, inputs, outputs, torch.ones_like(inputs), 1)
    assert not cache.is_exact
    assert not cache.batches
    with pytest.raises(RuntimeError, match="differs"):
        cache.score(0)


def test_cache_rejects_invalid_batch_size():
    inputs = torch.ones(2, 3, 6)
    with pytest.raises(ValueError, match="positive"):
        Qwen3MoeReconstructionCache(ToySparseMoe(True), inputs, inputs, inputs, 0)
