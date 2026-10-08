from types import SimpleNamespace

import pytest
import torch

from gemq.inference.cached_linear import CachedQuantLinear


@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("shape", [(1, 8), (2, 3, 8)])
def test_cached_linear_preserves_packed_forward(bias, shape):
    weight = torch.arange(32, dtype=torch.float32).reshape(4, 8) / 31
    offset = torch.arange(4, dtype=torch.float32) if bias else None
    calls = []
    packed = SimpleNamespace(dequantize=lambda: calls.append(1) or weight, bias=offset)
    cached = CachedQuantLinear(packed)
    x = torch.arange(torch.tensor(shape).prod().item(), dtype=torch.float32).reshape(shape) / 7
    expected = torch.matmul(x, weight.t())
    if offset is not None:
        expected = expected + offset
    for _ in range(2):
        assert torch.equal(cached(x), expected)
    assert len(calls) == 1
    assert not list(cached.parameters())
    assert not cached.state_dict()
