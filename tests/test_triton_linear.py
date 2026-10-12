from types import ModuleType, SimpleNamespace
from unittest.mock import Mock
import sys
from contextlib import nullcontext

import pytest
import torch

from gemq.inference.triton_linear import TritonQuantLinear


@pytest.mark.parametrize("bits", [1, 2, 3, 4])
def test_keeps_packed_buffers_and_metadata(monkeypatch, bits):
    packed = SimpleNamespace(W_q=torch.zeros(1, 4, dtype=torch.int32),
                             scales=torch.ones(1, 4, dtype=torch.float16),
                             zeros=torch.zeros(1, 4, dtype=torch.float16),
                             W_nbits=bits, group_size=8)
    patch = ModuleType("gemq.inference.patch")
    patch.create_gemlite_from_hqq = Mock(return_value=packed)
    monkeypatch.setitem(sys.modules, "gemq.inference.patch", patch)
    device_context = Mock(side_effect=lambda device: nullcontext())
    monkeypatch.setattr(torch.cuda, "device", device_context)
    source = SimpleNamespace(compute_dtype=torch.float16, meta={"shape": (4, 8)}, bias=None, W_q=packed.W_q)
    linear = TritonQuantLinear(source)
    assert linear.w_q is packed.W_q
    assert linear.nbits == bits and linear.group_size == 8
    assert (linear.out_features, linear.in_features) == (4, 8)
    assert not list(linear.parameters())
    assert "weight" not in dict(linear.named_buffers())
    assert not linear.state_dict()
    patch.create_gemlite_from_hqq.assert_called_once_with(source, dtype=torch.float16)
    device_context.assert_called_once_with(source.W_q.device)


def test_rejects_cpu_input_before_launch(monkeypatch):
    packed = SimpleNamespace(W_q=torch.zeros(1, 4, dtype=torch.int32),
                             scales=torch.ones(1, 4, dtype=torch.float16),
                             zeros=torch.zeros(1, 4, dtype=torch.float16), W_nbits=2, group_size=8)
    patch = ModuleType("gemq.inference.patch")
    patch.create_gemlite_from_hqq = Mock(return_value=packed)
    monkeypatch.setitem(sys.modules, "gemq.inference.patch", patch)
    kernel = ModuleType("gemq.triton_kernels.dequant_gemm")
    kernel.dequant_gemm_triton = Mock()
    monkeypatch.setitem(sys.modules, "gemq.triton_kernels.dequant_gemm", kernel)
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    linear = TritonQuantLinear(SimpleNamespace(compute_dtype=torch.float16, meta={"shape": (4, 8)},
                                               bias=None, W_q=packed.W_q))
    with pytest.raises(ValueError, match="CUDA FP16"):
        linear(torch.zeros(2, 8, dtype=torch.float16))
    kernel.dequant_gemm_triton.assert_not_called()
