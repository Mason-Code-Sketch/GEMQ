"""Use the existing packed GEMM with native fused Qwen3 expert layouts.

Conversion keeps the native Router/decoder interfaces and borrows packed
storage. No dequantized weight cache or expert-weight concatenation is built.
GPU numerical equivalence and memory/throughput require separate validation.
"""

import math

import torch
from torch import nn

from gemq.inference.fused_moe import HQQFusedMoeExperts


def _packed_gemm(x, weight, scales, zeros, nbits, group_size):
    from gemq.triton_kernels.dequant_gemm import dequant_gemm_triton

    return dequant_gemm_triton(x, weight, scales, zeros, nbits, group_size,
                               autotune=False)


class PackedExpertLinear(nn.Module):
    """Borrow GemLite's layout and evaluate with GEMQ's existing packed GEMM."""

    def __init__(self, projection):
        super().__init__()
        self.in_features = int(projection.in_features)
        self.out_features = int(projection.out_features)
        self.nbits = int(projection.W_nbits)
        self.group_size = int(projection.group_size)
        if self.nbits not in (1, 2, 3, 4):
            raise ValueError("Native fused packed experts require 1/2/3/4-bit weights")
        if projection.compute_dtype != torch.float16:
            raise ValueError("The existing packed GEMM requires FP16 compute")
        if self.group_size <= 0 or self.in_features % self.group_size:
            raise ValueError("Packed expert input width must be divisible by group size")
        mode = getattr(projection, 'W_group_mode', None)
        channel = getattr(projection, 'channel_scale_mode', None)
        if not ((mode == 3 and channel == 0) or
                (mode == 1 and channel == 1 and self.in_features == self.group_size)):
            raise ValueError("Packed GEMM requires separate unmodified scales and zero points")
        if projection.bias is not None:
            raise ValueError("Native fused experts must be bias-free")
        expected_shape = (math.ceil(self.in_features / (32 // self.nbits)),
                          self.out_features)
        if projection.W_q.dtype != torch.int32 or tuple(projection.W_q.shape) != expected_shape:
            raise ValueError("Expected 32-bit GemLite column-packed expert weights")
        for name in ('scales', 'zeros'):
            value = getattr(projection, name)
            if (not isinstance(value, torch.Tensor) or value.dtype != torch.float16
                    or tuple(value.shape) != (self.in_features // self.group_size, self.out_features)
                    or value.device != projection.W_q.device):
                raise ValueError("Expected FP16 per-group scale and zero tensors on the weight device")
        if projection.scales.stride() != projection.zeros.stride():
            raise ValueError("Packed GEMM requires identical scale/zero strides")
        for name in ('W_q', 'scales', 'zeros'):
            self.register_buffer(name, getattr(projection, name))

    def forward(self, x):
        if x.dtype != torch.float16 or x.ndim != 2 or x.shape[1] != self.in_features:
            raise ValueError("Packed expert GEMM expects a 2-D FP16 input of the original width")
        return _packed_gemm(x, self.W_q, self.scales, self.zeros,
                            self.nbits, self.group_size)


class PackedFusedMoeExperts(HQQFusedMoeExperts):
    """Retain HQQ's routing, token ordering and accumulation implementation."""

    def __init__(self, experts):
        nn.Module.__init__(self)
        if not isinstance(experts, HQQFusedMoeExperts):
            raise TypeError("Expected the current HQQ fused expert container")
        self.num_experts = experts.num_experts
        self.act_fn = experts.act_fn
        if (len(experts.gate_up_proj) != self.num_experts
                or len(experts.down_proj) != self.num_experts):
            raise ValueError("Incomplete fused expert projections")
        # Build wrappers without modifying the source; validation failure leaves it intact.
        self.gate_up_proj = nn.ModuleList(PackedExpertLinear(p) for p in experts.gate_up_proj)
        self.down_proj = nn.ModuleList(PackedExpertLinear(p) for p in experts.down_proj)
        for up, down in zip(self.gate_up_proj, self.down_proj):
            if up.out_features != 2 * down.in_features or up.in_features != down.out_features:
                raise ValueError("Fused gate/up and down dimensions do not match")
