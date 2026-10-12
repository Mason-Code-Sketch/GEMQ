"""Weight-only GEMQ GEMM without replacing model routing or aggregation."""

import torch
from torch import nn


class TritonQuantLinear(nn.Module):
    def __init__(self, quantized_linear):
        super().__init__()
        from gemq.inference.patch import create_gemlite_from_hqq

        if quantized_linear.compute_dtype != torch.float16:
            raise ValueError("GEMQ low-bit GEMM requires FP16 compute")
        packed = create_gemlite_from_hqq(quantized_linear, dtype=torch.float16)
        for name, value in (("w_q", packed.W_q), ("scales", packed.scales),
                            ("zeros", packed.zeros), ("bias", quantized_linear.bias)):
            self.register_buffer(name, value, persistent=False)
        self.nbits = int(packed.W_nbits)
        self.group_size = int(packed.group_size)
        self.out_features, self.in_features = map(int, quantized_linear.meta["shape"])

    def forward(self, x):
        from gemq.triton_kernels.dequant_gemm import dequant_gemm_triton

        if x.dtype != torch.float16 or not x.is_cuda:
            raise ValueError("GEMQ low-bit GEMM requires CUDA FP16 inputs")
        if x.shape[-1] != self.in_features:
            raise ValueError("Input width differs from packed weight width")
        if x.numel() == 0:
            return x.new_empty((*x.shape[:-1], self.out_features))
        output = dequant_gemm_triton(x.reshape(-1, self.in_features), self.w_q,
                                    self.scales, self.zeros, self.nbits, self.group_size)
        output = output.reshape(*x.shape[:-1], self.out_features)
        return output if self.bias is None else output + self.bias


def use_triton_linears(module):
    from hqq.core.quantize import HQQLinear

    count = 0
    for name, child in module.named_children():
        if isinstance(child, HQQLinear):
            setattr(module, name, TritonQuantLinear(child))
            count += 1
        else:
            count += use_triton_linears(child)
    return count
