"""Cache packed weights for repeated inference with HQQ's original arithmetic."""

import torch
from torch import nn


class CachedQuantLinear(nn.Module):
    def __init__(self, quantized_linear):
        super().__init__()
        self.register_buffer("weight", quantized_linear.dequantize(), persistent=False)
        self.register_buffer("bias", quantized_linear.bias, persistent=False)

    def forward(self, x):
        output = torch.matmul(x, self.weight.t())
        if self.bias is not None:
            output = output + self.bias
        return output


def cache_quantized_linears(module):
    from hqq.core.quantize import HQQLinear

    for name, child in module.named_children():
        if isinstance(child, HQQLinear):
            setattr(module, name, CachedQuantLinear(child))
        else:
            cache_quantized_linears(child)
