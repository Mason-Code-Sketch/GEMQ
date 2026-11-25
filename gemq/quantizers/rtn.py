import torch
import torch.nn as nn


class RTNWeightQuantizer(nn.Module):
    def __init__(self, x, name="", nbits=4, groupsize=-1):
        super().__init__()

        self.name = name
        self.nbits = nbits
        self.groupsize = groupsize

        self.W = x.clone()

    def quantize(self, W=None):
        """
        Round-to-nearest (RTN) pseudo quantization with min-max clip range (adapted from AWQ).
        """
        nbits, groupsize = self.nbits, self.groupsize
        if W is None:
            assert self.W is not None, "No weight tensor to quantize! Please provide W or initialize self.W"
            W = self.W

        org_w_shape = W.shape
        if groupsize > 0:
            assert org_w_shape[-1] % groupsize == 0, f"w_shape: {org_w_shape[-1]} | groupsize: {groupsize}"
            W = W.reshape(-1, groupsize) # [num_groups, groupsize]

        if nbits == 0:
            num_groups = W.shape[0]
            Q = torch.zeros_like(W)
            scales = torch.ones(num_groups, 1, dtype=W.dtype, device=W.device)
            zeros = torch.zeros(num_groups, 1, dtype=W.dtype, device=W.device)
            return Q, scales, zeros
    
        if nbits == 1:
            Q = torch.where(W >= 0, 1, 0).to(W.dtype)
            scales = torch.mean(torch.abs(W), dim=1, keepdim=True) * 2
            zeros = 0.5 * torch.ones_like(scales)
            return Q, scales, zeros

        max_val = W.amax(dim=1, keepdim=True)
        min_val = W.amin(dim=1, keepdim=True)

        max_int = 2 ** nbits - 1
        min_int = 0
        scales = (max_val - min_val).clamp(min=1e-5) / max_int
        zeros = (-torch.round(min_val / scales)).clamp_(min_int, max_int)

        assert torch.isnan(scales).sum() == 0
        assert torch.isnan(W).sum() == 0

        Q = torch.clamp(torch.round(W / scales) + zeros, 0, max_int)

        return Q, scales, zeros
    
    def dequantize(self, Q, scales, zeros):
        """
        Dequantize a quantized tensor using the given quantization parameters.
        """
        return (Q - zeros) * scales
    
    def pseudo_quantize(self, W=None):
        """
        Quantize and dequantize the weight tensor.
        """
        W_q = self.dequantize(*self.quantize(W))
        if self.groupsize > 0:
            W_q = W_q.reshape(self.W.shape)
        return W_q




def _candidate(group, bit, factors):
    if bit == 1:
        scale = 2 * group.abs().mean(dim=1, keepdim=True)
        return ((group >= 0).float() - 0.5) * scale
    maximum = 2**bit - 1
    low = group.amin(dim=1, keepdim=True).clamp_max(0)
    high = group.amax(dim=1, keepdim=True).clamp_min(0)
    high = torch.where((low == 0) & (high == 0), maximum, high)
    best_error = torch.full_like(low, float("inf"))
    best_scale = (high - low) / maximum
    best_zero = -low / best_scale
    for factor in factors:
        lower = factor * low
        scale = (factor * high - lower) / maximum
        zero = -lower / scale
        levels = (group / scale + zero).round().clamp_(0, maximum)
        error = ((levels - zero) * scale - group).abs_().pow_(2.4).sum(dim=1, keepdim=True)
        improved = error < best_error
        best_error = torch.where(improved, error, best_error)
        best_scale = torch.where(improved, scale, best_scale)
        best_zero = torch.where(improved, zero, best_zero)
    levels = (group / best_scale + best_zero).round().clamp_(0, maximum)
    return (levels - best_zero) * best_scale


@torch.no_grad()
def normal_quantize(w, blocksize=128, wbit=2, *, mcmoe_rules=False):
    """Search each row and input-column group independently, including a short tail."""
    if wbit == 0:
        return torch.zeros_like(w)
    if blocksize <= 0 or not 1 <= wbit <= 8 or w.ndim != 2:
        raise ValueError("RTN requires a matrix, a positive group size and 1..8 bits")
    factors = [1 - i / 100 for i in range(80)]
    if mcmoe_rules and wbit > 1:
        factors = torch.cat((w.new_ones(1, dtype=torch.float32),
                             torch.linspace(1, 1.1, 51, device=w.device, dtype=torch.float32)[1:],
                             torch.linspace(1, 0.9, 51, device=w.device, dtype=torch.float32)[1:]))
    rows, columns = w.shape
    full = columns - columns % blocksize
    result = torch.empty_like(w)
    chunk = max(1, (8 * 1024 * 1024) // columns)
    for start in range(0, rows, chunk):
        for left, right, width in ((0, full, blocksize), (full, columns, columns - full)):
            if left == right:
                continue
            source = w[start:start + chunk, left:right]
            group = source.reshape(-1, width).float()
            result[start:start + chunk, left:right] = _candidate(group, wbit, factors).reshape_as(source)
    return result


class MCMoeRTNWeightQuantizer(nn.Module):
    def __init__(self, x, nbits, blocksize=128, *, mcmoe_rules=False):
        super().__init__()
        self.x = x.clone()
        self.nbits = nbits
        self.blocksize = blocksize
        self.mcmoe_rules = mcmoe_rules

    normal_quantize = staticmethod(normal_quantize)

    def quantize(self):
        return self.normal_quantize(self.x, self.blocksize, self.nbits, mcmoe_rules=self.mcmoe_rules)
