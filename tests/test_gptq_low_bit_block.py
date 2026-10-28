"""Compare block quantization with MC-MoE's columnwise arithmetic."""

import pytest
import torch

from gemq.quantizers.gptq import MCMoeGPTQWeightQuantizer


def _columnwise(quantizer):
    weight = quantizer.W.float().clone()
    hessian = quantizer.H.clone()
    from gemq.quantizers.hessian import inverse_hessian_factor
    factor = inverse_hessian_factor(hessian, quantizer.percdamp)
    output = torch.zeros_like(weight)
    scales_list, zeros_list = [], []
    for start in range(0, weight.shape[1], quantizer.blocksize):
        end = min(start + quantizer.blocksize, weight.shape[1])
        block = weight[:, start:end].clone()
        errors = torch.zeros_like(block)
        for offset in range(end - start):
            column = start + offset
            if column % quantizer.groupsize == 0:
                scales, zeros, maximum = quantizer.find_params(
                    weight[:, column:column + quantizer.groupsize]
                )
                scales_list.append(scales)
                zeros_list.append(zeros)
            values = block[:, offset]
            levels = quantizer.quantize_vector(
                values.unsqueeze(1), scales, zeros, maximum
            )
            restored = ((levels - zeros) * scales).flatten()
            error = (values - restored) / factor[column, column]
            block[:, offset:] -= error.unsqueeze(1) @ factor[
                column, column:end
            ].unsqueeze(0)
            output[:, column] = levels.flatten()
            errors[:, offset] = error
        weight[:, end:] -= errors @ factor[start:end, end:]
    return (
        output.to(quantizer.W.dtype).reshape(-1, quantizer.groupsize),
        torch.cat(scales_list, dim=1).reshape(-1, 1),
        torch.cat(zeros_list, dim=1).reshape(-1, 1),
    )


@pytest.fixture(params=["cpu", "cuda"])
def quantization_device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    return request.param


def _check(device, bits, dtype, mse, group_size, block_size):
    generator = torch.Generator().manual_seed(7)
    weight = torch.randn(5, 384, generator=generator) * 0.1
    weight[0, :128] = 0
    weight[:, 1] = 0  # Exercise the 1-bit zero-sign rule.
    inputs = torch.randn(512, 384, generator=generator)
    inputs[:, 0] = 0  # Exercise dead-column handling.
    quantizer = MCMoeGPTQWeightQuantizer(
        weight.to(device=device, dtype=dtype), nbits=bits,
        groupsize=group_size, blocksize=block_size, mse=mse,
    )
    quantizer.H = (inputs.T @ inputs / inputs.shape[0]).to(device)
    expected = _columnwise(quantizer)
    actual = quantizer.quantize()
    for candidate, reference in zip(actual, expected):
        torch.testing.assert_close(candidate, reference, rtol=0, atol=0)
    torch.testing.assert_close(
        quantizer.dequantize(*actual), quantizer.dequantize(*expected),
        rtol=0, atol=0,
    )


@pytest.mark.parametrize("bits", [1, 2, 3])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("mse", [False, True])
def test_low_bit_blocks_match_columnwise(quantization_device, bits, dtype, mse):
    _check(quantization_device, bits, dtype, mse, 128, 128)


@pytest.mark.parametrize("bits", [1, 3, 4])
@pytest.mark.parametrize("group,block", [(64, 128), (128, 64), (128, 128)])
def test_other_layouts_and_four_bits(quantization_device, bits, group, block):
    _check(quantization_device, bits, torch.float32, False, group, block)
