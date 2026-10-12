"""Exact-zero regression coverage for MC-MoE's degenerate weight groups."""

import pytest

torch = pytest.importorskip("torch")

from gemq.quantizers.gptq import MCMoeGPTQWeightQuantizer


def quantize(weight, nbits, mse, hessian=None):
    quantizer = MCMoeGPTQWeightQuantizer(
        weight, nbits=nbits, groupsize=128, blocksize=128, mse=mse,
    )
    if hessian is not None:
        quantizer.H.copy_(hessian)
        quantizer.nsamples = 1
    codes, scales, zeros = quantizer.quantize()
    decoded = quantizer.dequantize(codes, scales, zeros).reshape_as(weight)
    return codes, scales, zeros, decoded


@pytest.mark.parametrize("nbits", [1, 2, 3])
@pytest.mark.parametrize("mse", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_unobserved_expert_decodes_to_exact_zero(nbits, mse, dtype):
    weight = torch.full((4, 128), 0.02, dtype=dtype)
    codes, scales, zeros, decoded = quantize(weight, nbits, mse)
    assert torch.isfinite(codes).all()
    assert torch.isfinite(scales).all()
    assert torch.isfinite(zeros).all()
    assert torch.count_nonzero(decoded) == 0


@pytest.mark.parametrize("nbits", [2, 3])
@pytest.mark.parametrize("mse", [False, True])
def test_zero_groups_preserve_normal_rows(nbits, mse):
    torch.manual_seed(731)
    normal = (torch.randn(2, 256) * 0.02).half()
    weight = torch.cat([torch.zeros_like(normal), normal])
    # A live expert may have zero groups as well as nonzero groups.
    weight[1, 128:] = normal[0, 128:]
    inputs = torch.randn(320, 256)
    hessian = inputs.T @ inputs / 320 + torch.eye(256) * 0.05
    _, _, _, decoded = quantize(weight, nbits, mse, hessian)
    normal_result = quantize(normal, nbits, mse, hessian)
    assert torch.count_nonzero(decoded[0]) == 0
    assert torch.count_nonzero(decoded[1, :128]) == 0
    assert torch.equal(decoded[2:], normal_result[3])


@pytest.mark.cuda
@pytest.mark.parametrize("nbits", [1, 2, 3])
@pytest.mark.parametrize("mse", [False, True])
@pytest.mark.parametrize("unobserved", [False, True])
def test_gpu_zero_groups_survive_pipeline_packing(device, nbits, mse, unobserved):
    from gemq.utils.quant_utils import create_hqq_linear_from_quantized_weights

    torch.manual_seed(743)
    weight = (torch.randn(8, 256, device=device) * 0.02).half()
    weight[0] = 0
    weight[1, :128] = 0
    hessian = None
    if not unobserved:
        inputs = torch.randn(320, 256, device=device)
        hessian = inputs.T @ inputs / 320 + torch.eye(256, device=device) * 0.05
    codes, scales, zeros, decoded = quantize(weight, nbits, mse, hessian)
    # The actual pipeline casts weights and buffers through BF16 during FT.
    decoded = decoded.bfloat16().half()
    scales = scales.bfloat16().half()
    zeros = zeros.bfloat16().half()
    groups = scales.shape[0]
    repacked_codes = torch.round(decoded.reshape(groups, -1) / scales + zeros)
    repacked_codes = repacked_codes.clamp(0, 2**nbits - 1)
    packed = create_hqq_linear_from_quantized_weights(
        repacked_codes, scales, zeros, tuple(weight.shape), nbits, 128,
        device=device,
    )
    unpacked = packed.dequantize()
    assert torch.isfinite(unpacked).all()
    assert torch.count_nonzero(unpacked[0]) == 0
    assert torch.count_nonzero(unpacked[1, :128]) == 0
    reference = torch.where(
        scales == 0, torch.zeros_like(repacked_codes),
        (repacked_codes - zeros) * scales,
    ).reshape_as(weight)
    assert torch.equal(unpacked, reference)
    if unobserved:
        assert torch.count_nonzero(unpacked) == 0
