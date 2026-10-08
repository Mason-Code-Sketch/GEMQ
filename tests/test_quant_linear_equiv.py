"""
Level 1: does a single real-quant linear compute the same thing as an fp16 matmul
against the dequantized weight?

This pins down the two conventions the rest of the stack silently depends on:
  * the scales/zeros meaning. GPTQ/RTN and the Triton helper both dequantize as
    (Q - zeros) * scales, retaining fractional zero-points and FP16 rounding.
  * the 3-bit pack/unpack round trip, including the padding-row truncation in
    patch.create_gemlite_from_hqq.

If this file fails, nothing further up the stack is trustworthy.
"""

import pytest
import torch
import torch.nn.functional as F

from conftest import make_gemlite_linear, relative_error, fp16_matmul_noise


# in_features must be divisible by the group size; out_features kept GEMM-friendly
IN_FEATURES = 512
OUT_FEATURES = 256
NUM_TOKENS = 16

# how many times the plain-fp16-matmul error the kernel is allowed to be
KERNEL_ERROR_BUDGET = 5.0


@pytest.mark.cuda
@pytest.mark.parametrize("nbits", [1, 2, 3, 4])
def test_gemlite_keeps_fractional_zero_points(device, nbits):
    from gemq.inference.patch import create_gemlite_from_hqq
    from gemq.utils.quant_utils import create_hqq_linear_from_quantized_weights

    torch.manual_seed(0)
    shape = (OUT_FEATURES, IN_FEATURES)
    groups = OUT_FEATURES * IN_FEATURES // 128
    q = torch.randint(0, 2 ** nbits, (groups, 128), device=device, dtype=torch.uint8)
    scales = (torch.rand(groups, 1, device=device) * 0.1 + 0.01).half()
    zeros = (torch.rand(groups, 1, device=device) * (2 ** nbits - 1)).half()
    hqq = create_hqq_linear_from_quantized_weights(
        q, scales, zeros, shape, nbits, 128, device=device,
    )
    gem = create_gemlite_from_hqq(hqq)
    assert gem.W_group_mode == 3
    assert torch.equal(gem.zeros.t().reshape_as(zeros), zeros)
    x = torch.randn(NUM_TOKENS, IN_FEATURES, device=device).half()
    from gemq.triton_kernels.dequant_gemm import dequant_gemm_triton
    actual = dequant_gemm_triton(x, gem.W_q, gem.scales, gem.zeros, nbits, 128)
    expected = F.linear(x, hqq.dequantize())
    assert relative_error(actual, expected) < 1e-3


@pytest.mark.cuda
@pytest.mark.parametrize("nbits", [1, 2, 3, 4])
@pytest.mark.parametrize("group_size", [64, 128])
def test_hqq_dequant_round_trip(device, nbits, group_size):
    """
    The packed HQQ weight must dequantize back to exactly the fp16 weight that the
    fake-quant model carries. Any drift here is a silent weight difference between
    the two checkpoints, independent of any kernel.
    """
    torch.manual_seed(0)
    W = (torch.randn(OUT_FEATURES, IN_FEATURES, device=device) * 0.02).half()

    _, hqq_linear, W_deq = make_gemlite_linear(W, nbits, group_size, device=device)

    W_round_trip = hqq_linear.dequantize().reshape(W_deq.shape)
    max_abs = (W_round_trip.float() - W_deq.float()).abs().max().item()
    assert max_abs == 0.0, (
        f"packed weight does not round-trip (nbits={nbits}, group_size={group_size}): "
        f"max |dequant(pack(Q)) - W_fake| = {max_abs}"
    )


@pytest.mark.cuda
@pytest.mark.parametrize("nbits", [1, 2, 3, 4])
@pytest.mark.parametrize("group_size", [64, 128])
def test_repo_gemm_kernel_matches_fp16_matmul(device, nbits, group_size):
    """
    The path that actually matters for low-bit weights: GEMQ's own dequant GEMM
    reading the GemLite-packed buffer, exactly as QuantFusedDeepseekV2MoEBlock does
    for shared experts. Expert weights never reach GemLite's own forward -- the fused
    blocks pull .W_q/.scales/.zeros off the GemLite object and run these kernels.
    """
    from gemq.triton_kernels.dequant_gemm import dequant_gemm_triton

    torch.manual_seed(0)
    W = (torch.randn(OUT_FEATURES, IN_FEATURES, device=device) * 0.02).half()
    x = (torch.randn(NUM_TOKENS, IN_FEATURES, device=device) * 0.5).half()

    gemlite_linear, _, W_deq = make_gemlite_linear(W, nbits, group_size, device=device)

    y_kernel = dequant_gemm_triton(
        x, gemlite_linear.W_q, gemlite_linear.scales, gemlite_linear.zeros,
        gemlite_linear.W_nbits, gemlite_linear.group_size,
    )
    torch_noise, ref32 = fp16_matmul_noise(x, W_deq)
    kernel_error = relative_error(y_kernel, ref32)

    budget = max(KERNEL_ERROR_BUDGET * torch_noise, 1e-3)
    assert kernel_error <= budget, (
        f"GEMQ dequant_gemm_triton diverges from fp16 matmul on the dequantized weight "
        f"(nbits={nbits}, group_size={group_size}): kernel rel err {kernel_error:.3e} "
        f"vs torch fp16 rel err {torch_noise:.3e} (budget {budget:.3e})"
    )


@pytest.mark.cuda
@pytest.mark.parametrize("nbits", [1, 2, 3, 4])
def test_repo_gemv_kernel_matches_fp16_matmul(device, nbits):
    """Same, at the decode shape, through the split-k GEMV the shared experts use."""
    from gemq.triton_kernels.dequant_gemv import dequant_splitk_gemv_triton

    torch.manual_seed(0)
    W = (torch.randn(OUT_FEATURES, IN_FEATURES, device=device) * 0.02).half()
    x = (torch.randn(1, IN_FEATURES, device=device) * 0.5).half()

    gemlite_linear, _, W_deq = make_gemlite_linear(W, nbits, 128, device=device)

    y_kernel = dequant_splitk_gemv_triton(
        x, gemlite_linear.W_q, gemlite_linear.scales, gemlite_linear.zeros,
        gemlite_linear.W_nbits, gemlite_linear.group_size,
    )
    torch_noise, ref32 = fp16_matmul_noise(x, W_deq)
    kernel_error = relative_error(y_kernel, ref32)

    budget = max(KERNEL_ERROR_BUDGET * torch_noise, 1e-3)
    assert kernel_error <= budget, (
        f"GEMQ dequant_splitk_gemv_triton diverges at batch size 1 (nbits={nbits}): "
        f"{kernel_error:.3e} vs {torch_noise:.3e} (budget {budget:.3e})"
    )


@pytest.mark.cuda
@pytest.mark.parametrize("nbits", [1, 2, 3, 4])
@pytest.mark.parametrize("group_size", [64, 128])
def test_gemlite_linear_matches_fp16_matmul(device, nbits, group_size):
    """
    GemLite's *own* forward on the same packed weight. Only 4-bit is load-bearing
    today (attention and dense layers keep their GemLiteLinearTriton), but the low
    bit-widths are covered to document where the packing is and is not portable.

    3-bit is expected to fail: GemLite does not support it natively, patch.py appends
    it to SUPPORTED_BITS_TRITON and swaps in GEMQ's own packer so that *GEMQ's*
    kernels can read the buffer. GemLite's kernels never learned that layout.
    """
    if nbits == 3:
        pytest.xfail(
            "3-bit is not natively supported by GemLite; patch.py enables it only for "
            "GEMQ's own kernels. A 3-bit weight left as a plain GemLiteLinearTriton "
            "would silently return garbage -- see test_repo_gemm_kernel_* for the path "
            "that is actually used."
        )

    torch.manual_seed(0)
    W = (torch.randn(OUT_FEATURES, IN_FEATURES, device=device) * 0.02).half()
    x = (torch.randn(NUM_TOKENS, IN_FEATURES, device=device) * 0.5).half()

    gemlite_linear, _, W_deq = make_gemlite_linear(W, nbits, group_size, device=device)

    y_kernel = gemlite_linear(x)
    torch_noise, ref32 = fp16_matmul_noise(x, W_deq)
    kernel_error = relative_error(y_kernel, ref32)

    budget = max(KERNEL_ERROR_BUDGET * torch_noise, 1e-3)
    assert kernel_error <= budget, (
        f"GemLite output diverges from fp16 matmul on the dequantized weight "
        f"(nbits={nbits}, group_size={group_size}): kernel rel err {kernel_error:.3e} "
        f"vs torch fp16 rel err {torch_noise:.3e} (budget {budget:.3e})"
    )


@pytest.mark.cuda
@pytest.mark.parametrize("nbits", [2, 3])
def test_gemlite_linear_matches_at_batch_one(device, nbits):
    """
    Same check with a single token. Decode-time shapes hit different kernel
    configurations than the batched case, so cover both.
    """
    if nbits == 3:
        pytest.xfail("see test_gemlite_linear_matches_fp16_matmul: GemLite has no 3-bit support")

    torch.manual_seed(0)
    W = (torch.randn(OUT_FEATURES, IN_FEATURES, device=device) * 0.02).half()
    x = (torch.randn(1, IN_FEATURES, device=device) * 0.5).half()

    gemlite_linear, _, W_deq = make_gemlite_linear(W, nbits, 128, device=device)

    y_kernel = gemlite_linear(x)
    torch_noise, ref32 = fp16_matmul_noise(x, W_deq)
    kernel_error = relative_error(y_kernel, ref32)

    budget = max(KERNEL_ERROR_BUDGET * torch_noise, 1e-3)
    assert kernel_error <= budget, (
        f"GemLite output diverges at batch size 1 (nbits={nbits}): "
        f"{kernel_error:.3e} vs {torch_noise:.3e} (budget {budget:.3e})"
    )
