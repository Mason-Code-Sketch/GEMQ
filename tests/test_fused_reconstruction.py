"""Compare streamed scoring against the unchanged native grouped-MM scorer."""
from unittest.mock import patch

import pytest
import torch
from transformers import Qwen3MoeConfig
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeSparseMoeBlock

from gemq.compute_model_stats import compute_fused_moe_layer_reconstruction_errors
from gemq.quantizers.rtn import MCMoeRTNWeightQuantizer
from gemq.utils.fused_reconstruction import (
    GroupedBatchContributions, streamed_fused_reconstruction_scores,
    cross_expert_reconstruction_scores, DiskContributions,
)


def fixture(dtype=torch.float32, normalize=True):
    torch.manual_seed(47)
    config = Qwen3MoeConfig(hidden_size=128, moe_intermediate_size=128,
        num_experts=4, num_experts_per_tok=2, norm_topk_prob=normalize)
    config._experts_implementation = "grouped_mm"
    block = Qwen3MoeSparseMoeBlock(config).to(dtype=dtype).eval()
    with torch.no_grad():
        for parameter in block.parameters():
            parameter.normal_(0, .02)
        inputs = torch.randn(3, 16, 128, dtype=dtype)
        reference = torch.cat([block(inputs[i:i + 1]) for i in range(3)])
    gradients = torch.randn_like(inputs)
    return block, inputs, reference, gradients


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("normalize", [False, True])
@torch.inference_mode()
def test_scores_and_weights_match_native(dtype, normalize):
    block, x, y, gradients = fixture(dtype, normalize)
    before = {k: t.clone() for k, t in block.state_dict().items()}
    cache = GroupedBatchContributions(block, x[:1], y[:1])
    assert cache.exact
    calls = []
    def quantizer(weight, nbits):
        calls.append(nbits)
        return MCMoeRTNWeightQuantizer(weight, nbits=nbits)
    expected = compute_fused_moe_layer_reconstruction_errors(
        block, x, y, gradients.double().pow(2), [1, 2, 3], 1)
    actual = streamed_fused_reconstruction_scores(block, x, y, gradients,
                                                [1, 2, 3], 1, quantizer)
    assert actual == expected
    # Two projections for each expert and bit, independent of sample count.
    assert len(calls) == 2 * 4 * 3
    assert all(torch.equal(t, block.state_dict()[k]) for k, t in before.items())


@torch.inference_mode()
def test_mismatched_reference_falls_back_to_native():
    block, x, y, gradients = fixture()
    y = y + .01
    expected = compute_fused_moe_layer_reconstruction_errors(
        block, x, y, gradients.double().pow(2), [2], 1)
    assert streamed_fused_reconstruction_scores(block, x, y, gradients,
        [2], 1, MCMoeRTNWeightQuantizer) == expected


@torch.inference_mode()
def test_error_restores_quantized_expert():
    block, x, y, gradients = fixture()
    before = {k: t.clone() for k, t in block.state_dict().items()}
    with patch.object(GroupedBatchContributions, "candidate_output",
                      side_effect=RuntimeError("probe error")):
        with pytest.raises(RuntimeError, match="probe error"):
            streamed_fused_reconstruction_scores(block, x, y, gradients,
                [2], 1, MCMoeRTNWeightQuantizer)
    assert all(torch.equal(t, block.state_dict()[k]) for k, t in before.items())


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("normalize", [False, True])
@torch.inference_mode()
def test_cross_expert_exact_and_cleanup(tmp_path, dtype, normalize):
    block, x, y, gradients = fixture(dtype, normalize)
    before = {k: t.clone() for k, t in block.state_dict().items()}
    expected = streamed_fused_reconstruction_scores(block, x, y, gradients,
        [1, 2, 3], 1, MCMoeRTNWeightQuantizer)
    calls = []
    def quantizer(weight, nbits):
        calls.append(nbits)
        return MCMoeRTNWeightQuantizer(weight, nbits=nbits)
    actual = cross_expert_reconstruction_scores(block, x, y, gradients,
        [1, 2, 3], 1, quantizer, scratch_dir=tmp_path)
    assert actual == expected
    assert len(calls) == 24
    assert list(tmp_path.iterdir()) == []
    assert all(torch.equal(t, block.state_dict()[k]) for k, t in before.items())


@torch.inference_mode()
def test_cross_expert_error_cleanup(tmp_path):
    block, x, y, gradients = fixture()
    before = {k: t.clone() for k, t in block.state_dict().items()}
    with patch.object(DiskContributions, "read_rows", side_effect=RuntimeError("read error")):
        with pytest.raises(RuntimeError, match="read error"):
            cross_expert_reconstruction_scores(block, x, y, gradients,
                [2], 1, MCMoeRTNWeightQuantizer, scratch_dir=tmp_path)
    assert list(tmp_path.iterdir()) == []
    assert all(torch.equal(t, block.state_dict()[k]) for k, t in before.items())


@torch.inference_mode()
def test_cross_expert_mismatch_fallback(tmp_path):
    block, x, y, gradients = fixture()
    y = y + .01
    expected = streamed_fused_reconstruction_scores(block, x, y, gradients,
        [2], 1, MCMoeRTNWeightQuantizer)
    assert cross_expert_reconstruction_scores(block, x, y, gradients,
        [2], 1, MCMoeRTNWeightQuantizer, scratch_dir=tmp_path) == expected
    assert list(tmp_path.iterdir()) == []


def test_anonymous_cache_released_after_sigkill(tmp_path):
    import multiprocessing
    import os
    import signal
    context = multiprocessing.get_context("fork")
    parent, child = context.Pipe()
    def worker():
        storage = DiskContributions(tmp_path, (1, 16, 2, 128), torch.float32)
        storage.tensor.fill_(1)
        storage.mapping.flush()
        child.send((os.fstat(storage.file.fileno()).st_nlink, str(storage.file.name)))
        child.recv()
    process = context.Process(target=worker)
    process.start()
    try:
        assert parent.poll(10)
        links, _ = parent.recv()
        assert links == 0
        assert list(tmp_path.iterdir()) == []
        os.kill(process.pid, signal.SIGKILL)
        process.join(10)
        assert process.exitcode == -signal.SIGKILL
        assert not os.path.exists(f"/proc/{process.pid}/fd")
        assert list(tmp_path.iterdir()) == []
    finally:
        if process.is_alive():
            process.kill()
            process.join()


@torch.inference_mode()
def test_disk_pages_do_not_accumulate(tmp_path):
    import ctypes
    import types
    storage = DiskContributions(tmp_path, (8, 2048, 8, 2048), torch.float16)
    try:
        cache = types.SimpleNamespace(
            contributions=torch.zeros(2048 * 8, 2048, dtype=torch.float16),
            inverse=torch.arange(2048 * 8))
        libc = ctypes.CDLL(None, use_errno=True)
        page = storage.mmap_module.PAGESIZE
        resident = (ctypes.c_ubyte * ((storage.size + page - 1) // page))()
        def resident_pages():
            assert libc.mincore(ctypes.c_void_p(storage.tensor.data_ptr()),
                ctypes.c_size_t(storage.size), resident) == 0
            return sum(v & 1 for v in resident)
        for sample in range(8):
            storage.write_batch(sample, cache)
            assert resident_pages() == 0
        rows = storage.read_rows(0, torch.tensor([0, 10, 100]), "cpu")
        assert rows.shape == (3, 8, 2048)
        assert resident_pages() == 0
    finally:
        storage.close()
