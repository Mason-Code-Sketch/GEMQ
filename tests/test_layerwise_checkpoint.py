"""Opt-in real-checkpoint correctness and throughput gate; no result files written."""

import gc
import hashlib
import statistics
from pathlib import Path
from time import perf_counter

import pytest
import torch


@pytest.mark.cuda
@pytest.mark.checkpoint
def test_layerwise_cached_checkpoint(model_path):
    from transformers import AutoTokenizer
    from hqq.core.quantize import HQQLinear
    from gemq.inference.cached_linear import cache_quantized_linears
    from gemq.inference.layerwise import distribute_packed_model
    from gemq.utils.eval_utils import run_lm_eval
    from gemq.utils.hf_loading import ensure_deepseek_v2_remote_code_compat, load_quantized_model

    if torch.cuda.device_count() < 4:
        pytest.skip("needs four visible CUDA devices")
    ensure_deepseek_v2_remote_code_compat()
    path = Path(model_path)
    with (path / "qmodel.pt").open("rb") as stream:
        source_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    tokens = tokenizer("A person is choosing the best way to complete a practical task. ", return_tensors="pt").input_ids
    inputs = [tokens.repeat(batch, (length + tokens.shape[1] - 1) // tokens.shape[1])[:, :length].to("cuda:0")
              for batch, length in [(1, 128), (2, 512), (8, 512)]]

    def measure(model):
        outputs, timings = [], []
        with torch.no_grad():
            for ids in inputs:
                model(ids, use_cache=False)
                seconds = []
                for _ in range(3):
                    for device in range(4):
                        torch.cuda.synchronize(device)
                    start = perf_counter()
                    logits = model(ids, use_cache=False).logits
                    for device in range(4):
                        torch.cuda.synchronize(device)
                    seconds.append(perf_counter() - start)
                assert torch.isfinite(logits).all()
                outputs.append(logits.detach().cpu())
                timings.append(statistics.median(seconds))
        return outputs, timings

    torch.manual_seed(0)
    for device in range(4):
        torch.cuda.reset_peak_memory_stats(device)
    baseline = load_quantized_model(str(path), compute_dtype=torch.float16, device="cuda:0")
    assert baseline.config.model_type == "deepseek_v2"
    baseline.eval()
    references, baseline_seconds = measure(baseline)
    baseline_piqa = run_lm_eval(baseline, tokenizer, tasks=["piqa"], batch_size="auto", max_batch_size=8, limit=4, seed=0)
    print("Baseline allocated peaks GiB:", [torch.cuda.max_memory_allocated(i) / 2**30 for i in range(4)], flush=True)
    del baseline
    gc.collect()
    for device in range(4):
        with torch.cuda.device(device):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)

    start = perf_counter()
    optimized = load_quantized_model(str(path), compute_dtype=torch.float16, device="cpu")
    optimized = distribute_packed_model(optimized, [f"cuda:{i}" for i in range(4)])
    optimized.eval()
    cache_quantized_linears(optimized)
    assert not any(isinstance(module, HQQLinear) for module in optimized.modules())
    for device in range(4):
        torch.cuda.synchronize(device)
    print(f"Layerwise load/cache seconds={perf_counter() - start:.3f}", flush=True)
    outputs, cached_seconds = measure(optimized)
    for ids, actual, reference in zip(inputs, outputs, references):
        error = (actual.float() - reference.float()).abs().max().item()
        print(f"shape={tuple(ids.shape)} max_logit_error={error}", flush=True)
        assert torch.equal(actual, reference), "Single-card and four-card logits differ"
        targets = ids[:, 1:].cpu().unsqueeze(-1)
        scores = [output[:, :-1].float().log_softmax(-1).gather(-1, targets).sum(1) for output in (actual, reference)]
        assert torch.equal(*scores)
        assert torch.equal(actual.argmax(-1), reference.argmax(-1))
    actual_piqa = run_lm_eval(optimized, tokenizer, tasks=["piqa"], batch_size="auto", max_batch_size=8, limit=4, seed=0)
    print("Cached allocated peaks GiB:", [torch.cuda.max_memory_allocated(i) / 2**30 for i in range(4)], flush=True)
    assert actual_piqa["summary"] == baseline_piqa["summary"]
    for index, (before, after) in enumerate(zip(baseline_seconds, cached_seconds)):
        print(f"case={index} pytorch_seconds={before:.6f} cached_seconds={after:.6f} speedup={before/after:.3f}", flush=True)
    assert sum(cached_seconds) < 0.95 * sum(baseline_seconds), "Speed improvement must exceed 5%"
    with (path / "qmodel.pt").open("rb") as stream:
        assert hashlib.file_digest(stream, "sha256").hexdigest() == source_hash
    print("LAYERWISE_SMOKE_PASSED exact_logits=true exact_scores=true piqa_equal=true", flush=True)
