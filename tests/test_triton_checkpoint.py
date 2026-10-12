"""Serial full-checkpoint comparison against the current four-device FP16 cache."""

import gc
import hashlib
import statistics
from pathlib import Path
from time import perf_counter

import pytest
import torch


@pytest.mark.cuda
@pytest.mark.checkpoint
def test_triton_checkpoint(model_path):
    from transformers import AutoTokenizer
    from gemq.inference.cached_linear import cache_quantized_linears
    from gemq.inference.layerwise import distribute_packed_model
    from gemq.inference.triton_linear import use_triton_linears
    from gemq.utils.eval_utils import run_lm_eval
    from gemq.utils.hf_loading import ensure_deepseek_v2_remote_code_compat, load_quantized_model

    if torch.cuda.device_count() < 4:
        pytest.skip("requires four CUDA devices")
    ensure_deepseek_v2_remote_code_compat()
    path = Path(model_path)
    def source_hash():
        with (path / "qmodel.pt").open("rb") as stream:
            return hashlib.file_digest(stream, "sha256").hexdigest()
    before_hash = source_hash()
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    tokens = tokenizer("A person is choosing the best way to complete a practical task. ",
                       return_tensors="pt").input_ids
    inputs = [tokens.repeat(batch, (length + tokens.shape[1] - 1) // tokens.shape[1])[:, :length].to("cuda:0")
              for batch, length in ((1, 128), (2, 512), (8, 512))]
    def synchronize():
        for device in range(4):
            torch.cuda.synchronize(device)
    def measure(convert):
        for device in range(4):
            torch.cuda.reset_peak_memory_stats(device)
        model = load_quantized_model(str(path), compute_dtype=torch.float16, device="cpu")
        model = distribute_packed_model(model, [f"cuda:{i}" for i in range(4)])
        model.eval()
        convert(model)
        outputs, timings = [], []
        with torch.inference_mode():
            for ids in inputs:
                start = perf_counter()
                warmup = model(ids, use_cache=False)
                del warmup
                synchronize()
                print(f"WARMUP shape={tuple(ids.shape)} seconds={perf_counter()-start:.3f}", flush=True)
                seconds = []
                for _ in range(3):
                    synchronize()
                    start = perf_counter()
                    logits = model(ids, use_cache=False).logits
                    synchronize()
                    seconds.append(perf_counter() - start)
                outputs.append(logits.detach().cpu())
                timings.append(statistics.median(seconds))
                del logits
        piqa = run_lm_eval(model, tokenizer, tasks=["piqa"], batch_size="auto",
                           max_batch_size=8, limit=4, seed=0)
        allocated = [torch.cuda.max_memory_allocated(i) / 2**30 for i in range(4)]
        reserved = [torch.cuda.max_memory_reserved(i) / 2**30 for i in range(4)]
        print(f"MEMORY allocated_GiB={allocated} reserved_GiB={reserved}", flush=True)
        del model
        gc.collect()
        for device in range(4):
            with torch.cuda.device(device):
                torch.cuda.empty_cache()
        return outputs, timings, piqa["summary"], allocated
    reference, cache_times, cache_piqa, cache_memory = measure(cache_quantized_linears)
    actual, triton_times, triton_piqa, triton_memory = measure(use_triton_linears)
    passed = True
    for ids, output, expected, cached, lowbit in zip(inputs, actual, reference, cache_times, triton_times):
        output, expected = output.float(), expected.float()
        relative = ((output - expected).norm() / expected.norm().clamp_min(1e-12)).item()
        targets = ids[:, 1:].cpu().unsqueeze(-1)
        ll = [x[:, :-1].log_softmax(-1).gather(-1, targets).mean().item() for x in (output, expected)]
        error = abs(ll[0] - ll[1])
        print(f"TRITON_CASE shape={tuple(ids.shape)} exact={torch.equal(output, expected)} "
              f"logits_relative_error={relative:.8g} mean_logprob_error={error:.8g} "
              f"cached_seconds={cached:.6f} triton_seconds={lowbit:.6f} speedup={cached/lowbit:.4f}", flush=True)
        passed &= bool(torch.isfinite(output).all()) and relative <= 0.005 and error <= 0.005
    print(f"TRITON_SUMMARY piqa_equal={cache_piqa == triton_piqa} "
          f"speedup={sum(cache_times)/sum(triton_times):.4f} "
          f"cached_peak_GiB={sum(cache_memory):.4f} triton_peak_GiB={sum(triton_memory):.4f}", flush=True)
    assert source_hash() == before_hash
    assert passed and cache_piqa == triton_piqa, "Low-bit numerical/PIQA gate failed"
    assert sum(triton_times) < 0.95 * sum(cache_times), "Low-bit prefill did not improve by 5%"
    assert sum(triton_memory) < sum(cache_memory), "Low-bit peak memory did not decrease"
    print("TRITON_SMOKE_PASSED", flush=True)
