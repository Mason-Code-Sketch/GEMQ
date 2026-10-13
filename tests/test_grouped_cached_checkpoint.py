"""Same-checkpoint, terminal-only grouped evaluation correctness and speed gate."""

import hashlib
import statistics
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

import pytest
import torch


@pytest.mark.cuda
@pytest.mark.checkpoint
def test_grouped_cached_checkpoint(model_path):
    from transformers import AutoTokenizer
    from lm_eval.models.huggingface import HFLM
    from gemq.utils.hf_loading import load_quantized_model
    from gemq.utils.eval_utils import run_lm_eval
    from gemq.inference.cached_linear import cache_quantized_linears
    from gemq.inference.grouped_cached import install_grouped_cached

    path = Path(model_path)
    def source_hash():
        with (path / "qmodel.pt").open("rb") as stream:
            return hashlib.file_digest(stream, "sha256").hexdigest()
    before = source_hash()
    torch.manual_seed(0)
    torch.cuda.set_per_process_memory_fraction(min(1.0, 116 * 2**30 / torch.cuda.get_device_properties(0).total_memory))
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = load_quantized_model(str(path), compute_dtype=torch.float16, device="cuda:0").eval()
    cache_quantized_linears(model)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    tokens = tokenizer("A person is choosing the best way to complete a practical task. ", return_tensors="pt").input_ids
    inputs = [tokens.repeat(batch, (length + tokens.shape[1] - 1) // tokens.shape[1])[:, :length].cuda()
              for batch, length in ((1, 128), (2, 512), (8, 512))]

    @torch.inference_mode()
    def measure():
        outputs, times = [], []
        for ids in inputs:
            for _ in range(2):
                model(ids, use_cache=False)
            seconds = []
            for _ in range(5):
                torch.cuda.synchronize()
                start = perf_counter()
                logits = model(ids, use_cache=False).logits
                torch.cuda.synchronize()
                seconds.append(perf_counter() - start)
            assert torch.isfinite(logits).all()
            outputs.append(logits.cpu())
            times.append(statistics.median(seconds))
        return outputs, times

    def options():
        scores = []
        original = HFLM.loglikelihood
        def record(self, *args, **kwargs):
            result = original(self, *args, **kwargs)
            scores.extend(result)
            return result
        with patch.object(HFLM, "loglikelihood", record):
            result = run_lm_eval(model, tokenizer, tasks=["piqa"], batch_size="auto",
                                 max_batch_size=8, limit=4, seed=0)
        assert scores
        return scores, result["summary"]

    reference, before_seconds = measure()
    before_options, before_summary = options()
    torch.cuda.reset_peak_memory_stats()
    install_grouped_cached(model)
    print(f"GROUPED_CONVERSION allocated_gib={torch.cuda.memory_allocated()/2**30:.3f} "
          f"reserved_gib={torch.cuda.memory_reserved()/2**30:.3f} "
          f"peak_allocated_gib={torch.cuda.max_memory_allocated()/2**30:.3f} "
          f"peak_reserved_gib={torch.cuda.max_memory_reserved()/2**30:.3f}", flush=True)
    model.eval()
    actual, after_seconds = measure()
    exact = True
    for ids, old, new, old_seconds, new_seconds in zip(inputs, reference, actual, before_seconds, after_seconds):
        equal = torch.equal(old, new)
        targets = ids[:, 1:].cpu().unsqueeze(-1)
        old_ll = old[:, :-1].float().log_softmax(-1).gather(-1, targets).sum(1)
        new_ll = new[:, :-1].float().log_softmax(-1).gather(-1, targets).sum(1)
        print(f"GROUPED shape={tuple(ids.shape)} exact_logits={equal} "
              f"max_logit_error={(old.float()-new.float()).abs().max().item()} "
              f"max_sequence_ll_error={(old_ll-new_ll).abs().max().item()} "
              f"cached_seconds={old_seconds:.6f} grouped_seconds={new_seconds:.6f} "
              f"speedup={old_seconds/new_seconds:.3f}", flush=True)
        exact &= equal and torch.equal(old_ll, new_ll)
    after_options, after_summary = options()
    option_error = max(abs(a[0]-b[0]) for a, b in zip(before_options, after_options))
    options_equal = before_options == after_options
    print(f"GROUPED options_equal={options_equal} max_option_ll_error={option_error} "
          f"piqa_equal={before_summary==after_summary}", flush=True)
    assert source_hash() == before
    assert exact and options_equal and before_summary == after_summary, "Exact correctness gate failed"
    assert sum(after_seconds) < .95 * sum(before_seconds), "Speed improvement must exceed 5%"
    print("GROUPED_SMOKE_PASSED exact_logits=true exact_options=true", flush=True)
