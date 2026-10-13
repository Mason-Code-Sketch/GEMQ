"""Full WT2/C4 same-checkpoint PPL comparison, printed without result files."""

import hashlib
import os
from pathlib import Path
from time import perf_counter

import pytest
import torch


@pytest.mark.cuda
@pytest.mark.checkpoint
def test_grouped_cached_ppl(model_path):
    from transformers import AutoTokenizer
    from gemq.utils.hf_loading import load_quantized_model
    from gemq.utils.eval_utils import get_testenc
    from gemq.inference.cached_linear import cache_quantized_linears
    from gemq.inference.grouped_cached import install_grouped_cached

    path = Path(model_path)
    def source_hash():
        with (path / "qmodel.pt").open("rb") as stream:
            return hashlib.file_digest(stream, "sha256").hexdigest()
    before = source_hash()
    torch.cuda.set_per_process_memory_fraction(min(1.0, 116 * 2**30 / torch.cuda.get_device_properties(0).total_memory))
    tokenizer = AutoTokenizer.from_pretrained(path)
    datasets = {name: get_testenc(tokenizer, name, 2048,
                dataset_root=os.environ["GEMQ_GROUPED_DATASET_ROOT"]).input_ids
                for name in ("wikitext2", "c4")}
    model = load_quantized_model(str(path), compute_dtype=torch.float16, device="cuda:0").eval()
    cache_quantized_linears(model)
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    @torch.inference_mode()
    def evaluate(backend):
        results = {}
        for name, ids in datasets.items():
            count = ids.numel() // 2048
            first = ids[:, :2048].cuda()
            for _ in range(2):
                model(first, use_cache=False)
            torch.cuda.synchronize()
            start, losses = perf_counter(), []
            for index in range(count):
                block = ids[:, index*2048:(index+1)*2048].cuda()
                logits = model(block, use_cache=False).logits
                loss = torch.nn.functional.cross_entropy(
                    logits[:, :-1].contiguous().float().view(-1, logits.shape[-1]),
                    block[:, 1:].reshape(-1))
                assert torch.isfinite(loss)
                losses.append(loss.cpu())
                del logits, loss
                if (index + 1) % 32 == 0:
                    print(f"PPL {backend} {name} blocks={index+1}/{count}", flush=True)
            torch.cuda.synchronize()
            seconds = perf_counter() - start
            losses = torch.stack(losses)
            ppl = torch.exp(losses.mean()).item()
            results[name] = losses, ppl, seconds
            print(f"PPL {backend} {name} blocks={count} ppl={ppl:.9f} seconds={seconds:.3f}", flush=True)
        return results

    reference = evaluate("cached")
    torch.cuda.reset_peak_memory_stats()
    install_grouped_cached(model)
    print(f"GROUPED_CONVERSION allocated_gib={torch.cuda.memory_allocated()/2**30:.3f} "
          f"reserved_gib={torch.cuda.memory_reserved()/2**30:.3f} "
          f"peak_allocated_gib={torch.cuda.max_memory_allocated()/2**30:.3f} "
          f"peak_reserved_gib={torch.cuda.max_memory_reserved()/2**30:.3f}", flush=True)
    actual = evaluate("grouped_cached")
    for name in datasets:
        old, old_ppl, old_seconds = reference[name]
        new, new_ppl, new_seconds = actual[name]
        print(f"PPL_COMPARE {name} exact_blocks={torch.equal(old,new)} "
              f"max_block_nll_error={(old-new).abs().max().item()} "
              f"speedup={old_seconds/new_seconds:.3f}", flush=True)
        assert torch.equal(old, new) and old_ppl == new_ppl
    assert source_hash() == before
    print("GROUPED_PPL_PASSED", flush=True)
