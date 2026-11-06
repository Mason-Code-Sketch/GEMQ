"""Compact single-token MMLU scoring for supported native causal models."""
import inspect
from contextlib import contextmanager
from types import MethodType


@contextmanager
def compact_mmlu_scoring(wrapped):
    """Preserve native tokenization, context grouping, full-vocabulary scores and order."""
    model = getattr(wrapped, "model", None)
    stats = {"enabled": False, "optimized_requests": 0, "fallback_calls": 0}
    supported = (getattr(wrapped, "backend", None) == "causal"
                 and getattr(wrapped, "world_size", None) == 1
                 and getattr(getattr(model, "config", None), "model_type", None)
                 in {"qwen2_moe", "qwen3_moe", "mixtral"}
                 and type(model).__module__.startswith("transformers.models."))
    if not supported or "logits_to_keep" not in inspect.signature(model.forward).parameters:
        yield stats
        return
    native = wrapped._loglikelihood_tokens
    had_override = "_loglikelihood_tokens" in wrapped.__dict__
    old_override = wrapped.__dict__.get("_loglikelihood_tokens")

    def score(self, requests, disable_tqdm=False, override_bs=None):
        # Native fallback also covers unusual tokenizers and multi-token answer labels.
        if not requests or any(len(r[2]) != 1 or not r[1] for r in requests):
            stats["fallback_calls"] += 1
            return native(requests, disable_tqdm=disable_tqdm, override_bs=override_bs)
        import torch
        from torch.nn import functional as F
        from lm_eval.models.utils import Collator
        from lm_eval.models.utils_hf import pad_and_concat
        from tqdm import tqdm

        ordered = Collator(requests,
            sort_fn=lambda r: (-len(r[1] + r[2]), tuple(r[1] + r[2])),
            group_by="contexts" if self.logits_cache else None,
            group_fn=lambda r: r[1] + r[2][:-1])
        batch = self.batch_size if self.batch_size != "auto" else (override_bs or 0)
        batch_fn = self._batch_scheduler if self.batch_size == "auto" and not override_bs else None
        if batch_fn is not None:
            self.batch_sizes = {}
        results = []
        # Auto probes retain the native full-output path, providing a conservative cap.
        with tqdm(total=len(requests), disable=disable_tqdm or self.rank != 0,
                  desc="Running MMLU scored-position requests") as progress:
            for chunk in ordered.get_batched(n=batch, batch_fn=batch_fn):
                inputs = [torch.tensor((ctx + cont)[-(self.max_length + 1):][:-1],
                                       dtype=torch.long, device=self.device)
                          for _, ctx, cont in chunk]
                lengths = [len(t) for t in inputs]
                padded = pad_and_concat(max(lengths), inputs, padding_side="right")
                # Select the union of valid final positions before lm_head. This
                # handles unequal lengths without changing native right padding.
                positions, inverse = torch.tensor([n - 1 for n in lengths]).unique(
                    sorted=True, return_inverse=True)
                # CPU indices also work when the output head resides on a
                # different GPU from the input embedding.
                with torch.no_grad(), torch.autocast(device_type=self.device.type,
                        dtype=self.mixed_precision_dtype,
                        enabled=self.mixed_precision_dtype is not None):
                    logits = self.model(padded, logits_to_keep=positions).logits
                    rows = torch.arange(len(chunk), device=logits.device)
                    selected = logits[rows, inverse.to(logits.device), :]
                    logprobs = F.log_softmax(selected, dim=-1, dtype=self.softmax_dtype)
                del logits, selected
                for (key, context, continuation), probs in zip(chunk, logprobs, strict=True):
                    probs = probs.view(1, 1, -1)
                    greedy = probs.argmax(dim=-1)
                    for key, continuation, cached in ordered.get_cache(
                            req_str=key, cxt_toks=context, cont_toks=continuation, logits=probs):
                        token = continuation[0]
                        answer = (float(cached[0, 0, token]), bool(greedy[0, 0] == token))
                        results.append(answer)
                        if key is not None:
                            self.cache_hook.add_partial("loglikelihood", key, answer)
                        stats["optimized_requests"] += 1
                        progress.update(1)
                del logprobs
        return ordered.get_original(results)

    wrapped._loglikelihood_tokens = MethodType(score, wrapped)
    stats["enabled"] = True
    try:
        yield stats
    finally:
        if had_override:
            wrapped._loglikelihood_tokens = old_override
        else:
            del wrapped._loglikelihood_tokens
