"""Probe native generation at bounded request lengths without full-sequence scores."""
from functools import wraps


def probe_generation_batch(wrapped, probe_tokens, context_tokens=None):
    """Cover end-of-generation cache length with a longer prefill and two decode steps."""
    import torch
    from accelerate.utils.memory import find_executable_batch_size
    from lm_eval.models.utils_hf import clear_torch_cache

    steps = min(2, probe_tokens - 1)

    @find_executable_batch_size(starting_batch_size=wrapped.max_batch_size)
    def generate_batch(batch_size):
        length = probe_tokens - steps
        tokens = context_tokens or [1]
        tokens = (tokens * ((length + len(tokens) - 1) // len(tokens)))[:length]
        context = torch.tensor(tokens, dtype=torch.long, device=wrapped.device).repeat(batch_size, 1)
        print(f"[generation-probe] batch={batch_size} tokens={probe_tokens} "
              f"decode_steps={steps}", flush=True)
        with torch.autocast(device_type=wrapped.device.type,
                dtype=wrapped.mixed_precision_dtype,
                enabled=wrapped.mixed_precision_dtype is not None):
            for _ in range(2):
                generated = wrapped.model.generate(input_ids=context,
                    max_length=probe_tokens, min_new_tokens=steps,
                    do_sample=False, use_cache=True,
                    pad_token_id=wrapped.tokenizer.pad_token_id)
                del generated
        return batch_size

    try:
        return generate_batch()
    finally:
        clear_torch_cache()


def install_generation_batch_probe(wrapped, *, max_new_tokens=256):
    """Use native cached generation for auto probing on a single causal evaluator.

    The formal generate_until requests, context limit and generation settings stay intact.
    """
    original_generate = wrapped.generate_until
    records = []

    @wraps(original_generate)
    def generate_until(requests, *args, **kwargs):
        if wrapped.batch_size != "auto":
            return original_generate(requests, *args, **kwargs)
        if not requests:
            return []
        lengths = []
        probe_context = None
        longest = -1
        for request in requests:
            context, generation = request.args
            budget = int(generation.get("max_gen_toks", max_new_tokens))
            if budget <= 0 or budget >= wrapped.max_length:
                raise ValueError("generation budget must be positive and below model context length")
            tokens = wrapped.tok_encode(context)
            input_tokens = len(tokens)
            effective_tokens = min(input_tokens, wrapped.max_length - budget)
            lengths.append((effective_tokens + budget, input_tokens, budget))
            if effective_tokens + budget > longest:
                longest = effective_tokens + budget
                probe_context = tokens[-effective_tokens:] if effective_tokens else []
        probe_tokens, _, _ = max(lengths)
        record = {"max_input_tokens": max(row[1] for row in lengths),
                  "max_new_tokens": max(row[2] for row in lengths),
                  "probe_tokens": probe_tokens, "model_max_length": wrapped.max_length,
                  "max_batch_size": wrapped.max_batch_size, "selected_batch_size": None}
        original_detect = wrapped._detect_batch_size
        missing = object()
        previous_override = wrapped.__dict__.get("_detect_batch_size", missing)

        def detect_batch_size(requests=None, pos=0):
            if requests is not None:
                return original_detect(requests, pos)
            if (probe_tokens < 2 or getattr(wrapped, "backend", None) != "causal"
                    or getattr(wrapped, "world_size", 1) != 1):
                synthetic = [(None, [1] * probe_tokens, [1])]
                selected = original_detect(synthetic, 0)
            else:
                selected = probe_generation_batch(wrapped, probe_tokens, probe_context)
            record["selected_batch_size"] = selected
            print("GSM8K_BATCH_PROBE " + " ".join(f"{k}={v}" for k, v in record.items()), flush=True)
            return selected

        wrapped._detect_batch_size = detect_batch_size
        try:
            return original_generate(requests, *args, **kwargs)
        finally:
            if previous_override is missing:
                del wrapped._detect_batch_size
            else:
                wrapped._detect_batch_size = previous_override
            records.append(record)

    wrapped.generate_until = generate_until
    return records
