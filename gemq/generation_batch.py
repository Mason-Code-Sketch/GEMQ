"""Bound generation batch probes by requests without changing model context limits."""
from functools import wraps


def install_generation_batch_probe(wrapped, *, max_new_tokens=256):
    """Keep HFLM's forward/OOM probe; replace only its synthetic input length.

    Installed on this evaluator instance only. The formal generate_until path,
    context limit, prompts, generation settings and output handling stay intact.
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
        for request in requests:
            context, generation = request.args
            budget = int(generation.get("max_gen_toks", max_new_tokens))
            if budget <= 0 or budget >= wrapped.max_length:
                raise ValueError("generation budget must be positive and below model context length")
            input_tokens = len(wrapped.tok_encode(context))
            effective_tokens = min(input_tokens, wrapped.max_length - budget)
            lengths.append((effective_tokens + budget, input_tokens, budget))
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
            # HFLM removes the last continuation token; this yields exactly
            # probe_tokens input tokens, including room for the generation budget.
            synthetic = [(None, [1] * probe_tokens, [1])]
            selected = original_detect(synthetic, 0)
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
