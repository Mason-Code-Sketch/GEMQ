"""Project only the final hidden position during native autoregressive generation."""
from contextlib import contextmanager


@contextmanager
def compact_generation_logits(wrapped):
    """Keep full-vocabulary next-token logits and release the temporary hook on exit."""
    model = wrapped.model
    head = getattr(model, "lm_head", None)
    kind = getattr(getattr(model, "config", None), "model_type", None)
    supported = (kind in {"deepseek_v2", "qwen2_moe", "qwen3_moe", "mixtral"}
                 and callable(getattr(head, "register_forward_pre_hook", None)))
    stats = {"enabled": supported, "trimmed_calls": 0, "max_input_positions": 0}
    if not supported:
        yield stats
        return

    def last_position(_head, args):
        hidden = args[0]
        stats["max_input_positions"] = max(stats["max_input_positions"], hidden.shape[1])
        if hidden.shape[1] == 1:
            return None
        stats["trimmed_calls"] += 1
        return (hidden[:, -1:, :], *args[1:])

    hook = head.register_forward_pre_hook(last_position)
    try:
        yield stats
    finally:
        hook.remove()
