from contextlib import contextmanager
from functools import partial

from torch.utils.checkpoint import checkpoint


@contextmanager
def checkpoint_attention(layers, num_layers):
    """Recompute the last attention blocks and restore their original forwards."""
    if num_layers < 0:
        raise ValueError("Attention checkpoint layer count must be nonnegative.")
    originals = []
    try:
        for layer in list(layers)[-num_layers:] if num_layers else []:
            attention = layer.self_attn
            originals.append((attention, attention.__dict__.get("forward")))
            attention.forward = partial(checkpoint, attention.forward, use_reentrant=False)
        yield len(originals)
    finally:
        for attention, forward in originals:
            if forward is None:
                del attention.forward
            else:
                attention.forward = forward
