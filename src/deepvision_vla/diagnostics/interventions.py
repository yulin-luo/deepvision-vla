"""Single-layer token intervention, with cached decoding left untouched."""

from contextlib import contextmanager


@contextmanager
def zero_visual_tokens(layer, positions):
    positions = tuple(positions)
    if len(set(positions)) != len(positions) or any(type(x) is not int or x < 0 for x in positions):
        raise ValueError("Dropped token positions must be unique nonnegative integers")
    observed = {"prefill_calls": 0, "cached_decode_calls": 0, "dropped_positions": positions}

    def hook(module, args, kwargs):
        positional = bool(args)
        hidden = args[0] if positional else kwargs.get("hidden_states")
        if hidden is None or hidden.ndim != 3 or hidden.shape[0] != 1:
            raise ValueError("Diagnosis currently requires one [1, tokens, width] sequence")
        if hidden.shape[1] == 1:
            observed["cached_decode_calls"] += 1
            return None
        if positions and max(positions) >= hidden.shape[1]:
            raise ValueError("Visual positions do not fit the observed prefill sequence")
        observed["prefill_calls"] += 1
        if not positions:
            return None
        modified = hidden.clone()
        modified[:, list(positions), :] = 0
        if positional:
            return (modified,) + args[1:], kwargs
        return args, {**kwargs, "hidden_states": modified}

    handle = layer.register_forward_pre_hook(hook, with_kwargs=True)
    try:
        yield observed
    finally:
        handle.remove()
