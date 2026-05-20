"""
numpy.lib.stride_tricks — windowing helpers.

This is a minimal compatibility shim that materializes a fresh array rather
than returning a strided view (rumpy doesn't yet have true views over the
same backing storage). The shapes and values match numpy's results.
"""


def sliding_window_view(x, window_shape, axis=None):
    """``np.lib.stride_tricks.sliding_window_view``.

    Build all overlapping windows of ``window_shape`` over ``x``. With
    ``axis=None``, ``window_shape`` may be an int (applied to the last axis)
    or a tuple. The output has ``window_shape`` appended to the input's
    leading axes.
    """
    if isinstance(window_shape, int):
        window_shape = (window_shape,)
    else:
        window_shape = tuple(window_shape)

    if axis is None:
        axes = tuple(range(x.ndim - len(window_shape), x.ndim))
    elif isinstance(axis, int):
        axes = (axis,)
    else:
        axes = tuple(axis)

    if len(window_shape) != len(axes):
        raise ValueError("window_shape and axis must match in length")

    # Build output shape: leading dims = input dims minus (window - 1) on each
    # windowed axis; trailing dims = window_shape.
    out_shape = list(x.shape)
    for win, ax in zip(window_shape, axes):
        if win > x.shape[ax]:
            raise ValueError("window larger than axis length")
        out_shape[ax] = x.shape[ax] - win + 1
    out_shape = out_shape + list(window_shape)

    # Allocate output array (zeros) and fill it by iterating over the
    # window positions.
    import numpy as np
    out = np.zeros(tuple(out_shape), dtype=x.dtype)

    # Helper: enumerate all valid window-position tuples along the windowed
    # axes (one tuple per leading-output cell).
    leading_iters = [range(out_shape[ax]) for ax in axes]

    def _walk(prefix, depth):
        if depth == len(leading_iters):
            # `prefix` is the leading index tuple for the windowed axes.
            window_idx = tuple(prefix)
            # Compute the slice of x for this window.
            slc = [slice(None)] * x.ndim
            for k, ax in enumerate(axes):
                start = prefix[k]
                slc[ax] = slice(start, start + window_shape[k])
            block = x[tuple(slc)]
            # Place block into out at out[<full leading index>, ...] position
            # — i.e. for non-axes leading dims, we keep them as-is; the
            # windowed leading dims are `prefix`.
            full_lead = list(x.shape)
            # Build the LHS index: leading dims (non-axes preserved, axes
            # replaced by prefix), then the trailing window dims = ":" .
            out_lead = list(slc)
            for k, ax in enumerate(axes):
                out_lead[ax] = prefix[k]
            # Combine: out_lead is integer/slice mix giving leading dims;
            # then ":" for trailing.
            full_idx = tuple(out_lead) + (slice(None),) * len(window_shape)
            out[full_idx] = block
            return
        for v in leading_iters[depth]:
            _walk(prefix + (v,), depth + 1)

    _walk((), 0)
    return out


def as_strided(x, shape=None, strides=None, subok=False, writeable=True):
    """``np.lib.stride_tricks.as_strided``.

    Without true views, we can only honor ``shape`` (broadcast / reshape into
    that shape if compatible). ``strides`` is ignored — passing a non-default
    value raises a clear error so callers don't get silently wrong results.
    """
    import numpy as np
    if strides is not None:
        raise NotImplementedError(
            "as_strided with explicit strides is not supported (no true views)"
        )
    if shape is None:
        return x.copy()
    target = tuple(shape)
    if int(np.prod(target)) == int(np.prod(x.shape)):
        return x.reshape(target)
    return np.broadcast_to(x, target).copy()


def broadcast_shapes(*shapes):
    """``np.broadcast_shapes(*shapes)``."""
    import numpy as np
    if not shapes:
        return ()
    result = []
    nd = max(len(s) for s in shapes)
    padded = [(1,) * (nd - len(s)) + tuple(s) for s in shapes]
    for axis_dims in zip(*padded):
        m = 1
        for d in axis_dims:
            if d == 1:
                continue
            if m == 1:
                m = d
            elif m != d:
                raise ValueError(f"shapes are not broadcastable: {shapes}")
        result.append(m)
    return tuple(result)
