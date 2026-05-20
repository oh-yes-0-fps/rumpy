"""
numpy index helpers: mgrid, ogrid, r_, c_, s_, ix_

These are typically used via subscripting (e.g. `np.mgrid[0:5, 0:3]`) rather
than function calls.
"""

# `_np` is the rumpy numpy module, injected from Rust.


class _MGridClass:
    """``np.mgrid[s1, s2, ...]`` — dense meshgrid."""

    def __getitem__(self, key):
        if not isinstance(key, tuple):
            key = (key,)
        ranges = []
        for s in key:
            if isinstance(s, slice):
                start = s.start if s.start is not None else 0
                stop = s.stop
                step = s.step if s.step is not None else 1
                if isinstance(step, complex):
                    # mgrid supports complex steps meaning "number of points"
                    n = int(step.imag)
                    ranges.append(_np.linspace(start, stop, n))
                else:
                    ranges.append(_np.arange(start, stop, step))
            else:
                ranges.append(_np.array([s]))
        if len(ranges) == 1:
            return ranges[0]
        # Build dense N-D grids — for each axis, broadcast its 1-D range across
        # the other axes' dimensions.
        out = []
        shape = tuple(r.shape[0] for r in ranges)
        for i, r in enumerate(ranges):
            new_shape = [1] * len(ranges)
            new_shape[i] = r.shape[0]
            broadcast = _np.broadcast_to(r.reshape(tuple(new_shape)), shape)
            out.append(broadcast)
        return _np.array(out)


class _OGridClass:
    """``np.ogrid[s1, s2, ...]`` — sparse meshgrid (1-D arrays per axis)."""

    def __getitem__(self, key):
        if not isinstance(key, tuple):
            key = (key,)
        out = []
        for i, s in enumerate(key):
            if isinstance(s, slice):
                start = s.start if s.start is not None else 0
                stop = s.stop
                step = s.step if s.step is not None else 1
                if isinstance(step, complex):
                    n = int(step.imag)
                    arr = _np.linspace(start, stop, n)
                else:
                    arr = _np.arange(start, stop, step)
            else:
                arr = _np.array([s])
            # Reshape so axis i is the length-of-axis and others are 1.
            shape = [1] * len(key)
            shape[i] = arr.shape[0]
            out.append(arr.reshape(tuple(shape)))
        if len(out) == 1:
            return out[0]
        return tuple(out)


class _RClass:
    """``np.r_[a, b, c, ...]`` — concatenate along axis 0."""

    def __getitem__(self, key):
        if not isinstance(key, tuple):
            key = (key,)
        parts = []
        for k in key:
            if isinstance(k, slice):
                start = k.start if k.start is not None else 0
                stop = k.stop
                step = k.step if k.step is not None else 1
                if isinstance(step, complex):
                    parts.append(_np.linspace(start, stop, int(step.imag)))
                else:
                    parts.append(_np.arange(start, stop, step))
            else:
                parts.append(_np.asarray(k))
        # numpy.r_ flattens scalar/0-D inputs.
        normalized = []
        for p in parts:
            if p.ndim == 0:
                normalized.append(p.reshape((1,)))
            else:
                normalized.append(p)
        return _np.concatenate(normalized)


class _CClass:
    """``np.c_[a, b]`` — stack columns (concatenate along axis 1 after 2D-ifying)."""

    def __getitem__(self, key):
        if not isinstance(key, tuple):
            key = (key,)
        parts = []
        for k in key:
            arr = _np.asarray(k)
            if arr.ndim == 1:
                arr = arr.reshape((arr.shape[0], 1))
            parts.append(arr)
        return _np.concatenate(parts, axis=1)


class _SClass:
    """``np.s_[a:b, c:d]`` — pass-through for slice objects."""

    def __getitem__(self, key):
        return key


def ix_(*args):
    """``np.ix_`` — build an N-D open mesh from 1-D index arrays."""
    out = []
    n = len(args)
    for i, arg in enumerate(args):
        a = _np.asarray(arg)
        shape = [1] * n
        shape[i] = a.shape[0] if a.ndim > 0 else 1
        out.append(a.reshape(tuple(shape)))
    return tuple(out)


# Singletons — match numpy's pattern of exposing instances.
mgrid = _MGridClass()
ogrid = _OGridClass()
r_ = _RClass()
c_ = _CClass()
s_ = _SClass()
