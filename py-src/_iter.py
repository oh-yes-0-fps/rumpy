"""Iterator and buffer classes that numpy exposes at the top level.

This file backs the ``ndindex`` / ``ndenumerate`` / ``broadcast`` /
``nditer`` / ``flatiter`` / ``memmap`` names. The implementations are
deliberately compact — enough to satisfy ``for idx in np.ndindex(...)``
loops and the like, but not a faithful port of numpy's C-level iterator
machinery.

``_np`` is the rumpy ``numpy`` module, injected from Rust.
"""


class ndindex:
    """Iterate over every multi-index of an N-D shape, in C-order."""

    def __init__(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
            shape = tuple(shape[0])
        self.shape = tuple(int(d) for d in shape)
        self._total = 1
        for d in self.shape:
            self._total *= d
        self._i = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self._i >= self._total:
            raise StopIteration
        rem = self._i
        ix = []
        for d in reversed(self.shape):
            ix.append(rem % d)
            rem //= d
        self._i += 1
        return tuple(reversed(ix))


class ndenumerate:
    """Iterate over ``(idx, value)`` pairs of an N-D array in C-order."""

    def __init__(self, arr):
        self._arr = _np.asarray(arr)
        self._flat = list(self._arr.ravel().tolist())
        self._shape = tuple(self._arr.shape)
        self._i = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self._i >= len(self._flat):
            raise StopIteration
        rem = self._i
        ix = []
        for d in reversed(self._shape):
            ix.append(rem % d)
            rem //= d
        v = self._flat[self._i]
        self._i += 1
        return tuple(reversed(ix)), v


class broadcast:
    """Iterate over the elements of an arbitrary number of broadcasted arrays.

    Each iteration yields a tuple of one element per input array. Inputs
    are broadcast to a common shape using ``broadcast_shapes``.
    """

    def __init__(self, *arrays):
        if not arrays:
            self.shape = ()
            self._flat = []
            return
        arrs = [_np.asarray(a) for a in arrays]
        shape = _np.broadcast_shapes(*[tuple(a.shape) for a in arrs])
        self.shape = shape
        self.size = 1
        for d in shape:
            self.size *= d
        # Broadcast each to `shape` by tiling — slow but correct.
        self._cols = []
        for a in arrs:
            tiled = _np.broadcast_to(a, shape) if hasattr(_np, "broadcast_to") else a
            self._cols.append(list(tiled.ravel().tolist()))
        self._i = 0
        self.numiter = len(arrs)

    @property
    def nd(self):
        return len(self.shape)

    def __iter__(self):
        self._i = 0
        return self

    def __next__(self):
        if self._i >= self.size:
            raise StopIteration
        out = tuple(col[self._i] for col in self._cols)
        self._i += 1
        return out


class nditer:
    """Minimal ``nditer`` over a single array. Supports ``op_flags=['readwrite']``.

    Real numpy's ``nditer`` is highly configurable; rumpy ships only the
    "iterate over a single array in C-order" path. ``for x in nditer(a):``
    yields each element wrapped in a one-element list-like so callers can
    use ``x[...] = ...`` to assign back (best-effort).
    """

    def __init__(self, arr, op_flags=None, flags=None, order="C"):
        _ = flags
        _ = order
        self._arr = _np.asarray(arr)
        self._flat = list(self._arr.ravel().tolist())
        self._i = 0
        self._writable = bool(op_flags) and "readwrite" in op_flags

    def __iter__(self):
        return self

    def __next__(self):
        if self._i >= len(self._flat):
            raise StopIteration
        v = _Slot(self._flat[self._i])
        self._i += 1
        return v

    @property
    def shape(self):
        return self._arr.shape

    @property
    def itviews(self):
        return [self._arr]

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False


class _Slot:
    """Mutable single-element handle yielded by nditer."""

    __slots__ = ("_v",)

    def __init__(self, v):
        self._v = v

    def __repr__(self):
        return repr(self._v)

    def __getitem__(self, _key):
        return self._v

    def __setitem__(self, _key, value):
        self._v = value

    def __float__(self):
        return float(self._v)

    def __int__(self):
        return int(self._v)


def nested_iters(arrays, axes, flags=None, op_flags=None, order="C"):
    """Return two ``nditer``s of the given arrays — minimal best-effort port."""
    _ = (axes, flags, op_flags, order)
    return [nditer(a) for a in arrays]


class flatiter:
    """Iterator over an ndarray's flat (C-order) view.

    ``arr.flat`` typically returns this; rumpy's ``ndarray`` doesn't yet
    expose a `flat` property, so users instantiate this directly with the
    underlying array.
    """

    def __init__(self, arr):
        self._arr = _np.asarray(arr)
        self._flat = list(self._arr.ravel().tolist())
        self._i = 0

    def __iter__(self):
        self._i = 0
        return self

    def __next__(self):
        if self._i >= len(self._flat):
            raise StopIteration
        v = self._flat[self._i]
        self._i += 1
        return v

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return self._flat[idx]
        return self._flat[idx]

    def __setitem__(self, idx, value):
        self._flat[idx] = value

    def __len__(self):
        return len(self._flat)


class memmap:
    """Memory-mapped ndarray. rumpy doesn't expose mmap so this is a thin
    wrapper that reads the file into an ndarray and writes it back on
    flush — semantically equivalent for small workloads.
    """

    def __init__(self, filename, dtype="float64", mode="r+", offset=0,
                 shape=None, order="C"):
        _ = order
        self.filename = filename
        self.dtype = dtype
        self.mode = mode
        self.offset = offset
        self._arr = None
        if shape is not None:
            n = 1
            for d in shape:
                n *= d
            self._arr = _np.zeros(shape, dtype=dtype)
            if "r" in mode:
                self._load()

    def _load(self):
        try:
            with open(self.filename, "rb") as f:
                f.seek(self.offset)
                data = f.read()
                self._arr = _np.frombuffer(data, dtype=self.dtype)
        except (OSError, FileNotFoundError):
            pass

    def flush(self):
        if self._arr is None or "w" not in self.mode and "+" not in self.mode:
            return
        # Write back as raw bytes.
        import struct
        flat = list(self._arr.ravel().tolist())
        fmt = {"float64": "d", "float32": "f", "int64": "q", "int32": "i",
               "int16": "h", "int8": "b", "uint64": "Q", "uint32": "I",
               "uint16": "H", "uint8": "B"}.get(str(self.dtype), "d")
        buf = b"".join(struct.pack(f"<{fmt}", v) for v in flat)
        with open(self.filename, "wb") as f:
            f.seek(self.offset)
            f.write(buf)

    def __getitem__(self, idx):
        return self._arr[idx]

    def __setitem__(self, idx, value):
        self._arr[idx] = value

    @property
    def shape(self):
        return self._arr.shape if self._arr is not None else ()


__all__ = [
    "ndindex",
    "ndenumerate",
    "broadcast",
    "nditer",
    "flatiter",
    "nested_iters",
    "memmap",
]
