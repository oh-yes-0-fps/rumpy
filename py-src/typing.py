"""Bare-minimum ``numpy.typing`` for rumpy.

Exports the three names a typical ``from numpy.typing import ...`` line
expects. These are runtime placeholders aimed at making static-typing
imports succeed — rumpy itself doesn't enforce the hints.

The ``_ndarray`` name is injected into the module globals by rumpy's
Rust side before this source is executed, so ``NDArray[T]`` resolves to
the real ndarray class.

Exports:
    NDArray[T]   — generic alias resolving to ``numpy.ndarray``
    ArrayLike    — anything coercible to an array
    DTypeLike    — anything that can specify a dtype

The implementation deliberately avoids ``import typing`` so it works in
rustpython builds that don't ship the Python stdlib.
"""


class _Union:
    """Stand-in for ``typing.Union`` — ``_Union[A, B, …]`` returns a sentinel
    object carrying the member types. Pure-runtime placeholder; nothing
    inspects the contents."""

    def __class_getitem__(cls, items):
        if not isinstance(items, tuple):
            items = (items,)
        # Return a fresh instance so each alias compares as a distinct object.
        inst = object.__new__(cls)
        inst.__args__ = items
        return inst

    def __repr__(self):
        names = ", ".join(getattr(t, "__name__", repr(t)) for t in self.__args__)
        return f"Union[{names}]"


# DTypeLike — anything numpy accepts as a `dtype=` argument:
#   * one of the scalar types (``np.float64``, ``np.int32``, …),
#   * a dtype string ("float64", "int32", "f8", "<i4", …).
DTypeLike = _Union[type, str]


# ArrayLike — anything coercible to an ndarray by ``np.array(...)``:
#   * an existing ndarray (``_ndarray``),
#   * a (nested) sequence of numbers / bools,
#   * a Python scalar (int / float / bool / complex).
ArrayLike = _Union[_ndarray, list, tuple, int, float, bool, complex]


# NDArray[T] — generic alias for "ndarray of element type T". rumpy doesn't
# carry element-type info on the static type, so ``NDArray[float]`` is just
# ``numpy.ndarray`` at runtime — the parameter is purely documentary.


class NDArray:
    """``NDArray[T]`` ≡ ``numpy.ndarray`` (T unenforced)."""

    def __class_getitem__(cls, _item):
        return _ndarray


class NBitBase:
    """Stand-in for numpy's static-typing generic ``NBitBase``.

    Real numpy uses this as a base for parametrized integer/float bit-width
    aliases (``np.intp`` etc.). rumpy doesn't enforce those constraints at
    runtime; the class exists so ``from numpy.typing import NBitBase`` works.
    """

    def __init_subclass__(cls, **kwargs):
        pass


__all__ = ["NDArray", "ArrayLike", "DTypeLike", "NBitBase"]
