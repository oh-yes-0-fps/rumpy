"""``numpy.exceptions`` — exception and warning classes used by numpy.

This is the user-facing slice of numpy 2.x's ``numpy.exceptions`` module:
the exception/warning *types* that downstream code catches or raises. The
runtime numpy operations in rumpy raise the standard Python exceptions
(``ValueError``, ``TypeError`` …) for the same error conditions, but
these names exist so that ``isinstance(...)`` checks and ``except``
clauses written against numpy's public types still compile and run.
"""


class AxisError(ValueError, IndexError):
    """An axis is out of bounds or is duplicated."""

    def __init__(self, axis, ndim=None, msg_prefix=None):
        self.axis = axis
        self.ndim = ndim
        if ndim is None:
            msg = str(axis)
        else:
            msg = f"axis {axis} is out of bounds for array of dimension {ndim}"
        if msg_prefix is not None:
            msg = f"{msg_prefix}: {msg}"
        super().__init__(msg)


class ComplexWarning(RuntimeWarning):
    """Casting a complex value to a real type discards the imaginary part."""


class RankWarning(UserWarning):
    """Issued by ``polyfit`` when a polynomial fit is poorly conditioned."""


class TooHardError(RuntimeError):
    """An operation exceeded an internal complexity budget."""


class VisibleDeprecationWarning(UserWarning):
    """A deprecation that users (not just downstream libraries) should see."""


class DTypePromotionError(TypeError):
    """No common dtype exists for the requested promotion."""


__all__ = [
    "AxisError",
    "ComplexWarning",
    "RankWarning",
    "TooHardError",
    "VisibleDeprecationWarning",
    "DTypePromotionError",
]
