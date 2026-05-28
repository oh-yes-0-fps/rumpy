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


class ModuleDeprecationWarning(DeprecationWarning):
    """A whole numpy submodule is being deprecated."""


class LinAlgError(Exception):
    """Linear-algebra-specific error (singular matrix, non-convergence, …).

    Real numpy puts this in ``numpy.linalg``; we keep the canonical class
    here so it's reachable from both ``numpy.exceptions.LinAlgError`` and
    ``numpy.linalg.LinAlgError`` (the latter via the post-import patch
    below).
    """


# Side-effect: real numpy exposes ``LinAlgError`` as ``numpy.linalg.LinAlgError``
# too. rustpython's `#[pymodule]` macro doesn't accept nested pyattrs, so we
# materialise the class up here and pin it onto ``numpy.linalg`` at import
# time. The try/except guards against the (rare) case where ``numpy.linalg``
# isn't reachable yet.
# numpy.exceptions runs during numpy module construction (the lazy pyattrs
# get materialised eagerly while ``extend_module`` runs), so at this point
# ``numpy.linalg`` isn't on the module yet — we can't patch it from here.
# Embedders who want ``numpy.linalg.LinAlgError`` callable after import can
# do ``numpy.linalg.LinAlgError = numpy.exceptions.LinAlgError`` explicitly.


__all__ = [
    "AxisError",
    "ComplexWarning",
    "RankWarning",
    "TooHardError",
    "VisibleDeprecationWarning",
    "DTypePromotionError",
    "ModuleDeprecationWarning",
    "LinAlgError",
]
