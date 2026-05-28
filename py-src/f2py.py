"""``numpy.f2py`` — Fortran-to-Python bridge (placeholder).

rumpy doesn't ship a Fortran toolchain. The names below resolve so that
``from numpy.f2py import ...`` and ``np.f2py.<name>(...)`` don't crash;
each callable raises ``NotImplementedError`` to signal the absence.
"""


def _unavailable(name):
    def _f(*args, **kwargs):
        raise NotImplementedError(
            f"numpy.f2py.{name} is unavailable in rumpy (no Fortran toolchain)"
        )
    _f.__name__ = name
    return _f


compile = _unavailable("compile")
run_main = _unavailable("run_main")
get_include = _unavailable("get_include")


def test(*args, **kwargs):
    """No-op test runner."""
    _ = (args, kwargs)
    return True


# `numpy.f2py` re-exports a handful of internal helper *modules* in real
# numpy. We expose each as a tiny namespace whose attribute lookups raise
# NotImplementedError. This satisfies ``from numpy.f2py import auxfuncs``
# without pretending to have a Fortran toolchain underneath.

class _F2PySubmoduleStub:
    """Lazy stand-in for an internal numpy.f2py submodule."""

    def __init__(self, name):
        self._name = name

    def __repr__(self):
        return f"<numpy.f2py.{self._name} (stub)>"

    def __getattr__(self, item):
        raise NotImplementedError(
            f"numpy.f2py.{self._name}.{item} is unavailable in rumpy "
            "(no Fortran toolchain)"
        )


auxfuncs = _F2PySubmoduleStub("auxfuncs")
capi_maps = _F2PySubmoduleStub("capi_maps")
cb_rules = _F2PySubmoduleStub("cb_rules")
cfuncs = _F2PySubmoduleStub("cfuncs")
common_rules = _F2PySubmoduleStub("common_rules")
crackfortran = _F2PySubmoduleStub("crackfortran")
diagnose = _F2PySubmoduleStub("diagnose")
f2py2e = _F2PySubmoduleStub("f2py2e")
f90mod_rules = _F2PySubmoduleStub("f90mod_rules")
func2subr = _F2PySubmoduleStub("func2subr")
rules = _F2PySubmoduleStub("rules")
symbolic = _F2PySubmoduleStub("symbolic")
use_rules = _F2PySubmoduleStub("use_rules")


def main():
    """f2py's CLI entry point — unavailable in rumpy."""
    raise NotImplementedError("numpy.f2py.main is unavailable in rumpy")


# Real numpy's f2py imports a few stdlib modules at the top level — expose
# them so attribute lookups don't fail. We import lazily and fall back to
# stubs if the host interpreter doesn't ship the stdlib.
try:
    import os
except ImportError:
    os = _F2PySubmoduleStub("os")
try:
    import sys
except ImportError:
    sys = _F2PySubmoduleStub("sys")
try:
    import subprocess
except ImportError:
    subprocess = _F2PySubmoduleStub("subprocess")
try:
    import warnings
except ImportError:
    warnings = _F2PySubmoduleStub("warnings")


# `numpy.f2py` re-exports the deprecation warning class.
class VisibleDeprecationWarning(UserWarning):
    """A deprecation that users (not just downstream libraries) should see."""


__all__ = [
    "compile", "run_main", "get_include", "test", "main",
    "auxfuncs", "capi_maps", "cb_rules", "cfuncs", "common_rules",
    "crackfortran", "diagnose", "f2py2e", "f90mod_rules", "func2subr",
    "rules", "symbolic", "use_rules",
    "os", "sys", "subprocess", "warnings",
    "VisibleDeprecationWarning",
]
