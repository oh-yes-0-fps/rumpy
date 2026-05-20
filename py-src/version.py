"""``numpy.version`` — version metadata.

Mirrors the public surface of ``numpy.version`` so code that reads
``numpy.version.version`` / ``numpy.version.full_version`` keeps working.
The values are injected from the host crate's version at module-init
time (see rumpy's ``build_py_submodule``); the assignments below are
fallbacks for static analysis only.
"""


# These names are overwritten by the host before the module is exposed.
version = "0.0.0"
full_version = version
git_revision = "unknown"
release = True

short_version = version

__all__ = ["version", "full_version", "git_revision", "release", "short_version"]
