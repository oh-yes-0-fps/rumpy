"""
numpy.lib — top-level shim that re-exports the most-commonly-used helpers.
"""

# These will be populated by Rust-side injection of the submodules.
# We expose stride_tricks at the same level numpy.lib does.

__all__ = ["stride_tricks"]
