"""``numpy.compat`` — deprecated Python 2/3 compatibility shim.

In numpy 2.x this module is deprecated and slated for removal. rumpy
ships it as a small back-compat layer so existing ``import numpy.compat``
statements don't fail at import time.
"""

# Legacy aliases that some older code expects.
unicode = str
long = int
basestring = (str, bytes)

# Removed in numpy 2.x but historically lived here. rumpy's minimal
# rustpython build doesn't ship every codec, so we default to UTF-8 and
# fall back to a per-byte conversion when no codec is available.
def _fallback_bytes(s):
    return bytes(ord(c) & 0xFF for c in s)


def _fallback_str(b):
    return "".join(chr(c) for c in b)


def asbytes(s):
    if isinstance(s, bytes):
        return s
    try:
        return s.encode("utf-8")
    except (LookupError, UnicodeEncodeError):
        return _fallback_bytes(s)


def asstr(s):
    if isinstance(s, str):
        return s
    try:
        return s.decode("utf-8")
    except (LookupError, UnicodeDecodeError):
        return _fallback_str(s)


def asunicode(s):
    return asstr(s)


__all__ = [
    "unicode",
    "long",
    "basestring",
    "asbytes",
    "asstr",
    "asunicode",
]
