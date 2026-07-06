"""``numpy.rec`` — record-array helpers (bare-bones).

This is a pure-Python record array: a list of records, where each record
exposes its fields both by name (attribute access) and by position
(indexing). ``recarray`` wraps the row list and supports column access
via attribute lookup (returning a list of per-row values).

It's not a true numpy structured ndarray — no shared dtype-described
memory layout — but it's enough for code that just wants to construct,
iterate, and field-access record-like data.
"""


def _parse_names(names):
    if isinstance(names, str):
        return [n.strip() for n in names.split(",")]
    return list(names)


class record:
    """A single record. Fields are accessible by name or by index."""

    __slots__ = ("_names", "_values")

    def __init__(self, names, values):
        names = list(names)
        values = list(values)
        if len(names) != len(values):
            raise ValueError(
                f"record: names ({len(names)}) and values ({len(values)}) "
                "must have the same length"
            )
        # Bypass __setattr__ for the two internal slots.
        object.__setattr__(self, "_names", names)
        object.__setattr__(self, "_values", values)

    def __getattr__(self, name):
        names = object.__getattribute__(self, "_names")
        if name in names:
            return self._values[names.index(name)]
        raise AttributeError(name)

    def __setattr__(self, name, value):
        names = object.__getattribute__(self, "_names")
        if name in names:
            self._values[names.index(name)] = value
            return
        object.__setattr__(self, name, value)

    def __getitem__(self, idx):
        if isinstance(idx, str):
            return self._values[self._names.index(idx)]
        return self._values[idx]

    def __setitem__(self, idx, value):
        if isinstance(idx, str):
            self._values[self._names.index(idx)] = value
        else:
            self._values[idx] = value

    def __iter__(self):
        return iter(self._values)

    def __len__(self):
        return len(self._values)

    def __eq__(self, other):
        if isinstance(other, record):
            return self._names == other._names and self._values == other._values
        if isinstance(other, (list, tuple)):
            return list(self._values) == list(other)
        return NotImplemented

    def __repr__(self):
        body = ", ".join(f"{n}={v!r}" for n, v in zip(self._names, self._values))
        return f"record({body})"

    @property
    def dtype_names(self):
        return tuple(self._names)


class recarray:
    """A list of ``record`` rows. Column access returns a list."""

    def __init__(self, rows, names):
        names = _parse_names(names)
        self._names = names
        # Normalize each row to a `record`.
        self._rows = []
        for r in rows:
            if isinstance(r, record):
                if list(r._names) != names:
                    raise ValueError("recarray: row field names mismatch")
                self._rows.append(r)
            else:
                self._rows.append(record(names, list(r)))

    # ----- column access (by name) -----
    def __getattr__(self, name):
        # __getattr__ is only consulted on miss, so this is only for fields.
        names = object.__getattribute__(self, "_names")
        if name in names:
            i = names.index(name)
            return [r[i] for r in self._rows]
        raise AttributeError(name)

    def __setattr__(self, name, value):
        if name in ("_names", "_rows"):
            object.__setattr__(self, name, value)
            return
        if name in self._names:
            i = self._names.index(name)
            for r, v in zip(self._rows, value):
                r[i] = v
            return
        object.__setattr__(self, name, value)

    # ----- row access (by index) -----
    def __getitem__(self, idx):
        if isinstance(idx, str):
            return self.__getattr__(idx)
        if isinstance(idx, slice):
            return recarray(self._rows[idx], self._names)
        return self._rows[idx]

    def __setitem__(self, idx, value):
        if isinstance(idx, str):
            self.__setattr__(idx, value)
            return
        self._rows[idx] = (
            value if isinstance(value, record) else record(self._names, list(value))
        )

    def __iter__(self):
        return iter(self._rows)

    def __len__(self):
        return len(self._rows)

    def __eq__(self, other):
        if isinstance(other, recarray):
            return self._names == other._names and self._rows == other._rows
        return NotImplemented

    def __repr__(self):
        return f"recarray({self._rows!r}, names={self._names!r})"

    @property
    def names(self):
        return tuple(self._names)

    @property
    def shape(self):
        return (len(self._rows),)

    @property
    def size(self):
        return len(self._rows)

    def tolist(self):
        return [list(r) for r in self._rows]


def fromarrays(array_list, names):
    """Construct a recarray from a list of equal-length per-column arrays."""
    names = _parse_names(names)
    if len(array_list) != len(names):
        raise ValueError("fromarrays: len(array_list) != len(names)")
    cols = [list(a) for a in array_list]
    if cols and any(len(c) != len(cols[0]) for c in cols):
        raise ValueError("fromarrays: column lengths differ")
    n = len(cols[0]) if cols else 0
    rows = [tuple(cols[c][i] for c in range(len(cols))) for i in range(n)]
    return recarray(rows, names)


def fromrecords(records_list, names):
    """Construct a recarray from a list of row tuples."""
    return recarray(records_list, names)


def array(obj, names=None):
    """Best-effort recarray constructor mirroring numpy's overloaded form."""
    if isinstance(obj, recarray):
        return obj
    if names is None:
        raise ValueError("rec.array: names is required")
    return fromrecords(obj, names)


def find_duplicate(names):
    """Return the list of duplicated names (preserves first-seen order)."""
    seen = set()
    dups = []
    for n in names:
        if n in seen and n not in dups:
            dups.append(n)
        seen.add(n)
    return dups


class format_parser:
    """Tiny parser placeholder: stores formats/names/titles as attributes."""

    def __init__(self, formats, names, titles=None, aligned=False, byteorder=None):
        _ = aligned
        _ = byteorder
        self.formats = formats
        self.names = _parse_names(names) if names else []
        self.titles = list(titles) if titles else []


_FORMAT_CODES = {
    "b1": ("int", 1, False),
    "i1": ("int", 1, True),
    "i2": ("int", 2, True),
    "i4": ("int", 4, True),
    "i8": ("int", 8, True),
    "u1": ("int", 1, False),
    "u2": ("int", 2, False),
    "u4": ("int", 4, False),
    "u8": ("int", 8, False),
    "f4": ("float", 4, True),
    "f8": ("float", 8, True),
}


def _parse_formats(formats):
    """Split a numpy-style formats spec into ``[(kind, size, signed), …]``.

    Trailing fixed-width byte-string fields (``S<n>``) come back as
    ``("bytes", n, False)``. Optional leading endian markers on individual
    fields (``<i4``, ``>f8``) are stripped.
    """
    if isinstance(formats, str):
        items = [f.strip() for f in formats.split(",")]
    else:
        items = [str(f).strip() for f in formats]
    parsed = []
    for f in items:
        if f.startswith(("<", ">", "=", "|")):
            f = f[1:]
        if f in _FORMAT_CODES:
            parsed.append(_FORMAT_CODES[f])
        elif len(f) > 1 and f[0] in ("S", "a"):
            try:
                n = int(f[1:])
            except ValueError as exc:
                raise ValueError(f"rec: invalid byte-string format {f!r}") from exc
            parsed.append(("bytes", n, False))
        else:
            raise ValueError(f"rec: unsupported format code {f!r}")
    return parsed


def _decode_float(buf, big_endian):
    """IEEE-754 decode for 4- or 8-byte buffers."""
    n = len(buf)
    bits = int.from_bytes(buf, "big" if big_endian else "little", signed=False)
    if n == 4:
        sign = (bits >> 31) & 0x1
        exp = (bits >> 23) & 0xFF
        mant = bits & 0x7FFFFF
        bias = 127
        mant_bits = 23
    elif n == 8:
        sign = (bits >> 63) & 0x1
        exp = (bits >> 52) & 0x7FF
        mant = bits & ((1 << 52) - 1)
        bias = 1023
        mant_bits = 52
    else:
        raise ValueError(f"_decode_float: unsupported width {n}")
    if exp == (1 << (8 if n == 4 else 11)) - 1:
        if mant == 0:
            return float("-inf") if sign else float("inf")
        return float("nan")
    if exp == 0:
        value = mant * (2.0 ** (1 - bias - mant_bits))
    else:
        value = (1.0 + mant * (2.0**-mant_bits)) * (2.0 ** (exp - bias))
    return -value if sign else value


def _unpack_record(raw, parsed, big_endian):
    pos = 0
    values = []
    for kind, size, signed in parsed:
        chunk = raw[pos : pos + size]
        pos += size
        if kind == "int":
            values.append(
                int.from_bytes(
                    chunk,
                    "big" if big_endian else "little",
                    signed=signed,
                )
            )
        elif kind == "float":
            values.append(_decode_float(chunk, big_endian))
        else:
            values.append(bytes(chunk).rstrip(b"\x00"))
    return tuple(values)


def fromstring(
    datastring,
    dtype=None,
    shape=None,
    offset=0,
    formats=None,
    names=None,
    titles=None,
    aligned=False,
    byteorder=None,
):
    """Parse a typed binary buffer into a :class:`recarray`.

    Only the ``formats``/``names`` route is supported (rumpy's rec is not
    backed by a structured ndarray, so a full ``dtype`` argument can't carry
    field info). Set ``byteorder`` to ``"<"``/``">"``/``"big"``/``"little"``
    to override the little-endian default.
    """
    _ = (dtype, titles, aligned)
    if formats is None:
        raise NotImplementedError(
            "rec.fromstring: only the formats=... route is supported"
        )
    if names is None:
        raise ValueError("rec.fromstring: names is required alongside formats")

    parsed = _parse_formats(formats)
    name_list = _parse_names(names)
    if len(name_list) != len(parsed):
        raise ValueError(
            f"rec.fromstring: {len(name_list)} names but {len(parsed)} formats"
        )
    big_endian = byteorder in (">", "big")
    record_size = sum(size for _kind, size, _signed in parsed)

    raw = bytes(datastring)[offset:]
    n_avail = len(raw) // record_size if record_size else 0
    if shape is None:
        n = n_avail
    else:
        n = shape[0] if isinstance(shape, tuple) else int(shape)
        if n < 0:
            n = n_avail
        if n > n_avail:
            raise ValueError(
                f"rec.fromstring: requested {n} records, buffer holds {n_avail}"
            )

    rows = [
        _unpack_record(raw[i * record_size : (i + 1) * record_size], parsed, big_endian)
        for i in range(n)
    ]
    return fromrecords(rows, name_list)


def fromfile(
    fd,
    dtype=None,
    shape=None,
    offset=0,
    formats=None,
    names=None,
    titles=None,
    aligned=False,
    byteorder=None,
):
    """Read records from a file-like / path source. See :func:`fromstring`."""
    if hasattr(fd, "read"):
        data = fd.read()
    else:
        with open(fd, "rb") as fh:
            data = fh.read()
    return fromstring(
        data,
        dtype=dtype,
        shape=shape,
        offset=offset,
        formats=formats,
        names=names,
        titles=titles,
        aligned=aligned,
        byteorder=byteorder,
    )


__all__ = [
    "record",
    "recarray",
    "fromarrays",
    "fromrecords",
    "array",
    "find_duplicate",
    "format_parser",
    "fromstring",
    "fromfile",
]
