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
        body = ", ".join(
            f"{n}={v!r}" for n, v in zip(self._names, self._values)
        )
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


# ----------------- public constructors -----------------

def fromarrays(array_list, names):
    """Construct a recarray from a list of equal-length per-column arrays."""
    names = _parse_names(names)
    if len(array_list) != len(names):
        raise ValueError(
            "fromarrays: len(array_list) != len(names)"
        )
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


def fromstring(*args, **kwargs):
    raise NotImplementedError("numpy.rec.fromstring needs typed binary buffers")


def fromfile(*args, **kwargs):
    raise NotImplementedError("numpy.rec.fromfile needs typed binary buffers")


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
