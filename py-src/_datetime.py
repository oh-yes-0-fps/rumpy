"""``datetime64`` / ``timedelta64`` family for rumpy.

Real numpy stores these as 64-bit integers + a unit tag and provides
full broadcasting arithmetic. rumpy wraps the host interpreter's
``datetime.datetime`` / ``datetime.timedelta`` (when available) so the
public surface (parsing strings, arithmetic, formatting) works for
scalar use. Array-of-datetimes is best-effort.

``_chrono_parse`` and ``_chrono_format`` are injected from Rust and use
the ``chrono`` crate for ISO 8601 parsing/formatting that the host
interpreter's ``datetime.fromisoformat`` may not cover (microsecond and
nanosecond precision strings).
"""


class datetime64:
    """Scalar datetime — wraps an ISO 8601 string or seconds-since-epoch int."""

    __slots__ = ("_iso", "_unit")

    def __init__(self, value=None, unit="us"):
        self._unit = unit
        if value is None or value == "NaT":
            self._iso = None
        elif isinstance(value, str):
            self._iso = _chrono_parse(value) if "_chrono_parse" in globals() else value
        elif isinstance(value, (int, float)):
            # Seconds (or units) since epoch.
            try:
                import datetime as _dt

                dt = _dt.datetime.utcfromtimestamp(float(value))
                self._iso = dt.isoformat()
            except Exception:
                self._iso = str(value)
        elif isinstance(value, datetime64):
            self._iso = value._iso
            self._unit = value._unit
        else:
            self._iso = str(value)

    def __repr__(self):
        if self._iso is None:
            return "numpy.datetime64('NaT')"
        return f"numpy.datetime64({self._iso!r})"

    def __str__(self):
        return "NaT" if self._iso is None else self._iso

    def __eq__(self, other):
        if isinstance(other, datetime64):
            return self._iso == other._iso
        return NotImplemented

    def __lt__(self, other):
        if isinstance(other, datetime64):
            return (self._iso or "") < (other._iso or "")
        return NotImplemented

    def __sub__(self, other):
        if isinstance(other, datetime64):
            import datetime as _dt

            if self._iso is None or other._iso is None:
                return timedelta64(None)
            a = _dt.datetime.fromisoformat(self._iso)
            b = _dt.datetime.fromisoformat(other._iso)
            return timedelta64(int((a - b).total_seconds() * 1_000_000), unit="us")
        if isinstance(other, timedelta64):
            import datetime as _dt

            if self._iso is None:
                return datetime64(None)
            a = _dt.datetime.fromisoformat(self._iso)
            td = _dt.timedelta(microseconds=other._micros or 0)
            return datetime64((a - td).isoformat(), unit=self._unit)
        return NotImplemented

    def __add__(self, other):
        if isinstance(other, timedelta64):
            import datetime as _dt

            if self._iso is None:
                return datetime64(None)
            a = _dt.datetime.fromisoformat(self._iso)
            td = _dt.timedelta(microseconds=other._micros or 0)
            return datetime64((a + td).isoformat(), unit=self._unit)
        return NotImplemented


class timedelta64:
    """Scalar timedelta — stored as integer microseconds (best-effort)."""

    __slots__ = ("_micros", "_unit")

    def __init__(self, value=None, unit="us"):
        self._unit = unit
        if value is None or value == "NaT":
            self._micros = None
        elif isinstance(value, (int, float)):
            scale = {
                "us": 1,
                "ms": 1000,
                "s": 1_000_000,
                "m": 60_000_000,
                "h": 3_600_000_000,
                "D": 86_400_000_000,
            }.get(unit, 1)
            self._micros = int(float(value) * scale)
        elif isinstance(value, timedelta64):
            self._micros = value._micros
            self._unit = value._unit
        else:
            self._micros = int(value)

    def __repr__(self):
        return f"numpy.timedelta64({self._micros!r},{self._unit!r})"

    def __str__(self):
        return "NaT" if self._micros is None else str(self._micros)

    def __int__(self):
        return self._micros or 0

    def __float__(self):
        return float(self._micros or 0)

    def __eq__(self, other):
        if isinstance(other, timedelta64):
            return self._micros == other._micros
        return NotImplemented

    def __add__(self, other):
        if isinstance(other, timedelta64):
            return timedelta64((self._micros or 0) + (other._micros or 0), unit="us")
        return NotImplemented

    def __sub__(self, other):
        if isinstance(other, timedelta64):
            return timedelta64((self._micros or 0) - (other._micros or 0), unit="us")
        return NotImplemented

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return timedelta64(int((self._micros or 0) * other), unit="us")
        return NotImplemented


def datetime_as_string(arr, unit="auto", timezone="naive", casting="same_kind"):
    """Convert a datetime64 array (or scalar) to a list of ISO strings."""
    _ = (unit, timezone, casting)
    if isinstance(arr, datetime64):
        return arr._iso or "NaT"
    return [str(x) for x in arr]


def datetime_data(dtype):
    """Return ``(unit, count)`` from a datetime dtype descriptor."""
    if isinstance(dtype, str):
        return (dtype.replace("datetime64[", "").replace("]", ""), 1)
    return ("us", 1)


def isnat(x):
    """Return True for NaT scalars / element-wise for arrays."""
    if isinstance(x, datetime64):
        return x._iso is None
    if isinstance(x, timedelta64):
        return x._micros is None
    if hasattr(x, "__iter__"):
        return [isnat(v) for v in x]
    return False


# Business day helpers ----------------------------------------------------


class busdaycalendar:
    """A business-day calendar — wraps a weekmask + list of holiday dates."""

    def __init__(self, weekmask="1111100", holidays=None):
        self.weekmask = weekmask
        self.holidays = list(holidays) if holidays else []


def _to_date(x):
    if isinstance(x, datetime64):
        return x._iso
    return str(x)


def _is_weekend(iso_str, weekmask="1111100"):
    """Return True if ``iso_str`` is a weekend day per ``weekmask``."""
    try:
        import datetime as _dt

        d = _dt.datetime.fromisoformat(iso_str).date()
        # Monday=0, Sunday=6; weekmask: MTWTFSS as "1111100".
        return weekmask[d.weekday()] == "0"
    except Exception:
        return False


def is_busday(dates, weekmask="1111100", holidays=None, busdaycal=None, out=None):
    """Element-wise test for business days."""
    _ = out
    if busdaycal is not None:
        weekmask = busdaycal.weekmask
        holidays = busdaycal.holidays
    holidays = set(holidays or [])

    def one(d):
        s = _to_date(d)
        return (not _is_weekend(s, weekmask)) and (s not in holidays)

    if isinstance(dates, (list, tuple)):
        return [one(d) for d in dates]
    return one(dates)


def busday_offset(
    dates,
    offsets,
    roll="raise",
    weekmask="1111100",
    holidays=None,
    busdaycal=None,
    out=None,
):
    """Offset each input by ``offsets`` business days. Best-effort scalar version."""
    _ = (roll, out)
    if busdaycal is not None:
        weekmask = busdaycal.weekmask
        holidays = busdaycal.holidays
    holidays = set(holidays or [])
    import datetime as _dt

    def one(d, off):
        s = _to_date(d)
        try:
            dt = _dt.datetime.fromisoformat(s)
            step = 1 if off >= 0 else -1
            n = abs(int(off))
            while n > 0:
                dt += _dt.timedelta(days=step)
                if (
                    not _is_weekend(dt.isoformat(), weekmask)
                    and dt.isoformat() not in holidays
                ):
                    n -= 1
            return datetime64(dt.isoformat())
        except Exception:
            return d

    if isinstance(dates, (list, tuple)):
        offs = offsets if isinstance(offsets, (list, tuple)) else [offsets] * len(dates)
        return [one(d, o) for d, o in zip(dates, offs)]
    return one(dates, offsets)


def busday_count(
    begindates, enddates, weekmask="1111100", holidays=None, busdaycal=None, out=None
):
    """Count business days between two dates (exclusive of enddate)."""
    _ = out
    if busdaycal is not None:
        weekmask = busdaycal.weekmask
        holidays = busdaycal.holidays
    holidays = set(holidays or [])
    import datetime as _dt

    def one(b, e):
        a = _dt.datetime.fromisoformat(_to_date(b)).date()
        z = _dt.datetime.fromisoformat(_to_date(e)).date()
        if a > z:
            return -one(e, b)
        count = 0
        cur = a
        while cur < z:
            iso = cur.isoformat()
            if not _is_weekend(iso, weekmask) and iso not in holidays:
                count += 1
            cur += _dt.timedelta(days=1)
        return count

    if isinstance(begindates, (list, tuple)):
        return [one(b, e) for b, e in zip(begindates, enddates)]
    return one(begindates, enddates)


__all__ = [
    "datetime64",
    "timedelta64",
    "datetime_as_string",
    "datetime_data",
    "isnat",
    "busdaycalendar",
    "is_busday",
    "busday_offset",
    "busday_count",
]
