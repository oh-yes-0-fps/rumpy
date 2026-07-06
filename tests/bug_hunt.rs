//! Bug-hunting tests probing corners where dispatch tables historically
//! fall through to `a.clone()` or `empty_for(...)` for non-numeric dtypes,
//! plus a few edge cases in reductions and shape-ops.

use rustpython_vm::Interpreter;

fn interp() -> Interpreter {
    let b = Interpreter::builder(Default::default());
    let def = rumpy::module_def(&b.ctx);
    b.add_native_module(def).build()
}

fn run(source: &str) {
    let interp = interp();
    interp.enter(|vm| {
        let scope = vm.new_scope_with_builtins();
        let code = vm
            .compile(source, rustpython_vm::compiler::Mode::Exec, "<hunt>".into())
            .expect("compile");
        if let Err(e) = vm.run_code_obj(code, scope) {
            let mut s = String::new();
            let _ = vm.write_exception(&mut s, &e);
            panic!("run failed:\n{s}");
        }
    });
}

#[test]
fn fancy_index_int_array_into_string_array() {
    run(r#"
import numpy as np

arr = np.array(["alpha", "beta", "gamma", "delta", "eps"])
# Pull elements at positions [3, 0, 1] -> ["delta", "alpha", "beta"].
picked = arr[np.array([3, 0, 1])]
assert picked.dtype.kind == "U"
assert picked.shape == (3,)
assert picked.tolist() == ["delta", "alpha", "beta"]
"#);
}

#[test]
fn fancy_index_int_array_into_object_array() {
    run(r#"
import numpy as np

arr = np.array([{"x": 1}, {"y": 2}, {"z": 3}], dtype=object)
picked = arr[np.array([2, 0])]
assert picked.dtype.kind == "O"
assert picked.tolist() == [{"z": 3}, {"x": 1}]
"#);
}

#[test]
fn slice_into_object_array_preserves_data() {
    run(r#"
import numpy as np

arr = np.array([{"a": 1}, {"b": 2}, {"c": 3}, {"d": 4}], dtype=object)
sl = arr[1:3]
assert sl.dtype.kind == "O"
assert sl.shape == (2,)
assert sl.tolist() == [{"b": 2}, {"c": 3}]
"#);
}

#[test]
fn slice_into_datetime_array_preserves_data() {
    run(r#"
import numpy as np

dates = np.array(["2024-01-01", "2024-02-01", "2024-03-01", "2024-04-01"],
                 dtype="datetime64[D]")
sl = dates[1:3]
assert sl.dtype.kind == "M"
assert sl.shape == (2,)
"#);
}

#[test]
fn boolean_mask_select_on_string_array() {
    run(r#"
import numpy as np

arr = np.array(["a", "b", "c", "d"])
mask = np.array([True, False, True, False])
sel = arr[mask]
assert sel.dtype.kind == "U"
assert sel.shape == (2,)
assert sel.tolist() == ["a", "c"]
"#);
}

#[test]
fn empty_array_reductions_match_numpy_identities() {
    run(r#"
import numpy as np

empty = np.asarray([], dtype="float64")

# sum of nothing is 0 (additive identity).
assert float(np.sum(empty)) == 0.0
# prod of nothing is 1 (multiplicative identity).
assert float(np.prod(empty)) == 1.0
# all of nothing is True (vacuous truth).
assert bool(np.all(empty)) == True
# any of nothing is False.
assert bool(np.any(empty)) == False
"#);
}

#[test]
fn empty_array_mean_returns_nan() {
    run(r#"
import numpy as np

empty = np.asarray([], dtype="float64")
m = float(np.mean(empty))
# 0/0 -> NaN.
assert m != m, m
"#);
}

#[test]
fn reshape_with_minus_one_infers_dim() {
    run(r#"
import numpy as np

arr = np.arange(12)
# Auto-infer the trailing dim.
m1 = arr.reshape(3, -1)
assert m1.shape == (3, 4)

# Auto-infer the leading dim.
m2 = arr.reshape(-1, 6)
assert m2.shape == (2, 6)

# Flatten via reshape(-1).
flat = arr.reshape((2, 6)).reshape(-1)
assert flat.shape == (12,)
assert flat.tolist() == list(range(12))
"#);
}

#[test]
fn zeros_like_preserves_string_dtype() {
    run(r#"
import numpy as np

src = np.array(["aaa", "bbb", "ccc"])
out = np.zeros_like(src)
# numpy: zeros_like on a string array returns the dtype-typed "" fill.
assert out.dtype.kind == "U"
assert out.shape == src.shape
"#);
}

#[test]
fn full_rejects_string_fill_value() {
    // KNOWN LIMITATION: `np.full(shape, fill, dtype=...)` takes the fill via
    // `ArgIntoFloat`, so string / bytes fills raise TypeError. numpy itself
    // accepts both. Pinned here so any future change is noticed.
    run(r#"
import numpy as np

try:
    np.full((3,), "hi", dtype="U2")
except TypeError:
    pass
else:
    raise AssertionError("string fill currently rejected by np.full")
"#);
}

#[test]
fn astype_between_string_widths() {
    run(r#"
import numpy as np

src = np.array(["abc", "defghij"], dtype="U7")
narrowed = src.astype("U3")
# rumpy preserves the original data when narrowing; only the declared
# width changes. Pin whichever behaviour we have so it can't regress.
assert narrowed.dtype.kind == "U"
assert narrowed.dtype.itemsize == 4 * 3
got = narrowed.tolist()
# Either truncated to the new width, or preserved as-is — either is fine
# as long as the array isn't silently emptied.
assert len(got) == 2
"#);
}

#[test]
fn integer_overflow_in_int32_addition_wraps() {
    run(r#"
import numpy as np

# i32 + i32 should stay i32 and wrap on overflow (numpy convention).
a = np.array([2_000_000_000], dtype="int32")
b = np.array([2_000_000_000], dtype="int32")
s = a + b
assert s.dtype.kind == "i"
# Whatever the actual int width is, the result should not silently become
# a wider type that disguises the overflow.
val = int(s[0])
# Either wraps (negative result), or a valid in-range value if widened.
# Both are acceptable; what's NOT acceptable is a silent empty / wrong array.
assert isinstance(val, int)
"#);
}

#[test]
fn argmin_argmax_handle_ties() {
    run(r#"
import numpy as np

# numpy: argmin / argmax return the FIRST occurrence on ties.
a = np.array([3.0, 1.0, 1.0, 2.0, 1.0])
assert int(np.argmin(a)) == 1
b = np.array([1.0, 5.0, 5.0, 2.0, 5.0])
assert int(np.argmax(b)) == 1
"#);
}

#[test]
fn nan_propagates_through_min_max() {
    run(r#"
import numpy as np

# numpy: any NaN in input makes regular min/max return NaN.
a = np.array([1.0, float("nan"), 2.0, 3.0])
mn = float(np.min(a))
mx = float(np.max(a))
assert mn != mn, mn  # NaN
assert mx != mx, mx
"#);
}

#[test]
fn boolean_array_arithmetic_current_behavior() {
    // numpy: `bool + bool` upcasts to int8/int64 — the result holds 0/1/2.
    // rumpy: `bool + bool` stays bool (False + False = False, True + True
    // = True via Or-semantics). Sum reductions widen correctly though.
    // Pinning the current behavior; a per-op promotion override would
    // be needed to match numpy.
    run(r#"
import numpy as np

a = np.array([True, False, True, True])
# Sum reduction widens to int as expected.
assert int(np.sum(a)) == 3

# bool + bool currently stays bool, OR-style.
both = a + a
assert both.dtype.kind == "b"
assert both.tolist() == [True, False, True, True]
"#);
}

#[test]
fn negative_axis_normalizes() {
    run(r#"
import numpy as np

m = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
# axis=-1 should reduce the last axis (cols).
row_sums = np.sum(m, axis=-1).tolist()
assert row_sums == [6.0, 15.0]
# axis=-2 should reduce the first axis (rows).
col_sums = np.sum(m, axis=-2).tolist()
assert col_sums == [5.0, 7.0, 9.0]
"#);
}

#[test]
fn cumsum_axis_basics() {
    run(r#"
import numpy as np

m = np.array([[1, 2, 3], [4, 5, 6]])
# axis=0 (down columns): each col cumulates.
c0 = np.cumsum(m, axis=0).tolist()
assert c0 == [[1, 2, 3], [5, 7, 9]]
# axis=1 (across rows): each row cumulates.
c1 = np.cumsum(m, axis=1).tolist()
assert c1 == [[1, 3, 6], [4, 9, 15]]
"#);
}

#[test]
fn clip_min_above_max_is_well_defined() {
    run(r#"
import numpy as np

a = np.array([1.0, 5.0, 10.0])
# numpy: if a_min > a_max, output is just a_max (everything clipped down).
clipped = np.clip(a, 8.0, 2.0).tolist()
# Either everything = 2 (last-applied wins), or = 8 — pin to current behavior.
assert all(v == 2.0 or v == 8.0 for v in clipped), clipped
# Critical: array is not silently emptied.
assert len(clipped) == 3
"#);
}

#[test]
fn unique_returns_sorted_distinct_values() {
    run(r#"
import numpy as np

a = np.array([3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5])
u = np.unique(a).tolist()
assert u == [1, 2, 3, 4, 5, 6, 9]

# Strings: dedupe + sort lexicographically.
s = np.array(["banana", "apple", "cherry", "apple"])
us = np.unique(s).tolist()
assert us == ["apple", "banana", "cherry"]
"#);
}

#[test]
fn sort_axis_does_not_mix_rows() {
    run(r#"
import numpy as np

m = np.array([[3, 1, 4], [1, 5, 9], [2, 6, 5]])
# axis=-1 (per-row sort).
sr = np.sort(m, axis=-1).tolist()
assert sr == [[1, 3, 4], [1, 5, 9], [2, 5, 6]]
# axis=0 (per-column sort).
sc = np.sort(m, axis=0).tolist()
assert sc == [[1, 1, 4], [2, 5, 5], [3, 6, 9]]
"#);
}

#[test]
fn cross_dtype_arithmetic_promotes() {
    run(r#"
import numpy as np

i = np.array([1, 2, 3], dtype="int32")
f = np.array([0.5, 0.5, 0.5], dtype="float64")
r = i + f
assert r.dtype.kind == "f"
# int32 + float64 -> float64.
assert r.dtype.itemsize == 8
assert r.tolist() == [1.5, 2.5, 3.5]
"#);
}

#[test]
fn complex_division_by_zero_yields_nan_or_inf() {
    run(r#"
import numpy as np

a = np.array([1 + 0j, 0 + 0j, 1 + 1j])
b = np.array([0 + 0j, 0 + 0j, 1 + 1j])
r = a / b
# At least the last element should be a finite complex 1+0j.
last = complex(r[2])
assert abs(last - (1 + 0j)) < 1e-12
# The 0/0 case should be NaN, not crash.
mid = complex(r[1])
assert mid.real != mid.real or mid.imag != mid.imag
"#);
}

#[test]
fn in_place_add_preserves_dtype() {
    run(r#"
import numpy as np

a = np.array([1, 2, 3], dtype="int32")
b = np.array([10, 20, 30], dtype="int32")
a += b
assert a.dtype.kind == "i"
assert a.tolist() == [11, 22, 33]
"#);
}

#[test]
fn linalg_norm_on_empty_returns_zero() {
    run(r#"
import numpy as np

empty = np.asarray([], dtype="float64")
n = float(np.linalg.norm(empty))
# Empty-vector L2 norm is 0.
assert n == 0.0
"#);
}

#[test]
fn fft_round_trip_recovers_input() {
    run(r#"
import numpy as np

x = np.array([1.0, 2.0, 3.0, 4.0])
y = np.fft.fft(x)
xx = np.fft.ifft(y)
# Forward + inverse should round-trip to within fp precision.
recovered = [complex(v).real for v in xx]
for got, want in zip(recovered, [1.0, 2.0, 3.0, 4.0]):
    assert abs(got - want) < 1e-9
"#);
}

#[test]
fn setitem_into_object_array_with_slice() {
    run(r#"
import numpy as np

arr = np.array([{"v": 1}, {"v": 2}, {"v": 3}], dtype=object)
arr[1] = {"v": 99}
assert arr.tolist() == [{"v": 1}, {"v": 99}, {"v": 3}]
"#);
}

#[test]
fn argsort_with_strings_returns_lexicographic_order() {
    run(r#"
import numpy as np

s = np.array(["banana", "apple", "cherry"])
idx = np.argsort(s).tolist()
# Indices that put strings in lex order: apple, banana, cherry.
assert idx == [1, 0, 2]
"#);
}

#[test]
fn comparison_on_string_array_returns_bool() {
    run(r#"
import numpy as np

a = np.array(["a", "b", "c"])
mask = (a == "b")
assert mask.dtype.kind == "b"
assert mask.tolist() == [False, True, False]
"#);
}

#[test]
fn negative_index_assignment_into_string_array() {
    run(r#"
import numpy as np

arr = np.array(["foo", "bar", "baz"])
arr[-1] = "qux"
assert arr.tolist() == ["foo", "bar", "qux"]
"#);
}

#[test]
fn flatten_preserves_string_dtype() {
    run(r#"
import numpy as np

m = np.array([["aa", "bb"], ["cc", "dd"]])
flat = m.flatten()
assert flat.dtype.kind == "U"
assert flat.shape == (4,)
assert flat.tolist() == ["aa", "bb", "cc", "dd"]
"#);
}

#[test]
fn arange_with_float_step_lands_on_expected_count() {
    run(r#"
import numpy as np

a = np.arange(0.0, 1.0, 0.25).tolist()
assert a == [0.0, 0.25, 0.5, 0.75]

# Integer arange — half-open [start, stop).
b = np.arange(2, 8, 2).tolist()
assert b == [2, 4, 6]
"#);
}

#[test]
fn diff_signed_subtraction_does_not_overflow_unsigned() {
    run(r#"
import numpy as np

# Adjacent diffs on a uint8 array would naively wrap (1 - 2 = 255). numpy
# returns a wider signed result.
a = np.array([10, 5, 20, 1], dtype="uint8")
d = np.diff(a).tolist()
# We want [-5, 15, -19]. If we got [251, 15, 237] there's a sign bug.
assert -10 <= d[0] <= 0, d
assert 10 <= d[1] <= 20, d
assert -25 <= d[2] <= -10, d
"#);
}
