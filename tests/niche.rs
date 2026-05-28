//! Niche / edge-case coverage for things the main test suites only touch
//! superficially:
//!   * `numpy.roots` corner cases (leading zeros, repeated roots, …)
//!   * `linalg.matrix_power` (n=0, n=1, negative)
//!   * `linalg.slogdet` on singular vs. well-conditioned matrices
//!   * `linalg.pinv` for rectangular inputs (left + right inverse identities)
//!   * `numpy.cross`, `unwrap`, `percentile`, `cov`, `corrcoef`
//!   * `numpy.random.seed` reproducibility
//!   * `rec.fromstring` across i1/u1/u4/f4 dtypes and `offset=`
//!   * `ma.count(axis=…)` with fully-masked slices
//!
//! All tests run inside a RustPython VM with the rumpy native module
//! registered, mirroring the pattern in `tests/typing.rs`.

use rustpython_vm::Interpreter;

fn interp() -> Interpreter {
    let b = Interpreter::builder(Default::default());
    let def = rumpy::module_def(&b.ctx);
    b.add_native_module(def).build()
}

fn run(source: &str) {
    let interp = interp();
    interp.enter(|vm| {
        let numpy_mod = vm.import("numpy", 0).expect("import numpy");
        let sys_modules = vm
            .sys_module
            .get_attr("modules", vm)
            .expect("sys.modules");
        for sub in ["rec", "ma", "testing", "polynomial", "random"] {
            if let Ok(m) = numpy_mod.get_attr(sub, vm) {
                let dotted = format!("numpy.{sub}");
                let _ = sys_modules.set_item(dotted.as_str(), m, vm);
            }
        }
        let scope = vm.new_scope_with_builtins();
        let code = vm
            .compile(source, rustpython_vm::compiler::Mode::Exec, "<niche>".into())
            .expect("compile");
        if let Err(e) = vm.run_code_obj(code, scope) {
            let mut s = String::new();
            let _ = vm.write_exception(&mut s, &e);
            panic!("run failed:\n{s}");
        }
    });
}

#[test]
fn roots_handles_leading_zeros_and_repeated_roots() {
    run(
        r#"
import numpy as np

# Leading zero coefficient is stripped before forming the companion matrix.
# 0*x^2 + 1*x - 2 = 0  -> single root x = 2.
r = np.roots([0.0, 1.0, -2.0])
assert len(r) == 1
assert abs(complex(r[0]) - 2.0) < 1e-10

# Constant polynomial -> no roots, empty complex128 array.
r = np.roots([3.0])
assert len(r) == 0

# (x-1)^3 -> three roots all clustering at 1.0.
r = np.roots([1.0, -3.0, 3.0, -1.0])
assert len(r) == 3
for z in r:
    assert abs(complex(z) - 1.0) < 1e-4
"#,
    );
}

#[test]
fn roots_complex_pair_for_real_polynomial() {
    run(
        r#"
import numpy as np

# x^2 + 1 = 0  ->  roots are ±i.
r = sorted(np.roots([1.0, 0.0, 1.0]), key=lambda z: complex(z).imag)
assert len(r) == 2
z0, z1 = complex(r[0]), complex(r[1])
assert abs(z0 - (-1j)) < 1e-10
assert abs(z1 - 1j) < 1e-10
"#,
    );
}

#[test]
fn matrix_power_zero_one_and_negative() {
    run(
        r#"
import numpy as np

a = np.asarray([[2.0, 1.0], [0.0, 3.0]])

# n=0 -> identity, regardless of `a`.
eye = np.linalg.matrix_power(a, 0)
assert eye.tolist() == [[1.0, 0.0], [0.0, 1.0]]

# n=1 -> `a` itself.
one = np.linalg.matrix_power(a, 1)
assert one.tolist() == a.tolist()

# n=-1 -> inv(a). Verify via a @ inv(a) ≈ I.
inv = np.linalg.matrix_power(a, -1)
prod = a @ inv
for i in range(2):
    for j in range(2):
        target = 1.0 if i == j else 0.0
        assert abs(prod[i, j] - target) < 1e-10
"#,
    );
}

#[test]
fn slogdet_signs_for_singular_and_dense() {
    run(
        r#"
import numpy as np

# Well-conditioned: sign=+1, logabs = ln|det|.
a = np.asarray([[2.0, 0.0], [0.0, 3.0]])
sign, logabs = np.linalg.slogdet(a)
assert abs(float(sign) - 1.0) < 1e-12
assert abs(float(logabs) - np.log(6.0)) < 1e-10

# Negative determinant: sign=-1.
b = np.asarray([[0.0, 1.0], [1.0, 0.0]])
sign, logabs = np.linalg.slogdet(b)
assert abs(float(sign) - (-1.0)) < 1e-12
assert abs(float(logabs) - 0.0) < 1e-10

# Singular matrix: sign=0, logabs=-inf.
s = np.asarray([[1.0, 2.0], [2.0, 4.0]])
sign, logabs = np.linalg.slogdet(s)
assert float(sign) == 0.0
assert float(logabs) == float("-inf")
"#,
    );
}

#[test]
fn pinv_left_and_right_inverse_identities() {
    run(
        r#"
import numpy as np

# Tall (3x2) full column rank: pinv is the left inverse — pinv @ A == I_2.
a = np.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
p = np.linalg.pinv(a)
left = p @ a
for i in range(2):
    for j in range(2):
        target = 1.0 if i == j else 0.0
        assert abs(left[i, j] - target) < 1e-9

# Wide (2x3) full row rank: pinv is the right inverse — A @ pinv == I_2.
b = np.asarray([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])
q = np.linalg.pinv(b)
right = b @ q
for i in range(2):
    for j in range(2):
        target = 1.0 if i == j else 0.0
        assert abs(right[i, j] - target) < 1e-9
"#,
    );
}

#[test]
fn cross_product_3vectors() {
    run(
        r#"
import numpy as np

x = np.asarray([1.0, 0.0, 0.0])
y = np.asarray([0.0, 1.0, 0.0])
z = np.cross(x, y)
assert z.tolist() == [0.0, 0.0, 1.0]

# Anti-symmetry: a x b == -(b x a).
a = np.asarray([1.0, 2.0, 3.0])
b = np.asarray([4.0, 5.0, 6.0])
fwd = np.cross(a, b).tolist()
back = np.cross(b, a).tolist()
for f, r in zip(fwd, back):
    assert abs(f + r) < 1e-12
"#,
    );
}

#[test]
fn unwrap_undoes_wraparound() {
    run(
        r#"
import numpy as np

# Phase that wraps from +pi to -pi at index 2.
pi = float(np.pi)
phase = np.asarray([0.0, 0.9 * pi, -0.9 * pi, 0.0])
u = np.unwrap(phase).tolist()
# The big negative jump should be lifted by 2*pi.
assert abs(u[2] - (-0.9 * pi + 2 * pi)) < 1e-10
# Differences between successive unwrapped points stay below pi.
for prev, cur in zip(u[:-1], u[1:]):
    assert abs(cur - prev) <= pi + 1e-10
"#,
    );
}

#[test]
fn percentile_matches_known_quantiles() {
    run(
        r#"
import numpy as np

a = np.asarray([1.0, 2.0, 3.0, 4.0, 5.0])
# Median.
m = float(np.percentile(a, 50.0))
assert abs(m - 3.0) < 1e-12
# Min / max via 0th / 100th percentile.
assert abs(float(np.percentile(a, 0.0)) - 1.0) < 1e-12
assert abs(float(np.percentile(a, 100.0)) - 5.0) < 1e-12
# quantile uses the same engine but takes fractions.
assert abs(float(np.quantile(a, 0.5)) - 3.0) < 1e-12
"#,
    );
}

#[test]
fn cov_and_corrcoef_perfectly_correlated() {
    run(
        r#"
import numpy as np

# Two rows that are perfectly linearly related (y = 2x).
m = np.asarray([[1.0, 2.0, 3.0, 4.0, 5.0],
                [2.0, 4.0, 6.0, 8.0, 10.0]])
c = np.cov(m, 1)
# Variance of x = 2.5, variance of y = 10.0, cov(x,y) = 5.0.
assert abs(float(c[0, 0]) - 2.5) < 1e-10
assert abs(float(c[1, 1]) - 10.0) < 1e-10
assert abs(float(c[0, 1]) - 5.0) < 1e-10

# Correlation between perfectly correlated rows is 1.0.
r = np.corrcoef(m)
assert abs(float(r[0, 1]) - 1.0) < 1e-10
assert abs(float(r[1, 0]) - 1.0) < 1e-10
"#,
    );
}

#[test]
fn random_seed_is_reproducible() {
    run(
        r#"
import numpy as np

np.random.seed(42)
a = np.random.rand(5).tolist()
np.random.seed(42)
b = np.random.rand(5).tolist()
assert a == b

# Different seed -> different sequence (overwhelmingly likely).
np.random.seed(43)
c = np.random.rand(5).tolist()
assert a != c
"#,
    );
}

#[test]
fn rec_fromstring_all_supported_dtypes() {
    run(
        r#"
import numpy.rec as r

# Single record carrying every numeric format code we support, plus an
# S-string trailer. Built by hand without `struct` (which the no-stdlib
# embedded rumpy doesn't ship).
def le_int(v, n, signed=False):
    if signed and v < 0:
        v += 1 << (8 * n)
    return v.to_bytes(n, "little")

buf = (
    le_int(-1, 1, signed=True)   # i1 = -1
    + le_int(200, 1)              # u1 = 200
    + le_int(-1000, 2, signed=True)  # i2 = -1000
    + le_int(60000, 2)             # u2 = 60000
    + le_int(-123456, 4, signed=True)  # i4
    + le_int(4000000000, 4)        # u4
    + bytes.fromhex("0000803F")    # f4 = 1.0
    + bytes.fromhex("000000000000F03F")  # f8 = 1.0
    + b"hi"                        # S2
)
arr = r.fromstring(
    buf,
    formats="i1,u1,i2,u2,i4,u4,f4,f8,S2",
    names="a,b,c,d,e,f,g,h,tag",
)
rec = arr[0]
assert rec.a == -1
assert rec.b == 200
assert rec.c == -1000
assert rec.d == 60000
assert rec.e == -123456
assert rec.f == 4000000000
assert abs(rec.g - 1.0) < 1e-6
assert abs(rec.h - 1.0) < 1e-12
assert rec.tag == b"hi"
"#,
    );
}

#[test]
fn rec_fromstring_offset_skips_prefix() {
    run(
        r#"
import numpy.rec as r

# Prefix 4 bytes of garbage, then two i4 records {7, 9}.
prefix = b"\xde\xad\xbe\xef"
payload = (7).to_bytes(4, "little") + (9).to_bytes(4, "little")
buf = prefix + payload
arr = r.fromstring(buf, formats="i4", names="v", offset=4)
assert len(arr) == 2
assert arr[0].v == 7
assert arr[1].v == 9
"#,
    );
}

#[test]
fn ma_count_with_fully_masked_axis_returns_zero() {
    run(
        r#"
import numpy
ma = numpy.ma

# Row 1 is fully masked; row 0 is fully unmasked.
m = ma.masked_array(
    [[10, 20, 30], [40, 50, 60]],
    mask=[[False, False, False], [True, True, True]],
)
# axis=1 (per-row): row 0 has 3 unmasked, row 1 has 0.
got = m.count(axis=1).tolist()
assert got == [3, 0]
# axis=0 (per-column): each column has the unmasked row 0 cell only.
got0 = m.count(axis=0).tolist()
assert got0 == [1, 1, 1]
"#,
    );
}

#[test]
fn norm_ord_variants_match_known_values() {
    run(
        r#"
import numpy as np

# 1-D vector norms.
v = np.asarray([3.0, -4.0])
assert abs(float(np.linalg.norm(v)) - 5.0) < 1e-12          # default = L2
assert abs(float(np.linalg.norm(v, ord=1)) - 7.0) < 1e-12   # |3|+|4|
assert abs(float(np.linalg.norm(v, ord=float("inf"))) - 4.0) < 1e-12
assert abs(float(np.linalg.norm(v, ord=-float("inf"))) - 3.0) < 1e-12

# Frobenius norm of a 2x2 matrix.
m = np.asarray([[1.0, 2.0], [3.0, 4.0]])
fro = float(np.linalg.norm(m, ord="fro"))
assert abs(fro - (1 + 4 + 9 + 16) ** 0.5) < 1e-12
"#,
    );
}

#[test]
fn ndarray_iterates_over_first_axis() {
    run(
        r#"
import numpy as np

# 1-D: yields plain Python scalars (one per element).
a = np.asarray([10.0, 20.0, 30.0])
collected = [x for x in a]
assert collected == [10.0, 20.0, 30.0]

# 2-D: yields sub-arrays (rows). Each row is itself iterable.
m = np.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
rows = list(m)
assert len(rows) == 3
assert rows[0].tolist() == [1.0, 2.0]
assert rows[1].tolist() == [3.0, 4.0]
assert rows[2].tolist() == [5.0, 6.0]

# Nested iteration walks both axes (matches numpy semantics).
flat = [v for row in m for v in row]
assert flat == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

# 3-D: each step peels off one axis.
cube = np.asarray([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
sheets = list(cube)
assert len(sheets) == 2
assert sheets[0].shape == (2, 2)
assert sheets[1].tolist() == [[5, 6], [7, 8]]

# Empty along axis 0: iteration produces nothing, no error.
empty = np.asarray([])
assert list(empty) == []

# 0-D array: iteration is a TypeError, mirroring numpy.
scalar = np.asarray(42.0)
try:
    iter(scalar)
except TypeError:
    pass
else:
    raise AssertionError("0-d iter should raise TypeError")

# Comprehension equivalence: tolist matches list comprehension on the rows.
assert [r.tolist() for r in m] == m.tolist()
"#,
    );
}

#[test]
fn ndarray_iter_interops_with_python_builtins() {
    run(
        r#"
import numpy as np

a = np.asarray([1.0, 2.0, 3.0, 4.0])

# list / tuple constructors.
assert list(a) == [1.0, 2.0, 3.0, 4.0]
assert tuple(a) == (1.0, 2.0, 3.0, 4.0)

# Reducers fed an iterator.
assert sum(a) == 10.0
assert min(a) == 1.0
assert max(a) == 4.0

# enumerate / zip both consume the iterator protocol.
assert list(enumerate(a)) == [(0, 1.0), (1, 2.0), (2, 3.0), (3, 4.0)]
b = np.asarray([10.0, 20.0, 30.0, 40.0])
assert list(zip(a, b)) == [(1.0, 10.0), (2.0, 20.0), (3.0, 30.0), (4.0, 40.0)]

# next() walks one item at a time.
it = iter(a)
assert next(it) == 1.0
assert next(it) == 2.0
assert next(it) == 3.0
assert next(it) == 4.0
try:
    next(it)
except StopIteration:
    pass
else:
    raise AssertionError("expected StopIteration")

# Tuple-unpacking syntax (driven by __iter__).
x, y, z, w = a
assert (x, y, z, w) == (1.0, 2.0, 3.0, 4.0)

# `in` operator uses __iter__ when __contains__ isn't specialized.
assert 3.0 in a
assert 99.0 not in a
"#,
    );
}

#[test]
fn ndarray_iter_independence_and_snapshot() {
    run(
        r#"
import numpy as np

a = np.asarray([1.0, 2.0, 3.0])

# Two iterators from the same array advance independently.
it1 = iter(a)
it2 = iter(a)
assert next(it1) == 1.0
assert next(it2) == 1.0
assert next(it1) == 2.0
assert next(it2) == 2.0
assert next(it1) == 3.0

# Snapshot semantics: rumpy materialises the iter at __iter__ time, so
# mutating the array after iter() doesn't change what the iterator emits.
b = np.asarray([10.0, 20.0, 30.0])
snap = iter(b)
b[0] = 999.0
assert next(snap) == 10.0
"#,
    );
}

#[test]
fn ndarray_iter_preserves_dtype_and_sub_shape() {
    run(
        r#"
import numpy as np

# Int 1-D: each yielded item is a Python int (numpy collapses 0-d to scalar).
ai = np.asarray([1, 2, 3], dtype="int64")
items = list(ai)
assert items == [1, 2, 3]
assert all(isinstance(v, int) for v in items)

# Complex 1-D: yields Python complex scalars.
ac = np.asarray([1 + 2j, 3 + 4j])
items = list(ac)
assert items == [1 + 2j, 3 + 4j]
assert all(isinstance(v, complex) for v in items)

# 2-D: each row keeps its dtype.
m = np.asarray([[1, 2], [3, 4]], dtype="int32")
rows = list(m)
for row in rows:
    assert str(row.dtype) == "int32"
assert rows[0].tolist() == [1, 2]
assert rows[1].tolist() == [3, 4]

# 3-D: peels off one axis, the rest of the shape survives.
cube = np.zeros((2, 3, 4))
sheets = list(cube)
assert len(sheets) == 2
for sheet in sheets:
    assert sheet.shape == (3, 4)
"#,
    );
}

#[test]
fn ndarray_iter_round_trip_through_asarray() {
    run(
        r#"
import numpy as np

# Feeding the iter result back into np.asarray reproduces the original.
a = np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
rebuilt = np.asarray([row.tolist() for row in a])
assert rebuilt.shape == a.shape
assert rebuilt.tolist() == a.tolist()

# Iterating a sliced view yields the slice's rows, not the original's.
view = a[:, 1:]
collected = [row.tolist() for row in view]
assert collected == [[2.0, 3.0], [5.0, 6.0]]
"#,
    );
}

#[test]
fn set_ops_intersect_union_setdiff() {
    run(
        r#"
import numpy as np

a = np.asarray([3, 1, 4, 1, 5, 9, 2, 6])
b = np.asarray([5, 3, 5, 8, 9, 7])

# intersect1d returns the sorted unique intersection.
inter = np.intersect1d(a, b).tolist()
assert inter == [3, 5, 9]

# union1d returns the sorted unique union.
uni = np.union1d(a, b).tolist()
assert uni == [1, 2, 3, 4, 5, 6, 7, 8, 9]

# setdiff1d is the sorted unique values in `a` that aren't in `b`.
diff = np.setdiff1d(a, b).tolist()
assert diff == [1, 2, 4, 6]
"#,
    );
}

#[test]
fn linalg_cond_reflects_conditioning() {
    run(
        r#"
import numpy as np

# Identity has condition number 1.
eye = np.asarray([[1.0, 0.0], [0.0, 1.0]])
assert abs(float(np.linalg.cond(eye)) - 1.0) < 1e-10

# Diagonal(1, 100): cond = max_sv / min_sv = 100.
diag = np.asarray([[1.0, 0.0], [0.0, 100.0]])
assert abs(float(np.linalg.cond(diag)) - 100.0) < 1e-8

# A singular matrix has a non-finite or astronomical condition number.
sing = np.asarray([[1.0, 2.0], [2.0, 4.0]])
c = float(np.linalg.cond(sing))
is_inf = c == float("inf") or c == float("-inf")
is_nan = c != c
assert is_inf or is_nan or c > 1e14, f"unexpected cond {c}"
"#,
    );
}

#[test]
fn argsort_descending_trick() {
    run(
        r#"
import numpy as np

a = np.asarray([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0])

# argsort gives ascending order indices.
idx = np.argsort(a).tolist()
ascending = [a[i] for i in idx]
for prev, cur in zip(ascending[:-1], ascending[1:]):
    assert prev <= cur

# np.argsort(-a) is the standard descending-sort idiom.
desc_idx = np.argsort(-a).tolist()
descending = [a[i] for i in desc_idx]
for prev, cur in zip(descending[:-1], descending[1:]):
    assert prev >= cur
"#,
    );
}

#[test]
fn outer_product_complex_and_real() {
    run(
        r#"
import numpy as np

# Real outer product: shape (m, n) for inputs of length m, n.
a = np.asarray([1.0, 2.0])
b = np.asarray([3.0, 4.0, 5.0])
out = np.outer(a, b)
assert out.shape == (2, 3)
assert out.tolist() == [[3.0, 4.0, 5.0], [6.0, 8.0, 10.0]]

# Complex outer product: numpy does NOT conjugate `a` (only vdot does).
c1 = np.asarray([1 + 1j, 2 - 1j])
c2 = np.asarray([1j, 1.0])
prod = np.outer(c1, c2).tolist()
expected = [
    [(1 + 1j) * 1j, (1 + 1j) * 1.0],
    [(2 - 1j) * 1j, (2 - 1j) * 1.0],
]
for row_got, row_exp in zip(prod, expected):
    for got, exp in zip(row_got, row_exp):
        assert abs(complex(got) - complex(exp)) < 1e-12
"#,
    );
}

#[test]
fn polyfit_recovers_quadratic_coefficients() {
    run(
        r#"
import numpy as np

# y = 2*x^2 - 3*x + 5, sampled exactly.
xs = np.asarray([-2.0, -1.0, 0.0, 1.0, 2.0, 3.0])
ys = 2 * xs * xs - 3 * xs + 5
coef = np.polyfit(xs, ys, 2).tolist()
# polyfit returns in descending power: [a, b, c] for a*x^2 + b*x + c.
assert abs(coef[0] - 2.0) < 1e-9
assert abs(coef[1] - (-3.0)) < 1e-9
assert abs(coef[2] - 5.0) < 1e-9
"#,
    );
}
