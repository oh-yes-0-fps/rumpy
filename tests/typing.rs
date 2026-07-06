//! Tests for the embedded pure-Python submodules:
//! numpy.typing, numpy.exceptions, numpy.version,
//! numpy.compat, numpy.doc, numpy.core, numpy.ctypeslib,
//! numpy.char, numpy.rec, numpy.dtypes.

use rustpython_vm::Interpreter;

fn interp() -> Interpreter {
    let b = Interpreter::builder(Default::default());
    let def = rumpy::module_def(&b.ctx);
    b.add_native_module(def).build()
}

/// Register every pure-Python child of `numpy` in `sys.modules` under its
/// dotted name so `from numpy.<sub> import …` works through the import
/// machinery. Submodules that fail to initialize (e.g. because they need
/// a stdlib module we don't ship) are silently skipped — the tests that
/// actually exercise them will surface the failure directly.
fn run(source: &str) {
    let interp = interp();
    interp.enter(|vm| {
        let numpy_mod = vm.import("numpy", 0).expect("import numpy");
        let sys_modules = vm.sys_module.get_attr("modules", vm).expect("sys.modules");
        for sub in [
            "typing",
            "exceptions",
            "version",
            "compat",
            "doc",
            "core",
            "ctypeslib",
            "char",
            "rec",
            "dtypes",
            "testing",
            "emath",
            "polynomial",
            "strings",
        ] {
            if let Ok(m) = numpy_mod.get_attr(sub, vm) {
                let dotted = format!("numpy.{sub}");
                let _ = sys_modules.set_item(dotted.as_str(), m, vm);
            }
        }

        let scope = vm.new_scope_with_builtins();
        let code = vm
            .compile(source, rustpython_vm::compiler::Mode::Exec, "<t>".into())
            .expect("compile");
        if let Err(e) = vm.run_code_obj(code, scope) {
            let mut s = String::new();
            let _ = vm.write_exception(&mut s, &e);
            panic!("run failed:\n{s}");
        }
    });
}

#[test]
fn typing_module_attribute_access() {
    run(r#"
import numpy
t = numpy.typing
assert t.NDArray is not None
assert t.ArrayLike is not None
assert t.DTypeLike is not None
"#);
}

#[test]
fn typing_from_import() {
    run(r#"
from numpy.typing import NDArray, ArrayLike, DTypeLike
assert NDArray is not None
assert ArrayLike is not None
assert DTypeLike is not None
"#);
}

#[test]
fn typing_ndarray_subscript_returns_ndarray() {
    run(r#"
import numpy
NDArray = numpy.typing.NDArray
# NDArray[T] returns numpy.ndarray (the real class)
assert NDArray[float] is numpy.ndarray
assert NDArray[int]   is numpy.ndarray
"#);
}

#[test]
fn typing_all_export() {
    run(r#"
import numpy
exported = set(numpy.typing.__all__)
assert exported == {"NDArray", "ArrayLike", "DTypeLike", "NBitBase"}, exported
"#);
}

#[test]
fn exceptions_axis_error_subclasses_value_and_index_error() {
    run(r#"
from numpy.exceptions import AxisError
e = AxisError(2, ndim=1)
assert isinstance(e, ValueError)
assert isinstance(e, IndexError)
assert e.axis == 2
assert e.ndim == 1
assert "out of bounds" in str(e)
"#);
}

#[test]
fn exceptions_full_set_present() {
    run(r#"
from numpy.exceptions import (
    AxisError,
    ComplexWarning,
    RankWarning,
    TooHardError,
    VisibleDeprecationWarning,
    DTypePromotionError,
)
assert issubclass(ComplexWarning, RuntimeWarning)
assert issubclass(RankWarning, UserWarning)
assert issubclass(TooHardError, RuntimeError)
assert issubclass(VisibleDeprecationWarning, UserWarning)
assert issubclass(DTypePromotionError, TypeError)
"#);
}

#[test]
fn version_has_strings_and_release_flag() {
    run(r#"
import numpy
v = numpy.version
assert isinstance(v.version, str)
assert isinstance(v.full_version, str)
assert isinstance(v.short_version, str)
assert isinstance(v.git_revision, str)
assert v.release in (True, False)
# host injects the crate version, so these three agree.
assert v.version == v.full_version == v.short_version
"#);
}

#[test]
fn compat_exposes_legacy_aliases() {
    run(r#"
from numpy.compat import unicode, long, basestring, asbytes, asstr, asunicode
assert unicode is str
assert long is int
assert isinstance(basestring, tuple)
assert asbytes("x") == b"x"
assert asstr(b"x") == "x"
assert asunicode("x") == "x"
"#);
}

#[test]
fn doc_module_is_importable() {
    run(r#"
import numpy.doc
assert hasattr(numpy.doc, "__all__")
assert numpy.doc.__all__ == []
"#);
}

#[test]
fn core_module_is_importable() {
    run(r#"
import numpy.core
assert hasattr(numpy.core, "__all__")
"#);
}

#[test]
fn ctypeslib_stubs_raise_not_implemented() {
    run(r#"
from numpy.ctypeslib import as_array, ndpointer, load_library
for fn in (as_array, ndpointer, load_library):
    try:
        fn()
    except NotImplementedError:
        pass
    else:
        raise AssertionError("expected NotImplementedError")
"#);
}

#[test]
fn char_basic_string_ops() {
    run(r#"
import numpy.char as c
assert c.upper(["foo", "bar"]) == ["FOO", "BAR"]
assert c.lower(["FOO", "BAR"]) == ["foo", "bar"]
assert c.add(["a", "b"], ["1", "2"]) == ["a1", "b2"]
assert c.multiply(["ab", "cd"], 2) == ["abab", "cdcd"]
assert c.strip(["  foo  ", "\tbar\n"]) == ["foo", "bar"]
assert c.replace(["hello"], "l", "L") == ["heLLo"]
assert c.startswith(["abc", "xyz"], "ab") == [True, False]
assert c.count(["abcabc"], "b") == [2]
assert c.str_len(["", "abc"]) == [0, 3]
"#);
}

#[test]
fn char_comparison_ops() {
    run(r#"
import numpy.char as c
assert c.equal(["a", "b"], ["a", "c"]) == [True, False]
assert c.not_equal(["a"], ["b"]) == [True]
assert c.less(["a"], ["b"]) == [True]
assert c.greater_equal(["b"], ["b"]) == [True]
"#);
}

#[test]
fn rec_fromstring_parses_typed_buffer() {
    run(r#"
import numpy.rec as r

def le_int(v, n):
    return (v & ((1 << (8 * n)) - 1)).to_bytes(n, "little")

packed = (
    le_int(1, 4) + bytes.fromhex("000000000000F83F") + b"abc"
    + le_int(2, 4) + bytes.fromhex("0000000000000240") + b"de\x00"
)
arr = r.fromstring(packed, formats="i4,f8,S3", names="id,weight,tag")
assert len(arr) == 2
assert arr[0].id == 1
assert abs(arr[0].weight - 1.5) < 1e-12
assert arr[0].tag == b"abc"
assert arr[1].id == 2
assert abs(arr[1].weight - 2.25) < 1e-12
assert arr[1].tag == b"de"

be_packed = (1).to_bytes(4, "big") + bytes.fromhex("3FF8000000000000") + b"xyz"
arr2 = r.fromstring(be_packed, formats="i4,f8,S3",
                    names="id,weight,tag", byteorder=">")
assert arr2[0].id == 1
assert abs(arr2[0].weight - 1.5) < 1e-12
"#);
}

#[test]
fn dtypes_classes_carry_names() {
    run(r#"
from numpy.dtypes import (
    BoolDType, Int8DType, Int32DType, Int64DType,
    Float32DType, Float64DType, Complex128DType,
)
assert BoolDType().name == "bool"
assert Int8DType().name == "int8"
assert Int32DType().name == "int32"
assert Int64DType().name == "int64"
assert Float32DType().name == "float32"
assert Float64DType().name == "float64"
assert Complex128DType().name == "complex128"

# repr/str/eq round-trip
assert repr(Float64DType()) == "dtype('float64')"
assert str(Int32DType()) == "int32"
assert Int32DType() == Int32DType()
assert Int32DType() == "int32"
assert hash(Int32DType()) == hash(Int32DType())
"#);
}

#[test]
fn ma_count_along_axis() {
    run(r#"
import numpy
ma = numpy.ma

m = ma.masked_array(
    [[1, 2, 3], [4, 5, 6]],
    mask=[[False, True, False], [True, False, False]],
)
assert m.count() == 4
got0 = m.count(axis=0)
assert got0.tolist() == [1, 1, 2]
got1 = m.count(axis=1)
assert got1.tolist() == [2, 2]
"#);
}

#[test]
fn testing_assert_equal_and_array_equal() {
    run(r#"
import numpy.testing as t
t.assert_equal(1, 1)
t.assert_equal([1, 2, 3], [1, 2, 3])
t.assert_array_equal([[1, 2], [3, 4]], [[1, 2], [3, 4]])
try:
    t.assert_equal([1, 2], [1, 3])
except AssertionError:
    pass
else:
    raise AssertionError("expected mismatch")
"#);
}

#[test]
fn testing_assert_allclose_handles_tol_and_nan() {
    run(r#"
import numpy.testing as t
t.assert_allclose([1.0, 2.0], [1.0 + 1e-9, 2.0])
t.assert_allclose([float("nan")], [float("nan")])
try:
    t.assert_allclose([1.0], [1.5])
except AssertionError:
    pass
else:
    raise AssertionError("expected close failure")
"#);
}

#[test]
fn testing_assert_raises_and_less() {
    run(r#"
import numpy.testing as t
t.assert_raises(ValueError, int, "abc")
t.assert_array_less([1, 2], [2, 3])
try:
    t.assert_array_less([1, 3], [2, 3])
except AssertionError:
    pass
else:
    raise AssertionError("expected less failure")
"#);
}

#[test]
fn emath_sqrt_promotes_negative_reals() {
    run(r#"
import numpy.emath as e
# Real-domain sqrt for non-negatives.
assert e.sqrt(4) == 2.0
# Negative reals get a complex answer.
r = e.sqrt(-1)
assert isinstance(r, complex)
assert abs(r - 1j) < 1e-12
# Array variant maps element-wise.
out = e.sqrt([4, -4])
assert out[0] == 2.0
assert abs(out[1] - 2j) < 1e-12
"#);
}

#[test]
fn emath_log_family() {
    run(r#"
import numpy.emath as e
# log of negative is complex.
r = e.log(-1)
assert isinstance(r, complex)
# Real-domain log returns float.
assert abs(e.log(1) - 0.0) < 1e-12
assert abs(e.log10(100) - 2.0) < 1e-12
assert abs(e.log2(8) - 3.0) < 1e-12
assert abs(e.logn(3, 27) - 3.0) < 1e-12
"#);
}

#[test]
fn emath_inverse_trig_extends_domain() {
    run(r#"
import numpy.emath as e
# Inside the real domain — real answer.
inside = e.arccos(0.5)
assert isinstance(inside, float)
inside = e.arcsin(0.5)
assert isinstance(inside, float)
# Outside — complex answer.
r = e.arccos(2)
assert isinstance(r, complex)
r = e.arcsin(2)
assert isinstance(r, complex)
r = e.arctanh(2)
assert isinstance(r, complex)
"#);
}

#[test]
fn polynomial_eval_and_arith() {
    run(r#"
from numpy.polynomial import Polynomial
# p(x) = 1 + 2x + 3x^2
p = Polynomial([1, 2, 3])
assert p(0) == 1
assert p(1) == 6
assert p(2) == 17
# Vector input.
assert p([0, 1, 2]) == [1, 6, 17]
# Arithmetic.
q = Polynomial([0, 1])  # x
assert (p + q).coef == [1, 3, 3]
assert (p - q).coef == [1, 1, 3]
assert (q * q).coef == [0, 0, 1]
assert (-p).coef == [-1, -2, -3]
assert (p + 5).coef == [6, 2, 3]
"#);
}

#[test]
fn polynomial_deriv_integ_and_degree() {
    run(r#"
from numpy.polynomial import Polynomial
p = Polynomial([1, 2, 3])  # 1 + 2x + 3x^2
assert p.degree == 2
d = p.deriv()
assert d.coef == [2, 6]
# d/dx of the integral returns the original (up to integration constant).
i = p.integ(k=0)
# integ adds a leading constant 0.
assert i.coef[0] == 0
"#);
}

#[test]
fn polynomial_roots_recover_factors() {
    run(r#"
from numpy.polynomial import Polynomial, polyroots
# (x - 1)(x - 2) = 2 - 3x + x^2
roots = polyroots([2, -3, 1])
roots = sorted([r.real for r in roots])
assert abs(roots[0] - 1.0) < 1e-6
assert abs(roots[1] - 2.0) < 1e-6
"#);
}

#[test]
fn polynomial_fit_recovers_known_polynomial() {
    run(r#"
from numpy.polynomial import Polynomial
xs = [0, 1, 2, 3, 4]
# 1 + 2x + 3x^2 evaluated at xs.
ys = [1 + 2*x + 3*x*x for x in xs]
p = Polynomial.fit(xs, ys, 2)
assert abs(p.coef[0] - 1.0) < 1e-6
assert abs(p.coef[1] - 2.0) < 1e-6
assert abs(p.coef[2] - 3.0) < 1e-6
"#);
}

#[test]
fn strings_basic_ops() {
    run(r#"
import numpy.strings as s
assert s.upper(["foo", "bar"]) == ["FOO", "BAR"]
assert s.add(["a"], ["b"]) == ["ab"]
assert s.startswith(["abc", "xyz"], "ab") == [True, False]
assert s.equal(["a"], ["a"]) == [True]
assert s.str_len(["", "abc"]) == [0, 3]
"#);
}

#[test]
fn rec_construct_and_field_access() {
    run(r#"
import numpy.rec as r
arr = r.fromarrays([[1, 2, 3], [10.0, 20.0, 30.0]], names="id, value")
assert len(arr) == 3
assert arr.id == [1, 2, 3]
assert arr.value == [10.0, 20.0, 30.0]
assert arr.names == ("id", "value")
# Row access.
row = arr[0]
assert row.id == 1
assert row.value == 10.0
assert row["id"] == 1
assert row[0] == 1
"#);
}

#[test]
fn rec_fromrecords_and_mutation() {
    run(r#"
import numpy.rec as r
arr = r.fromrecords([(1, "a"), (2, "b")], names="i, s")
assert arr[1].s == "b"
arr.i = [10, 20]
assert arr.i == [10, 20]
arr[0].s = "z"
assert arr[0].s == "z"
"#);
}

#[test]
fn rec_helpers_and_format_parser() {
    run(r#"
import numpy.rec as r
assert r.find_duplicate(["a", "b", "a", "c", "b"]) == ["a", "b"]
fp = r.format_parser(["f8", "i4"], "x, y")
assert fp.names == ["x", "y"]
"#);
}

#[test]
fn dtypes_all_listed() {
    run(r#"
import numpy.dtypes as d
expected_prefix = [
    "Bool",
    "Int8", "Int16", "Int32", "Int64",
    "UInt8", "UInt16", "UInt32", "UInt64",
    "Float16", "Float32", "Float64",
    "Complex64", "Complex128",
    "Str", "Bytes", "Object",
]
for p in expected_prefix:
    cls_name = f"{p}DType"
    assert cls_name in d.__all__, cls_name
    assert hasattr(d, cls_name), cls_name
"#);
}
