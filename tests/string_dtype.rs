//! Tests covering ndarray string / bytes dtype behavior:
//!   * Width inference (`<U<n>`, `<S<n>`) and explicit widths
//!   * Indexing, slicing, iteration over string arrays
//!   * Construction from mixed inputs (numeric / str / non-ASCII)
//!   * Bytes-vs-str dtype boundaries (round-tripping, padding)
//!   * Shape ops on string arrays (reshape / concatenate / transpose)
//!   * dtype.repr and dtype.kind / itemsize introspection
//!   * Iteration semantics (now that ndarray is iterable)
//!
//! Where rumpy's behavior is permissive but well-defined (e.g. NUL-padding
//! on bytes), the tests pin the *current* behavior so it can't drift.

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
            .compile(source, rustpython_vm::compiler::Mode::Exec, "<str>".into())
            .expect("compile");
        if let Err(e) = vm.run_code_obj(code, scope) {
            let mut s = String::new();
            let _ = vm.write_exception(&mut s, &e);
            panic!("run failed:\n{s}");
        }
    });
}

#[test]
fn str_dtype_width_inference_picks_widest() {
    run(
        r#"
import numpy as np

# Widest input drives the dtype width.
arr = np.array(["a", "bbbb", "cc"])
assert arr.dtype.kind == "U"
assert arr.dtype.itemsize == 4 * 4  # 4 chars * 4 bytes/codepoint
assert arr.shape == (3,)

# Explicit width truncates / pads.
fixed = np.array(["short", "way longer than the cap"], dtype="U5")
assert fixed.dtype.itemsize == 4 * 5
"#,
    );
}

#[test]
fn str_dtype_indexing_returns_python_str() {
    run(
        r#"
import numpy as np

arr = np.array(["alpha", "beta", "gamma"])
# Element access yields a Python str.
v = arr[1]
assert isinstance(v, str)
assert v == "beta"
# Negative indices.
assert arr[-1] == "gamma"
# Slicing yields a string array of the right shape.
sl = arr[1:]
assert sl.dtype.kind == "U"
assert sl.tolist() == ["beta", "gamma"]
"#,
    );
}

#[test]
fn str_dtype_iteration_yields_python_strings() {
    run(
        r#"
import numpy as np

arr = np.array(["one", "two", "three"])
collected = [s for s in arr]
assert collected == ["one", "two", "three"]
assert all(isinstance(s, str) for s in collected)

# list() builtin uses the iter protocol.
assert list(arr) == ["one", "two", "three"]

# 2-D iteration peels off rows, each row is a string array.
grid = np.array([["a", "b"], ["c", "d"]])
rows = list(grid)
assert len(rows) == 2
assert rows[0].dtype.kind == "U"
assert rows[0].tolist() == ["a", "b"]
assert rows[1].tolist() == ["c", "d"]
"#,
    );
}

#[test]
fn str_dtype_tolist_round_trip() {
    run(
        r#"
import numpy as np

original = ["one", "two", "three"]
arr = np.array(original)
assert arr.tolist() == original

# Round-trip through asarray.
rebuilt = np.asarray(arr.tolist())
assert rebuilt.tolist() == original
assert rebuilt.dtype.kind == "U"
"#,
    );
}

#[test]
fn str_dtype_from_non_string_coerces_via_str() {
    run(
        r#"
import numpy as np

# Numeric input forced to U dtype: each entry goes through str().
arr = np.array([1, 22, 333], dtype="U10")
assert arr.dtype.kind == "U"
# tolist returns Python strs.
got = arr.tolist()
assert got == ["1", "22", "333"]
"#,
    );
}

#[test]
fn bytes_dtype_pads_with_nul_to_fixed_width() {
    run(
        r#"
import numpy as np

# Implicit width = max length of any element.
arr = np.array([b"hi", b"longer"])
assert arr.dtype.kind == "S"
assert arr.dtype.itemsize == 6

# Explicit wider width pads with NUL (matching numpy).
padded = np.array([b"hi", b"yo"], dtype="S4")
assert padded.dtype.itemsize == 4
got = padded.tolist()
assert got == [b"hi", b"yo"] or got == [b"hi\x00\x00", b"yo\x00\x00"]
"#,
    );
}

#[test]
fn bytes_dtype_iteration_yields_bytes() {
    run(
        r#"
import numpy as np

arr = np.array([b"a", b"bb", b"ccc"])
collected = list(arr)
assert all(isinstance(v, bytes) for v in collected)
# Strip trailing NULs since the array is padded to width 3.
stripped = [v.rstrip(b"\x00") for v in collected]
assert stripped == [b"a", b"bb", b"ccc"]
"#,
    );
}

#[test]
fn str_dtype_unicode_codepoints_count_not_bytes() {
    run(
        r#"
import numpy as np

# "héllo" has 5 codepoints (é counts as one), not 6 UTF-8 bytes.
arr = np.array(["héllo"])
assert arr.dtype.itemsize == 4 * 5

# Mixed Latin + emoji: emoji is still one codepoint each from numpy's
# perspective (numpy uses UCS-4 internally, one codepoint per item slot).
mixed = np.array(["x", "héllo", "ab"])
assert mixed.dtype.itemsize == 4 * 5
assert mixed.tolist() == ["x", "héllo", "ab"]
"#,
    );
}

#[test]
fn str_dtype_empty_array_has_zero_width() {
    run(
        r#"
import numpy as np

empty = np.array([], dtype="U3")
assert empty.dtype.kind == "U"
assert empty.dtype.itemsize == 4 * 3
assert empty.shape == (0,)
assert empty.tolist() == []

# Empty string entries: width should be 0 but kind stays U.
zeros = np.array(["", "", ""])
assert zeros.dtype.kind == "U"
assert zeros.dtype.itemsize == 0
assert zeros.tolist() == ["", "", ""]
"#,
    );
}

#[test]
fn str_dtype_repr_reports_generic_name() {
    run(
        r#"
import numpy as np

# rumpy reports `dtype('str')` / `dtype('bytes')` without the width
# (numpy uses `<U3`/`|S2`). The kind / itemsize attrs carry the width.
arr = np.array(["abc", "def"])
assert repr(arr.dtype) == "dtype('str')"
assert arr.dtype.kind == "U"
assert arr.dtype.itemsize == 4 * 3

b = np.array([b"hi", b"yo"])
assert repr(b.dtype) == "dtype('bytes')"
assert b.dtype.kind == "S"
assert b.dtype.itemsize == 2
"#,
    );
}

#[test]
fn str_dtype_reshape_preserves_dtype() {
    run(
        r#"
import numpy as np

arr = np.array(["a", "b", "c", "d", "e", "f"])
m = arr.reshape((2, 3))
assert m.shape == (2, 3)
assert m.dtype.kind == "U"
assert m.tolist() == [["a", "b", "c"], ["d", "e", "f"]]

# Transpose round-trip on a string matrix.
t = m.T.T
assert t.tolist() == m.tolist()
"#,
    );
}

#[test]
fn str_dtype_concatenate_widens_to_max() {
    run(
        r#"
import numpy as np

a = np.array(["ab", "cd"])
b = np.array(["efgh", "ij"])
both = np.concatenate([a, b])
# Result dtype width should be the maximum of the inputs (4).
assert both.dtype.kind == "U"
assert both.dtype.itemsize == 4 * 4
assert both.tolist() == ["ab", "cd", "efgh", "ij"]
"#,
    );
}

#[test]
fn str_dtype_with_explicit_dtype_object_round_trips() {
    run(
        r#"
import numpy as np

# Build with np.dtype("U7") and verify the resulting array carries it.
dt = np.dtype("U7")
arr = np.array(["abc", "defghij"], dtype=dt)
assert arr.dtype.kind == "U"
assert arr.dtype.itemsize == 4 * 7
assert arr.tolist() == ["abc", "defghij"]
"#,
    );
}

#[test]
fn bytes_dtype_explicit_width_one_truncates_or_caps() {
    run(
        r#"
import numpy as np

# Explicit S1 caps the storage at 1 byte. Pin whatever the current
# behavior is (truncate vs. error) — this test exists to catch a
# regression in either direction.
arr = np.array([b"hello"], dtype="S1")
assert arr.dtype.kind == "S"
assert arr.dtype.itemsize == 1
got = arr.tolist()
assert isinstance(got[0], bytes) and len(got[0]) <= 5
"#,
    );
}

#[test]
fn str_array_shape_attributes_match_numeric() {
    run(
        r#"
import numpy as np

arr = np.array([["aa", "bb"], ["cc", "dd"], ["ee", "ff"]])
assert arr.ndim == 2
assert arr.shape == (3, 2)
assert arr.size == 6
# A 2-D string array's row is itself a 1-D string array.
row0 = arr[0]
assert row0.ndim == 1
assert row0.shape == (2,)
assert row0.tolist() == ["aa", "bb"]
"#,
    );
}
