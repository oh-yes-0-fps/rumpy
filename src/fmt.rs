//! repr() and str() formatting for ArraysD.

use crate::dtype::{ArraysD, DType};
use ndarray::{Axis, IxDyn};

pub fn repr(a: &ArraysD) -> String {
    let body = format_body(a);
    let dt = a.dtype();
    match dt {
        DType::F64 => format!("array({body})"),
        _ => format!("array({body}, dtype={})", dt.name_owned()),
    }
}

fn format_body(a: &ArraysD) -> String {
    match a {
        ArraysD::Bool(arr) => format_nested(arr, &|v: &bool| format!("{v}")),
        ArraysD::I8(arr) => format_nested(arr, &|v: &i8| format!("{v}")),
        ArraysD::I16(arr) => format_nested(arr, &|v: &i16| format!("{v}")),
        ArraysD::I32(arr) => format_nested(arr, &|v: &i32| format!("{v}")),
        ArraysD::I64(arr) => format_nested(arr, &|v: &i64| format!("{v}")),
        ArraysD::U8(arr) => format_nested(arr, &|v: &u8| format!("{v}")),
        ArraysD::U16(arr) => format_nested(arr, &|v: &u16| format!("{v}")),
        ArraysD::U32(arr) => format_nested(arr, &|v: &u32| format!("{v}")),
        ArraysD::U64(arr) => format_nested(arr, &|v: &u64| format!("{v}")),
        ArraysD::F16(arr) => format_nested(arr, &|v: &half::f16| format!("{v}")),
        ArraysD::F32(arr) => format_nested(arr, &|v: &f32| format!("{v}")),
        ArraysD::F64(arr) => format_nested(arr, &|v: &f64| format!("{v}")),
        ArraysD::C64(arr) => {
            format_nested(arr, &|v: &num_complex::Complex<f32>| format!("({}+{}j)", v.re, v.im))
        }
        ArraysD::C128(arr) => {
            format_nested(arr, &|v: &num_complex::Complex<f64>| format!("({}+{}j)", v.re, v.im))
        }
        // Non-numeric variants render through their natural Debug/string
        // form. We use a generic-but-non-Copy helper since these element
        // types (PyObjectRef, String, Vec<u8>, i64) aren't all Copy.
        ArraysD::Object(arr) => format_nested_ref(arr, &|v| format!("{v:?}")),
        ArraysD::Str { data, .. } => {
            format_nested_ref(data, &|s: &String| format!("'{}'", s.replace('\'', "\\'")))
        }
        ArraysD::Bytes { data, .. } | ArraysD::Void { data, .. } => {
            format_nested_ref(data, &|b: &Vec<u8>| format!("{b:?}"))
        }
        ArraysD::Datetime64 { data, .. } | ArraysD::Timedelta64 { data, .. } => {
            format_nested(data, &|v: &i64| format!("{v}"))
        }
    }
}

/// `format_nested` variant for non-Copy element types — passes `&T` to the
/// formatter rather than reading the value out of the cell by index.
fn format_nested_ref<T, F: Fn(&T) -> String>(
    arr: &ndarray::ArrayBase<ndarray::OwnedRepr<T>, IxDyn>,
    f: &F,
) -> String {
    fn rec<T, F: Fn(&T) -> String>(
        a: &ndarray::ArrayBase<ndarray::ViewRepr<&T>, IxDyn>,
        f: &F,
    ) -> String {
        if a.ndim() == 0 {
            return f(&a[IxDyn(&[])]);
        }
        if a.ndim() == 1 {
            let parts: Vec<String> = a.iter().map(f).collect();
            return format!("[{}]", parts.join(", "));
        }
        let parts: Vec<String> = a.axis_iter(Axis(0)).map(|s| rec(&s, f)).collect();
        format!("[{}]", parts.join(", "))
    }
    rec(&arr.view(), f)
}

fn format_nested<T: Copy, F: Fn(&T) -> String>(
    arr: &ndarray::ArrayBase<ndarray::OwnedRepr<T>, IxDyn>,
    f: &F,
) -> String {
    fn rec<T: Copy, F: Fn(&T) -> String>(
        a: &ndarray::ArrayBase<ndarray::ViewRepr<&T>, IxDyn>,
        f: &F,
    ) -> String {
        if a.ndim() == 0 {
            return f(&a[IxDyn(&[])]);
        }
        if a.ndim() == 1 {
            let parts: Vec<String> = a.iter().map(f).collect();
            return format!("[{}]", parts.join(", "));
        }
        let parts: Vec<String> = a.axis_iter(Axis(0)).map(|s| rec(&s, f)).collect();
        format!("[{}]", parts.join(", "))
    }
    rec(&arr.view(), f)
}
