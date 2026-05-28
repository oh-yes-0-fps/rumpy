//! Indexing & slicing.

use crate::dtype::{ArraysD, DType};
use crate::internal::{OptionExt, ResultExt, internal};
use ndarray::{ArrayD, Axis, IxDyn, SliceInfo, SliceInfoElem};
use rustpython_vm::{
    AsObject, PyObjectRef, PyResult, VirtualMachine,
    builtins::{PyInt, PySlice, PyTuple},
};

#[derive(Debug)]
pub enum IdxItem {
    Int(isize),
    Slice(Option<isize>, Option<isize>, Option<isize>),
    /// Fancy (integer-array) index along this axis.
    IntArray(Vec<isize>),
    /// Boolean-mask index along this axis (1-D).
    BoolMask(Vec<bool>),
    /// `...` — expands to enough full-slice items to fill remaining axes.
    Ellipsis,
    /// `None` / `np.newaxis` — inserts a length-1 axis at this position.
    NewAxis,
}

pub fn parse_index(
    obj: &PyObjectRef,
    ndim: usize,
    vm: &VirtualMachine,
) -> PyResult<Vec<IdxItem>> {
    let items: Vec<PyObjectRef> = if let Some(t) = obj.downcast_ref::<PyTuple>() {
        t.as_slice().to_vec()
    } else {
        vec![obj.clone()]
    };
    // Count how many items consume an axis (everything except NewAxis/Ellipsis).
    let mut axis_consuming = 0usize;
    let mut ellipsis_count = 0usize;
    for it in &items {
        if it.is(&vm.ctx.none) {
            // NewAxis — does not consume an axis
        } else if it.is(&vm.ctx.ellipsis) {
            ellipsis_count += 1;
        } else {
            axis_consuming += 1;
        }
    }
    if ellipsis_count > 1 {
        return Err(vm.new_index_error(
            "an index can only have a single ellipsis ('...')".to_string(),
        ));
    }
    if axis_consuming > ndim {
        return Err(vm.new_index_error("too many indices for array".to_string()));
    }
    let mut out = Vec::with_capacity(items.len());
    for it in items {
        if it.is(&vm.ctx.none) {
            out.push(IdxItem::NewAxis);
        } else if it.is(&vm.ctx.ellipsis) {
            out.push(IdxItem::Ellipsis);
        } else if let Some(i) = it.downcast_ref::<PyInt>() {
            // bool is a subclass of int; if the value is True/False *and*
            // not an ndarray, treat it as int 1/0 (numpy does the same).
            out.push(IdxItem::Int(i.try_to_primitive::<isize>(vm)?));
        } else if let Some(s) = it.downcast_ref::<PySlice>() {
            let start = slice_part_opt(&s.start, vm)?;
            let stop = slice_part_required(&s.stop, vm)?;
            let step = slice_part_opt(&s.step, vm)?;
            out.push(IdxItem::Slice(start, stop, step));
        } else if let Some(arr) = it.downcast_ref::<crate::PyNdArray>() {
            // ndarray index: bool mask or integer array.
            use crate::dtype::CoerceArray;
            if arr.view().dtype() == DType::Bool {
                let mask: Vec<bool> = arr.view().coerce::<bool>().iter().copied().collect();
                out.push(IdxItem::BoolMask(mask));
            } else if arr.view().dtype().is_integer() {
                let idxs: Vec<isize> = arr
                    .view()
                    .coerce::<i64>()
                    .iter()
                    .map(|&v| v as isize)
                    .collect();
                out.push(IdxItem::IntArray(idxs));
            } else {
                return Err(vm.new_index_error(
                    "only integer or boolean ndarrays may be used as indices".to_string(),
                ));
            }
        } else if let Some(l) = it.downcast_ref::<rustpython_vm::builtins::PyList>() {
            // List of ints → fancy index.
            let vec = l.borrow_vec();
            let mut idxs = Vec::with_capacity(vec.len());
            let mut all_bool = !vec.is_empty();
            for v in vec.iter() {
                if v.is(&vm.ctx.true_value) || v.is(&vm.ctx.false_value) {
                    // ok
                } else {
                    all_bool = false;
                }
            }
            if all_bool {
                let mask: Vec<bool> = vec
                    .iter()
                    .map(|v| v.is(&vm.ctx.true_value))
                    .collect();
                out.push(IdxItem::BoolMask(mask));
            } else {
                for v in vec.iter() {
                    let i = v.try_int(vm)?.try_to_primitive::<isize>(vm)?;
                    idxs.push(i);
                }
                out.push(IdxItem::IntArray(idxs));
            }
        } else {
            return Err(vm.new_type_error("unsupported index type".to_string()));
        }
    }
    Ok(out)
}

fn slice_part_opt(opt: &Option<PyObjectRef>, vm: &VirtualMachine) -> PyResult<Option<isize>> {
    match opt {
        None => Ok(None),
        Some(o) if o.is(&vm.ctx.none) => Ok(None),
        Some(o) => Ok(Some(o.try_int(vm)?.try_to_primitive::<isize>(vm)?)),
    }
}

fn slice_part_required(o: &PyObjectRef, vm: &VirtualMachine) -> PyResult<Option<isize>> {
    if o.is(&vm.ctx.none) {
        return Ok(None);
    }
    Ok(Some(o.try_int(vm)?.try_to_primitive::<isize>(vm)?))
}

pub fn normalize_idx(i: isize, dim: usize, vm: &VirtualMachine) -> PyResult<usize> {
    let dim_i = dim as isize;
    let real = if i < 0 { i + dim_i } else { i };
    if real < 0 || real >= dim_i {
        return Err(vm.new_index_error(format!(
            "index {i} out of bounds for axis of size {dim}"
        )));
    }
    Ok(real as usize)
}

fn normalize_slice(
    s: Option<isize>,
    e: Option<isize>,
    st: Option<isize>,
    dim: usize,
) -> (isize, isize, isize) {
    let dim_i = dim as isize;
    let step = st.unwrap_or(1);
    if step >= 0 {
        (
            s.map(|v| if v < 0 { (v + dim_i).max(0) } else { v.min(dim_i) })
                .unwrap_or(0),
            e.map(|v| if v < 0 { (v + dim_i).max(0) } else { v.min(dim_i) })
                .unwrap_or(dim_i),
            step,
        )
    } else {
        (
            s.map(|v| {
                if v < 0 {
                    (v + dim_i).max(-1)
                } else {
                    v.min(dim_i - 1)
                }
            })
            .unwrap_or(dim_i - 1),
            e.map(|v| {
                if v < 0 {
                    (v + dim_i).max(-1)
                } else {
                    v.min(dim_i - 1)
                }
            })
            .unwrap_or(-1),
            step,
        )
    }
}

/// Expand `Ellipsis` items to the right number of full-slice items, and
/// return the resulting flat list (still containing `NewAxis` markers).
fn expand_ellipsis(parsed: &[IdxItem], ndim: usize) -> Vec<IdxItem> {
    let axis_consuming: usize = parsed
        .iter()
        .filter(|it| !matches!(it, IdxItem::NewAxis | IdxItem::Ellipsis))
        .count();
    let ellipsis_fill = ndim.saturating_sub(axis_consuming);
    let mut out = Vec::with_capacity(parsed.len() + ellipsis_fill);
    let mut ellipsis_done = false;
    for it in parsed {
        match it {
            IdxItem::Ellipsis if !ellipsis_done => {
                ellipsis_done = true;
                for _ in 0..ellipsis_fill {
                    out.push(IdxItem::Slice(None, None, None));
                }
            }
            IdxItem::Ellipsis => {
                // Defensive: parse_index already rejects >1 ellipsis; if a
                // second one ever reaches here, treat it as a full slice.
                out.push(IdxItem::Slice(None, None, None));
            }
            IdxItem::Int(v) => out.push(IdxItem::Int(*v)),
            IdxItem::Slice(s, e, st) => out.push(IdxItem::Slice(*s, *e, *st)),
            IdxItem::IntArray(v) => out.push(IdxItem::IntArray(v.clone())),
            IdxItem::BoolMask(m) => out.push(IdxItem::BoolMask(m.clone())),
            IdxItem::NewAxis => out.push(IdxItem::NewAxis),
        }
    }
    // If no ellipsis was present, append implicit trailing full slices so
    // every remaining axis is selected. numpy does this implicitly.
    if !ellipsis_done {
        for _ in 0..ellipsis_fill {
            out.push(IdxItem::Slice(None, None, None));
        }
    }
    out
}

/// Apply an index path to an array — returns either a 0-D array (scalar
/// access) or a sub-array (slice access).
pub fn apply_index(
    a: &ArraysD,
    parsed: &[IdxItem],
    vm: &VirtualMachine,
) -> PyResult<ArraysD> {
    // ---- Advanced indexing: a single ndarray index ----
    if parsed.len() == 1 {
        match &parsed[0] {
            IdxItem::BoolMask(mask) => return bool_mask_select(a, mask, vm),
            IdxItem::IntArray(indices) => return int_array_select(a, indices, vm),
            _ => {}
        }
    }
    let nd = a.ndim();
    let parsed = expand_ellipsis(parsed, nd);
    // Drop NewAxis markers for the per-axis walk; we re-interleave 1-dims
    // into the result shape after slicing.
    let has_newaxis = parsed.iter().any(|it| matches!(it, IdxItem::NewAxis));
    let axis_items: Vec<&IdxItem> = parsed
        .iter()
        .filter(|it| !matches!(it, IdxItem::NewAxis))
        .collect();
    let all_int = axis_items.len() == nd
        && axis_items.iter().all(|k| matches!(k, IdxItem::Int(_)));
    if all_int && !has_newaxis {
        let mut norm = Vec::with_capacity(axis_items.len());
        for (k, &dim) in axis_items.iter().zip(a.shape()) {
            if let IdxItem::Int(v) = k {
                norm.push(normalize_idx(*v, dim, vm)?);
            }
        }
        let idx = IxDyn(&norm);
        return Ok(scalar_at(a, idx));
    }
    // Partial / slicing — apply axis-by-axis.
    let mut arr = a.clone();
    let mut consumed = 0usize;
    for (i, item) in axis_items.iter().enumerate() {
        let cur_axis = i - consumed;
        match item {
            IdxItem::Int(v) => {
                let dim = arr.shape()[cur_axis];
                let n = normalize_idx(*v, dim, vm)?;
                arr = index_axis(&arr, cur_axis, n);
                consumed += 1;
            }
            IdxItem::Slice(s, e, st) => {
                let dim = arr.shape()[cur_axis];
                let (start, end, step) = normalize_slice(*s, *e, *st, dim);
                arr = slice_axis(&arr, cur_axis, start, end, step, vm)?;
            }
            IdxItem::IntArray(_) | IdxItem::BoolMask(_) => {
                return Err(vm.new_index_error(
                    "advanced indexing only supported as the sole index element"
                        .to_string(),
                ));
            }
            // `NewAxis`/`Ellipsis` items are filtered out before this loop —
            // `expand_ellipsis` rewrites the index list. Surfacing a clean
            // internal error here rather than panicking guards against any
            // future bug that lets one through.
            IdxItem::NewAxis | IdxItem::Ellipsis => {
                return Err(crate::internal::internal(
                    vm,
                    "apply_index: NewAxis/Ellipsis leaked past expand_ellipsis",
                ));
            }
        }
    }
    // Re-interleave length-1 axes for each NewAxis marker by reading the
    // sliced shape in order and inserting `1`s where the markers sit.
    if has_newaxis {
        let sliced_shape = arr.shape().to_vec();
        let mut next_sliced = 0usize;
        let mut final_shape: Vec<usize> = Vec::with_capacity(sliced_shape.len() + 4);
        for item in &parsed {
            match item {
                IdxItem::NewAxis => final_shape.push(1),
                IdxItem::Slice(..) => {
                    if next_sliced < sliced_shape.len() {
                        final_shape.push(sliced_shape[next_sliced]);
                        next_sliced += 1;
                    }
                }
                IdxItem::Int(_) => {} // collapsed in the axis walk
                IdxItem::IntArray(_) | IdxItem::BoolMask(_) | IdxItem::Ellipsis => {}
            }
        }
        arr = crate::linalg::reshape(&arr, &final_shape)
            .ok_or_else(|| internal(vm, "newaxis reshape failed"))?;
    }
    Ok(arr)
}

/// `a[mask]` — boolean indexing. Mask shape must match `a.shape` (numpy
/// flattens both and selects where True).
fn bool_mask_select(
    a: &ArraysD,
    mask: &[bool],
    vm: &VirtualMachine,
) -> PyResult<ArraysD> {
    if mask.len() != a.len() {
        return Err(vm.new_index_error(format!(
            "boolean mask length {} != array length {}",
            mask.len(),
            a.len()
        )));
    }
    let flat = crate::linalg::flatten(a);
    macro_rules! per {
        ($var:ident, $ty:ty, $arr:ident) => {{
            let v: Vec<$ty> = $arr
                .iter()
                .zip(mask.iter())
                .filter_map(|(&val, &m)| if m { Some(val) } else { None })
                .collect();
            ArrayD::from_shape_vec(IxDyn(&[v.len()]), v)
                .or_internal(vm, "bool_mask_select")
                .map(ArraysD::$var)
        }};
    }
    // Clone-based variant for non-Copy element types (String, Vec<u8>,
    // PyObjectRef). $build takes the resulting ArrayD and wraps it back
    // into the right ArraysD variant.
    macro_rules! per_clone {
        ($arr:ident, $build:expr) => {{
            let v: Vec<_> = $arr
                .iter()
                .zip(mask.iter())
                .filter_map(|(val, &m)| if m { Some(val.clone()) } else { None })
                .collect();
            ArrayD::from_shape_vec(IxDyn(&[v.len()]), v)
                .or_internal(vm, "bool_mask_select")
                .map($build)
        }};
    }
    match flat {
        ArraysD::Bool(arr) => per!(Bool, bool, arr),
        ArraysD::I8(arr) => per!(I8, i8, arr),
        ArraysD::I16(arr) => per!(I16, i16, arr),
        ArraysD::I32(arr) => per!(I32, i32, arr),
        ArraysD::I64(arr) => per!(I64, i64, arr),
        ArraysD::U8(arr) => per!(U8, u8, arr),
        ArraysD::U16(arr) => per!(U16, u16, arr),
        ArraysD::U32(arr) => per!(U32, u32, arr),
        ArraysD::U64(arr) => per!(U64, u64, arr),
        ArraysD::F16(arr) => per!(F16, half::f16, arr),
        ArraysD::F32(arr) => per!(F32, f32, arr),
        ArraysD::F64(arr) => per!(F64, f64, arr),
        ArraysD::C64(arr) => per!(C64, crate::dtype::C32, arr),
        ArraysD::C128(arr) => per!(C128, crate::dtype::C64, arr),
        ArraysD::Object(arr) => per_clone!(arr, ArraysD::Object),
        ArraysD::Str { itemsize_chars, data } => {
            let n = itemsize_chars;
            per_clone!(data, |d| ArraysD::Str { itemsize_chars: n, data: d })
        }
        ArraysD::Bytes { itemsize, data } => {
            let n = itemsize;
            per_clone!(data, |d| ArraysD::Bytes { itemsize: n, data: d })
        }
        ArraysD::Datetime64 { unit, data } => {
            per_clone!(data, |d| ArraysD::Datetime64 { unit, data: d })
        }
        ArraysD::Timedelta64 { unit, data } => {
            per_clone!(data, |d| ArraysD::Timedelta64 { unit, data: d })
        }
        ArraysD::Void { layout, data } => {
            per_clone!(data, |d| ArraysD::Void { layout: layout.clone(), data: d })
        }
    }
}

/// `a[[i, j, k, …]]` — fancy indexing along axis 0.
fn int_array_select(
    a: &ArraysD,
    indices: &[isize],
    vm: &VirtualMachine,
) -> PyResult<ArraysD> {
    // Empty index → return an empty array with shape (0, ...rest of a.shape).
    if indices.is_empty() {
        let mut shape: Vec<usize> = a.shape().to_vec();
        if !shape.is_empty() {
            shape[0] = 0;
        } else {
            shape.push(0);
        }
        return Ok(empty_like_shape(a, &shape));
    }
    let axis0 = a.shape()[0];
    let normalized: Vec<usize> = indices
        .iter()
        .map(|&i| normalize_idx(i, axis0, vm))
        .collect::<PyResult<_>>()?;

    // Stack the per-index sub-arrays along a new leading axis.
    let parts: Vec<ArraysD> = normalized
        .iter()
        .map(|&i| index_axis(a, 0, i))
        .collect();
    crate::extras::stack(&parts, 0, vm)
}

/// Produce an empty array with the given shape, same dtype as `a`.
fn empty_like_shape(a: &ArraysD, shape: &[usize]) -> ArraysD {
    macro_rules! per {
        ($var:ident, $ty:ty) => {{
            ArraysD::$var(ArrayD::<$ty>::default(IxDyn(shape)))
        }};
    }
    match a {
        ArraysD::Bool(_) => per!(Bool, bool),
        ArraysD::I8(_) => per!(I8, i8),
        ArraysD::I16(_) => per!(I16, i16),
        ArraysD::I32(_) => per!(I32, i32),
        ArraysD::I64(_) => per!(I64, i64),
        ArraysD::U8(_) => per!(U8, u8),
        ArraysD::U16(_) => per!(U16, u16),
        ArraysD::U32(_) => per!(U32, u32),
        ArraysD::U64(_) => per!(U64, u64),
        ArraysD::F16(_) => per!(F16, half::f16),
        ArraysD::F32(_) => per!(F32, f32),
        ArraysD::F64(_) => per!(F64, f64),
        ArraysD::C64(_) => per!(C64, crate::dtype::C32),
        ArraysD::C128(_) => per!(C128, crate::dtype::C64),
        _ => { a.clone() },
    }
}

fn scalar_at(a: &ArraysD, idx: IxDyn) -> ArraysD {
    use crate::dtype::ArraysD::*;
    macro_rules! one {
        ($variant:ident, $arr:ident) => {{
            $variant(ArrayD::from_elem(IxDyn(&[]), $arr[idx.clone()]))
        }};
    }
    match a {
        Bool(arr) => one!(Bool, arr),
        I8(arr) => one!(I8, arr),
        I16(arr) => one!(I16, arr),
        I32(arr) => one!(I32, arr),
        I64(arr) => one!(I64, arr),
        U8(arr) => one!(U8, arr),
        U16(arr) => one!(U16, arr),
        U32(arr) => one!(U32, arr),
        U64(arr) => one!(U64, arr),
        F16(arr) => one!(F16, arr),
        F32(arr) => one!(F32, arr),
        F64(arr) => one!(F64, arr),
        C64(arr) => one!(C64, arr),
        C128(arr) => one!(C128, arr),
        Object(arr) => Object(ArrayD::from_elem(IxDyn(&[]), arr[idx].clone())),
        Str { itemsize_chars, data } => Str {
            itemsize_chars: *itemsize_chars,
            data: ArrayD::from_elem(IxDyn(&[]), data[idx].clone()),
        },
        Bytes { itemsize, data } => Bytes {
            itemsize: *itemsize,
            data: ArrayD::from_elem(IxDyn(&[]), data[idx].clone()),
        },
        Datetime64 { unit, data } => Datetime64 {
            unit: *unit,
            data: ArrayD::from_elem(IxDyn(&[]), data[idx]),
        },
        Timedelta64 { unit, data } => Timedelta64 {
            unit: *unit,
            data: ArrayD::from_elem(IxDyn(&[]), data[idx]),
        },
        Void { layout, data } => Void {
            layout: layout.clone(),
            data: ArrayD::from_elem(IxDyn(&[]), data[idx].clone()),
        },
    }
}

fn index_axis(a: &ArraysD, axis: usize, n: usize) -> ArraysD {
    macro_rules! per {
        ($var:ident, $arr:ident) => {{
            ArraysD::$var($arr.clone().index_axis_move(Axis(axis), n))
        }};
    }
    match a {
        ArraysD::Bool(arr) => per!(Bool, arr),
        ArraysD::I8(arr) => per!(I8, arr),
        ArraysD::I16(arr) => per!(I16, arr),
        ArraysD::I32(arr) => per!(I32, arr),
        ArraysD::I64(arr) => per!(I64, arr),
        ArraysD::U8(arr) => per!(U8, arr),
        ArraysD::U16(arr) => per!(U16, arr),
        ArraysD::U32(arr) => per!(U32, arr),
        ArraysD::U64(arr) => per!(U64, arr),
        ArraysD::F16(arr) => per!(F16, arr),
        ArraysD::F32(arr) => per!(F32, arr),
        ArraysD::F64(arr) => per!(F64, arr),
        ArraysD::C64(arr) => per!(C64, arr),
        ArraysD::C128(arr) => per!(C128, arr),
        ArraysD::Object(arr) => ArraysD::Object(arr.clone().index_axis_move(Axis(axis), n)),
        ArraysD::Str { itemsize_chars, data } => ArraysD::Str {
            itemsize_chars: *itemsize_chars,
            data: data.clone().index_axis_move(Axis(axis), n),
        },
        ArraysD::Bytes { itemsize, data } => ArraysD::Bytes {
            itemsize: *itemsize,
            data: data.clone().index_axis_move(Axis(axis), n),
        },
        ArraysD::Datetime64 { unit, data } => ArraysD::Datetime64 {
            unit: *unit,
            data: data.clone().index_axis_move(Axis(axis), n),
        },
        ArraysD::Timedelta64 { unit, data } => ArraysD::Timedelta64 {
            unit: *unit,
            data: data.clone().index_axis_move(Axis(axis), n),
        },
        ArraysD::Void { layout, data } => ArraysD::Void {
            layout: layout.clone(),
            data: data.clone().index_axis_move(Axis(axis), n),
        },
    }
}

/// `a[idx] = value`. Modifies `a` in place. Supports scalar position,
/// integer-array fancy index along axis 0, and boolean-mask assign.
pub fn set_via_index(
    a: &mut ArraysD,
    parsed: &[IdxItem],
    value: &ArraysD,
    vm: &VirtualMachine,
) -> PyResult<()> {
    // ---- Single ndarray index: bool-mask or fancy ----
    if parsed.len() == 1 {
        match &parsed[0] {
            IdxItem::BoolMask(mask) => return set_bool_mask(a, mask, value, vm),
            IdxItem::IntArray(indices) => {
                return set_int_array(a, indices, value, vm);
            }
            _ => {}
        }
    }
    // Expand `...` and drop `None`/newaxis markers — they don't change the
    // underlying storage positions written.
    let nd = a.ndim();
    let parsed_owned: Vec<IdxItem> = expand_ellipsis(parsed, nd)
        .into_iter()
        .filter(|it| !matches!(it, IdxItem::NewAxis))
        .collect();
    let parsed: &[IdxItem] = &parsed_owned;
    // Fully-indexed scalar position.
    if parsed.len() == nd && parsed.iter().all(|k| matches!(k, IdxItem::Int(_))) {
        let mut norm = Vec::with_capacity(parsed.len());
        for (k, &dim) in parsed.iter().zip(a.shape()) {
            if let IdxItem::Int(v) = k {
                norm.push(normalize_idx(*v, dim, vm)?);
            }
        }
        if a.dtype().is_numeric() {
            let v = crate::convert::obj_as_scalar_from_array(value, vm)?;
            return set_scalar_at(a, IxDyn(&norm), v, vm);
        }
        // Non-numeric scalar assignment: copy the value's single element
        // directly (no f64 detour). value must be 0-D / single-element.
        return set_nonnumeric_scalar_at(a, IxDyn(&norm), value, vm);
    }
    // Slice/int-prefix assignment: write `value` (broadcast) into the
    // sub-view selected by the index path.
    let sub_shape = sub_shape_after_index(a, parsed, vm)?;
    let v_broadcast = broadcast_to_shape(value, &sub_shape, vm)?;
    set_subview(a, parsed, &v_broadcast, vm)
}

fn sub_shape_after_index(
    a: &ArraysD,
    parsed: &[IdxItem],
    vm: &VirtualMachine,
) -> PyResult<Vec<usize>> {
    let mut out = a.shape().to_vec();
    let mut cur_axis = 0usize;
    for item in parsed {
        match item {
            IdxItem::Int(_) => {
                out.remove(cur_axis);
            }
            IdxItem::Slice(s, e, st) => {
                let dim = out[cur_axis];
                let (start, end, step) = normalize_slice(*s, *e, *st, dim);
                let n = slice_count(start, end, step);
                out[cur_axis] = n;
                cur_axis += 1;
            }
            IdxItem::IntArray(_) | IdxItem::BoolMask(_) => {
                return Err(vm.new_index_error(
                    "advanced index in setitem must be the sole element".to_string(),
                ));
            }
            // Ellipsis and NewAxis are pre-expanded / dropped by set_via_index
            // before this is called, so reaching them here is a logic bug.
            IdxItem::Ellipsis | IdxItem::NewAxis => {
                return Err(internal(vm, "sub_shape_after_index: unexpected Ellipsis/NewAxis"));
            }
        }
    }
    Ok(out)
}

fn slice_count(start: isize, end: isize, step: isize) -> usize {
    if step == 0 {
        return 0;
    }
    if step > 0 {
        if start >= end {
            0
        } else {
            ((end - start - 1) / step + 1) as usize
        }
    } else if start <= end {
        0
    } else {
        ((start - end - 1) / (-step) + 1) as usize
    }
}

fn broadcast_to_shape(
    v: &ArraysD,
    shape: &[usize],
    vm: &VirtualMachine,
) -> PyResult<ArraysD> {
    crate::extras::broadcast_to(v, shape, vm)
}

fn set_subview(
    a: &mut ArraysD,
    parsed: &[IdxItem],
    value: &ArraysD,
    vm: &VirtualMachine,
) -> PyResult<()> {
    // Build the SliceInfoElem path that mirrors `parsed`.
    let nd = a.ndim();
    let mut info: Vec<SliceInfoElem> = vec![
        SliceInfoElem::Slice {
            start: 0,
            end: None,
            step: 1,
        };
        nd
    ];
    for (i, item) in parsed.iter().enumerate() {
        match item {
            IdxItem::Int(v) => {
                let dim = a.shape()[i];
                let n = normalize_idx(*v, dim, vm)?;
                info[i] = SliceInfoElem::Index(n as isize);
            }
            IdxItem::Slice(s, e, st) => {
                let dim = a.shape()[i];
                let (start, end, step) = normalize_slice(*s, *e, *st, dim);
                // ndarray's negative-step semantics: slice [start..end] then
                // reverse — translate from Python's start/end (where for neg
                // step start is the "first" index from the right).
                let (ndr_start, ndr_end) = if step < 0 {
                    (end + 1, start + 1)
                } else {
                    (start, end)
                };
                info[i] = SliceInfoElem::Slice {
                    start: ndr_start,
                    end: Some(ndr_end),
                    step,
                };
            }
            // IntArray / BoolMask are rejected earlier by `sub_shape_after_index`;
            // reaching here would be a logic bug, surface a clean error.
            IdxItem::IntArray(_) | IdxItem::BoolMask(_) => {
                return Err(internal(
                    vm,
                    "set_subview: advanced index in tuple position",
                ));
            }
            IdxItem::Ellipsis | IdxItem::NewAxis => {
                return Err(internal(vm, "set_subview: unexpected Ellipsis/NewAxis"));
            }
        }
    }
    let si = SliceInfo::<_, IxDyn, IxDyn>::try_from(info)
        .map_err(|e| vm.new_index_error(e.to_string()))?;
    // Cast `value` to the array's dtype, then copy.
    let v = value.cast(a.dtype());
    macro_rules! per {
        ($arr:ident, $val:ident) => {{
            let mut view = $arr.slice_mut(si.as_ref());
            ndarray::Zip::from(&mut view).and($val).for_each(|o, &x| *o = x);
        }};
    }
    match (a, &v) {
        (ArraysD::Bool(arr), ArraysD::Bool(val)) => per!(arr, val),
        (ArraysD::I8(arr), ArraysD::I8(val)) => per!(arr, val),
        (ArraysD::I16(arr), ArraysD::I16(val)) => per!(arr, val),
        (ArraysD::I32(arr), ArraysD::I32(val)) => per!(arr, val),
        (ArraysD::I64(arr), ArraysD::I64(val)) => per!(arr, val),
        (ArraysD::U8(arr), ArraysD::U8(val)) => per!(arr, val),
        (ArraysD::U16(arr), ArraysD::U16(val)) => per!(arr, val),
        (ArraysD::U32(arr), ArraysD::U32(val)) => per!(arr, val),
        (ArraysD::U64(arr), ArraysD::U64(val)) => per!(arr, val),
        (ArraysD::F16(arr), ArraysD::F16(val)) => per!(arr, val),
        (ArraysD::F32(arr), ArraysD::F32(val)) => per!(arr, val),
        (ArraysD::F64(arr), ArraysD::F64(val)) => per!(arr, val),
        (ArraysD::C64(arr), ArraysD::C64(val)) => per!(arr, val),
        (ArraysD::C128(arr), ArraysD::C128(val)) => per!(arr, val),
        _ => return Err(internal(vm, "set_subview: dtype mismatch after cast")),
    }
    Ok(())
}

fn set_bool_mask(
    a: &mut ArraysD,
    mask: &[bool],
    value: &ArraysD,
    vm: &VirtualMachine,
) -> PyResult<()> {
    if mask.len() != a.len() {
        return Err(vm.new_index_error(format!(
            "boolean mask length {} != array length {}",
            mask.len(),
            a.len()
        )));
    }
    let count = mask.iter().filter(|&&m| m).count();
    let provider = ValueProvider::from(value, count, vm)?;
    set_each(a, mask.iter().copied(), &provider, vm)
}

fn set_int_array(
    a: &mut ArraysD,
    indices: &[isize],
    value: &ArraysD,
    vm: &VirtualMachine,
) -> PyResult<()> {
    let axis0 = a.shape()[0];
    let normalized: Vec<usize> = indices
        .iter()
        .map(|&i| normalize_idx(i, axis0, vm))
        .collect::<PyResult<_>>()?;
    // value broadcasts: either scalar, or matches indices.len() along first axis.
    if value.ndim() == 0 {
        for &i in &normalized {
            let mut row = match a {
                ArraysD::Bool(arr) => {
                    set_row_scalar_bool(arr, i, value, vm)?;
                    continue;
                }
                _ => row_mut(a, i),
            };
            set_row_scalar_dispatch(&mut row, value, vm)?;
        }
        return Ok(());
    }
    if value.shape()[0] != normalized.len() {
        return Err(vm.new_value_error(format!(
            "cannot assign value of shape {:?} along fancy index of length {}",
            value.shape(),
            normalized.len()
        )));
    }
    let casted = value.cast(a.dtype());
    for (k, &i) in normalized.iter().enumerate() {
        let v_row = index_axis(&casted, 0, k);
        let parsed = vec![IdxItem::Int(i as isize)];
        set_subview(a, &parsed, &v_row, vm)?;
    }
    Ok(())
}

// --- helpers for bool-mask assignment ------------------------------------

struct ValueProvider {
    /// (dtype-erased) f64 fallback that we copy when value is a scalar.
    scalar_f64: Option<f64>,
    /// Or the full vector of casted values, one per True in the mask.
    full: Option<ArraysD>,
}

impl ValueProvider {
    fn from(value: &ArraysD, n_targets: usize, vm: &VirtualMachine) -> PyResult<Self> {
        if value.ndim() == 0 {
            // Scalar broadcasts.
            use crate::dtype::CoerceArray;
            let f = value
                .coerce::<f64>()
                .iter()
                .next()
                .copied()
                .or_internal(vm, "ValueProvider: empty 0-D scalar")?;
            Ok(Self {
                scalar_f64: Some(f),
                full: None,
            })
        } else if value.len() == n_targets {
            Ok(Self {
                scalar_f64: None,
                full: Some(crate::linalg::flatten(value)),
            })
        } else {
            Err(vm.new_value_error(format!(
                "boolean assignment: value length {} doesn't match {} True positions",
                value.len(),
                n_targets
            )))
        }
    }
}

fn set_each(
    a: &mut ArraysD,
    mask: impl Iterator<Item = bool>,
    provider: &ValueProvider,
    vm: &VirtualMachine,
) -> PyResult<()> {
    let mask: Vec<bool> = mask.collect();
    // Pre-cast the source values to the destination dtype so the inner
    // loop never needs a fallback branch (and therefore never has to
    // panic). `provider.full` is the flat source; we cast once.
    macro_rules! per {
        ($var:ident, $ty:ty, $arr:ident, $coerce:expr) => {{
            let cast_full: Option<ArraysD> = provider
                .full
                .as_ref()
                .map(|full| full.cast(<$ty as $crate::dtype::ArrayElement>::DTYPE));
            let mut vi = 0usize;
            for (slot, &m) in $arr.iter_mut().zip(mask.iter()) {
                if m {
                    let val: $ty = if let Some(f) = provider.scalar_f64 {
                        $coerce(f)
                    } else if let Some(ArraysD::$var(v)) = cast_full.as_ref() {
                        v[IxDyn(&[vi])]
                    } else {
                        // No scalar and the cast didn't land on this variant
                        // — fall back to the default value rather than panic.
                        <$ty as Default>::default()
                    };
                    *slot = val;
                    vi += 1;
                }
            }
        }};
    }
    match a {
        ArraysD::Bool(arr) => per!(Bool, bool, arr, |f: f64| f != 0.0),
        ArraysD::I8(arr) => per!(I8, i8, arr, |f: f64| f as i8),
        ArraysD::I16(arr) => per!(I16, i16, arr, |f: f64| f as i16),
        ArraysD::I32(arr) => per!(I32, i32, arr, |f: f64| f as i32),
        ArraysD::I64(arr) => per!(I64, i64, arr, |f: f64| f as i64),
        ArraysD::U8(arr) => per!(U8, u8, arr, |f: f64| f as u8),
        ArraysD::U16(arr) => per!(U16, u16, arr, |f: f64| f as u16),
        ArraysD::U32(arr) => per!(U32, u32, arr, |f: f64| f as u32),
        ArraysD::U64(arr) => per!(U64, u64, arr, |f: f64| f as u64),
        ArraysD::F16(arr) => per!(F16, half::f16, arr, |f: f64| half::f16::from_f64(f)),
        ArraysD::F32(arr) => per!(F32, f32, arr, |f: f64| f as f32),
        ArraysD::F64(arr) => per!(F64, f64, arr, |f: f64| f),
        ArraysD::C64(arr) => per!(
            C64,
            crate::dtype::C32,
            arr,
            |f: f64| crate::dtype::C32::new(f as f32, 0.0)
        ),
        ArraysD::C128(arr) => per!(
            C128,
            crate::dtype::C64,
            arr,
            |f: f64| crate::dtype::C64::new(f, 0.0)
        ),
        _ => { return Err(crate::internal::unsupported_dtype(vm, "set_each", a.dtype())) },
    }
    Ok(())
}

// --- helpers for fancy-int-array assignment ------------------------------

enum RowMut<'a> {
    Bool(ndarray::ArrayViewMutD<'a, bool>),
    I8(ndarray::ArrayViewMutD<'a, i8>),
    I16(ndarray::ArrayViewMutD<'a, i16>),
    I32(ndarray::ArrayViewMutD<'a, i32>),
    I64(ndarray::ArrayViewMutD<'a, i64>),
    U8(ndarray::ArrayViewMutD<'a, u8>),
    U16(ndarray::ArrayViewMutD<'a, u16>),
    U32(ndarray::ArrayViewMutD<'a, u32>),
    U64(ndarray::ArrayViewMutD<'a, u64>),
    F16(ndarray::ArrayViewMutD<'a, half::f16>),
    F32(ndarray::ArrayViewMutD<'a, f32>),
    F64(ndarray::ArrayViewMutD<'a, f64>),
    C64(ndarray::ArrayViewMutD<'a, crate::dtype::C32>),
    C128(ndarray::ArrayViewMutD<'a, crate::dtype::C64>),
    /// Sentinel for non-numeric variants: assignment helpers check for this
    /// and surface a `TypeError` to Python rather than panicking.
    Unsupported(DType),
}

fn row_mut(a: &mut ArraysD, i: usize) -> RowMut<'_> {
    match a {
        ArraysD::Bool(arr) => RowMut::Bool(arr.index_axis_mut(Axis(0), i)),
        ArraysD::I8(arr) => RowMut::I8(arr.index_axis_mut(Axis(0), i)),
        ArraysD::I16(arr) => RowMut::I16(arr.index_axis_mut(Axis(0), i)),
        ArraysD::I32(arr) => RowMut::I32(arr.index_axis_mut(Axis(0), i)),
        ArraysD::I64(arr) => RowMut::I64(arr.index_axis_mut(Axis(0), i)),
        ArraysD::U8(arr) => RowMut::U8(arr.index_axis_mut(Axis(0), i)),
        ArraysD::U16(arr) => RowMut::U16(arr.index_axis_mut(Axis(0), i)),
        ArraysD::U32(arr) => RowMut::U32(arr.index_axis_mut(Axis(0), i)),
        ArraysD::U64(arr) => RowMut::U64(arr.index_axis_mut(Axis(0), i)),
        ArraysD::F16(arr) => RowMut::F16(arr.index_axis_mut(Axis(0), i)),
        ArraysD::F32(arr) => RowMut::F32(arr.index_axis_mut(Axis(0), i)),
        ArraysD::F64(arr) => RowMut::F64(arr.index_axis_mut(Axis(0), i)),
        ArraysD::C64(arr) => RowMut::C64(arr.index_axis_mut(Axis(0), i)),
        ArraysD::C128(arr) => RowMut::C128(arr.index_axis_mut(Axis(0), i)),
        other => RowMut::Unsupported(other.dtype()),
    }
}

fn set_row_scalar_dispatch(
    row: &mut RowMut<'_>,
    value: &ArraysD,
    vm: &VirtualMachine,
) -> PyResult<()> {
    use crate::dtype::CoerceArray;
    let f = value
        .coerce::<f64>()
        .iter()
        .next()
        .copied()
        .or_internal(vm, "set_row_scalar_dispatch: empty scalar")?;
    match row {
        RowMut::Bool(v) => v.fill(f != 0.0),
        RowMut::I8(v) => v.fill(f as i8),
        RowMut::I16(v) => v.fill(f as i16),
        RowMut::I32(v) => v.fill(f as i32),
        RowMut::I64(v) => v.fill(f as i64),
        RowMut::U8(v) => v.fill(f as u8),
        RowMut::U16(v) => v.fill(f as u16),
        RowMut::U32(v) => v.fill(f as u32),
        RowMut::U64(v) => v.fill(f as u64),
        RowMut::F16(v) => v.fill(half::f16::from_f64(f)),
        RowMut::F32(v) => v.fill(f as f32),
        RowMut::F64(v) => v.fill(f),
        RowMut::C64(v) => v.fill(crate::dtype::C32::new(f as f32, 0.0)),
        RowMut::C128(v) => v.fill(crate::dtype::C64::new(f, 0.0)),
        RowMut::Unsupported(dt) => {
            return Err(crate::internal::unsupported_dtype(vm, "set_row_scalar", *dt));
        }
    }
    Ok(())
}

fn set_row_scalar_bool(
    arr: &mut ArrayD<bool>,
    i: usize,
    value: &ArraysD,
    vm: &VirtualMachine,
) -> PyResult<()> {
    use crate::dtype::CoerceArray;
    let f = value
        .coerce::<f64>()
        .iter()
        .next()
        .copied()
        .or_internal(vm, "set_row_scalar_bool: empty scalar")?;
    let mut row = arr.index_axis_mut(Axis(0), i);
    row.fill(f != 0.0);
    Ok(())
}

/// Scalar assignment for non-numeric arrays. `value` is converted to the
/// same dtype as `a` (which copies its single element) and that element is
/// written into `a[ix]`.
fn set_nonnumeric_scalar_at(
    a: &mut ArraysD,
    ix: IxDyn,
    value: &ArraysD,
    vm: &VirtualMachine,
) -> PyResult<()> {
    if value.len() != 1 {
        return Err(vm.new_value_error(format!(
            "expected a scalar value for assignment, got shape {:?}",
            value.shape()
        )));
    }
    let coerced = value.cast(a.dtype());
    macro_rules! per {
        ($dst:ident, $src:ident) => {{
            let v = $src
                .iter()
                .next()
                .cloned()
                .ok_or_else(|| crate::internal::internal(vm, "set_nonnumeric_scalar_at: empty src"))?;
            $dst[ix] = v;
        }};
    }
    match (a, coerced) {
        (ArraysD::Str { data: dst, .. }, ArraysD::Str { data: src, .. }) => per!(dst, src),
        (ArraysD::Bytes { data: dst, .. }, ArraysD::Bytes { data: src, .. }) => per!(dst, src),
        (ArraysD::Object(dst), ArraysD::Object(src)) => per!(dst, src),
        (ArraysD::Datetime64 { data: dst, .. }, ArraysD::Datetime64 { data: src, .. }) => per!(dst, src),
        (ArraysD::Timedelta64 { data: dst, .. }, ArraysD::Timedelta64 { data: src, .. }) => per!(dst, src),
        (ArraysD::Void { data: dst, .. }, ArraysD::Void { data: src, .. }) => per!(dst, src),
        (dst, _) => return Err(crate::internal::unsupported_dtype(vm, "set_nonnumeric_scalar_at", dst.dtype())),
    }
    Ok(())
}

fn set_scalar_at(
    a: &mut ArraysD,
    ix: IxDyn,
    scalar: f64,
    vm: &VirtualMachine,
) -> PyResult<()> {
    match a {
        ArraysD::Bool(arr) => arr[ix] = scalar != 0.0,
        ArraysD::I8(arr) => arr[ix] = scalar as i8,
        ArraysD::I16(arr) => arr[ix] = scalar as i16,
        ArraysD::I32(arr) => arr[ix] = scalar as i32,
        ArraysD::I64(arr) => arr[ix] = scalar as i64,
        ArraysD::U8(arr) => arr[ix] = scalar as u8,
        ArraysD::U16(arr) => arr[ix] = scalar as u16,
        ArraysD::U32(arr) => arr[ix] = scalar as u32,
        ArraysD::U64(arr) => arr[ix] = scalar as u64,
        ArraysD::F16(arr) => arr[ix] = half::f16::from_f64(scalar),
        ArraysD::F32(arr) => arr[ix] = scalar as f32,
        ArraysD::F64(arr) => arr[ix] = scalar,
        ArraysD::C64(arr) => arr[ix] = crate::dtype::C32::new(scalar as f32, 0.0),
        ArraysD::C128(arr) => arr[ix] = crate::dtype::C64::new(scalar, 0.0),
        _ => { return Err(crate::internal::unsupported_dtype(vm, "set_scalar_at", a.dtype())) },
    }
    Ok(())
}

fn slice_axis(
    a: &ArraysD,
    axis: usize,
    start: isize,
    end: isize,
    step: isize,
    vm: &VirtualMachine,
) -> PyResult<ArraysD> {
    let nd = a.ndim();
    let mut info: Vec<SliceInfoElem> = vec![
        SliceInfoElem::Slice { start: 0, end: None, step: 1 };
        nd
    ];
    // Python's negative-step semantics differ from ndarray's. In Python,
    // `a[start:end:-1]` iterates start, start-1, …, end+1; ndarray instead
    // slices `[start..end]` left-to-right *then* reverses if step<0.
    //
    // Convert Python (start, end, -|s|) into ndarray (end+1, start+1, -|s|).
    let (ndr_start, ndr_end) = if step < 0 {
        (end + 1, start + 1)
    } else {
        (start, end)
    };
    info[axis] = SliceInfoElem::Slice {
        start: ndr_start,
        end: Some(ndr_end),
        step,
    };
    let si = SliceInfo::<_, IxDyn, IxDyn>::try_from(info)
        .map_err(|e| vm.new_index_error(e.to_string()))?;
    macro_rules! per {
        ($var:ident, $arr:ident) => {{ ArraysD::$var($arr.slice(si.as_ref()).to_owned()) }};
    }
    Ok(match a {
        ArraysD::Bool(arr) => per!(Bool, arr),
        ArraysD::I8(arr) => per!(I8, arr),
        ArraysD::I16(arr) => per!(I16, arr),
        ArraysD::I32(arr) => per!(I32, arr),
        ArraysD::I64(arr) => per!(I64, arr),
        ArraysD::U8(arr) => per!(U8, arr),
        ArraysD::U16(arr) => per!(U16, arr),
        ArraysD::U32(arr) => per!(U32, arr),
        ArraysD::U64(arr) => per!(U64, arr),
        ArraysD::F16(arr) => per!(F16, arr),
        ArraysD::F32(arr) => per!(F32, arr),
        ArraysD::F64(arr) => per!(F64, arr),
        ArraysD::C64(arr) => per!(C64, arr),
        ArraysD::C128(arr) => per!(C128, arr),
        ArraysD::Object(arr) => ArraysD::Object(arr.slice(si.as_ref()).to_owned()),
        ArraysD::Str { itemsize_chars, data } => ArraysD::Str {
            itemsize_chars: *itemsize_chars,
            data: data.slice(si.as_ref()).to_owned(),
        },
        ArraysD::Bytes { itemsize, data } => ArraysD::Bytes {
            itemsize: *itemsize,
            data: data.slice(si.as_ref()).to_owned(),
        },
        ArraysD::Datetime64 { unit, data } => ArraysD::Datetime64 {
            unit: *unit,
            data: data.slice(si.as_ref()).to_owned(),
        },
        ArraysD::Timedelta64 { unit, data } => ArraysD::Timedelta64 {
            unit: *unit,
            data: data.slice(si.as_ref()).to_owned(),
        },
        ArraysD::Void { layout, data } => ArraysD::Void {
            layout: layout.clone(),
            data: data.slice(si.as_ref()).to_owned(),
        },
    })
}
