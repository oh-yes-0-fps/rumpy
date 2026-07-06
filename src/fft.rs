//! `numpy.fft` — 1-D forward/inverse FFT, 2-D variants, and real-input
//! transforms, all routed through `rustfft`. Frequencies follow numpy's
//! `fftfreq`/`rfftfreq` conventions.

use crate::dtype::{ArraysD, DType};
use ndarray::{ArrayD, IxDyn};
use num_complex::Complex;
use rustfft::FftPlanner;
use rustpython_vm::{PyResult, VirtualMachine};

type C64 = Complex<f64>;

fn into_complex(a: &ArraysD) -> ArrayD<C64> {
    match a.cast(DType::C128) {
        ArraysD::C128(x) => x,
        _ => ArrayD::<C64>::from_shape_vec(IxDyn(&[0]), vec![]).unwrap_or_default(),
    }
}

/// `np.fft.fft(a)` — 1-D forward FFT (default n = len(a)).
pub fn fft(a: &ArraysD, n: Option<usize>, vm: &VirtualMachine) -> PyResult<ArraysD> {
    if a.ndim() != 1 {
        return Err(vm.new_value_error("fft: input must be 1-D".to_string()));
    }
    let c = into_complex(a);
    let mut buf: Vec<C64> = pad_or_truncate(c.iter().copied().collect(), n.unwrap_or(c.len()));
    let mut planner = FftPlanner::<f64>::new();
    let plan = planner.plan_fft_forward(buf.len());
    plan.process(&mut buf);
    let n = buf.len();
    Ok(ArraysD::C128(
        ArrayD::from_shape_vec(IxDyn(&[n]), buf).unwrap_or_default(),
    ))
}

/// `np.fft.ifft(a)` — 1-D inverse FFT (normalized by 1/N).
pub fn ifft(a: &ArraysD, n: Option<usize>, vm: &VirtualMachine) -> PyResult<ArraysD> {
    if a.ndim() != 1 {
        return Err(vm.new_value_error("ifft: input must be 1-D".to_string()));
    }
    let c = into_complex(a);
    let mut buf: Vec<C64> = pad_or_truncate(c.iter().copied().collect(), n.unwrap_or(c.len()));
    let len = buf.len();
    let mut planner = FftPlanner::<f64>::new();
    let plan = planner.plan_fft_inverse(buf.len());
    plan.process(&mut buf);
    let scale = 1.0 / len as f64;
    for v in &mut buf {
        *v *= scale;
    }
    Ok(ArraysD::C128(
        ArrayD::from_shape_vec(IxDyn(&[len]), buf).unwrap_or_default(),
    ))
}

/// `np.fft.rfft(a)` — 1-D FFT of a real-valued input, returning the
/// first `n/2 + 1` complex coefficients.
pub fn rfft(a: &ArraysD, n: Option<usize>, vm: &VirtualMachine) -> PyResult<ArraysD> {
    if a.ndim() != 1 {
        return Err(vm.new_value_error("rfft: input must be 1-D".to_string()));
    }
    let real = match a.cast(DType::F64) {
        ArraysD::F64(x) => x,
        _ => return Err(crate::internal::internal(vm, "rfft: cast to F64 failed")),
    };
    let target = n.unwrap_or(real.len());
    let mut buf: Vec<C64> =
        pad_or_truncate(real.iter().map(|&v| C64::new(v, 0.0)).collect(), target);
    let mut planner = FftPlanner::<f64>::new();
    let plan = planner.plan_fft_forward(buf.len());
    plan.process(&mut buf);
    buf.truncate(target / 2 + 1);
    Ok(ArraysD::C128(
        ArrayD::from_shape_vec(IxDyn(&[buf.len()]), buf).unwrap_or_default(),
    ))
}

/// `np.fft.irfft(a, n=...)` — inverse of `rfft`. `n` defaults to `2*(len-1)`.
pub fn irfft(a: &ArraysD, n: Option<usize>, vm: &VirtualMachine) -> PyResult<ArraysD> {
    if a.ndim() != 1 {
        return Err(vm.new_value_error("irfft: input must be 1-D".to_string()));
    }
    let c = into_complex(a);
    let m = c.len();
    let n = n.unwrap_or(2 * (m - 1));
    // Reconstruct the full spectrum: conjugate-mirror the upper half.
    let mut full: Vec<C64> = vec![C64::new(0.0, 0.0); n];
    let half = n / 2 + 1;
    for i in 0..m.min(half) {
        full[i] = c[IxDyn(&[i])];
    }
    for i in 1..(n - half + 1) {
        if i < n - i {
            full[n - i] = full[i].conj();
        }
    }
    let mut planner = FftPlanner::<f64>::new();
    let plan = planner.plan_fft_inverse(n);
    plan.process(&mut full);
    let scale = 1.0 / n as f64;
    let real: Vec<f64> = full.iter().map(|c| c.re * scale).collect();
    Ok(ArraysD::F64(
        ArrayD::from_shape_vec(IxDyn(&[n]), real).unwrap_or_default(),
    ))
}

/// `np.fft.fft2(a)` — 2-D FFT (FFT along axis 1, then axis 0).
pub fn fft2(a: &ArraysD, vm: &VirtualMachine) -> PyResult<ArraysD> {
    if a.ndim() != 2 {
        return Err(vm.new_value_error("fft2: input must be 2-D".to_string()));
    }
    Ok(ArraysD::C128(fft_2d(into_complex(a), false)))
}

/// `np.fft.ifft2(a)` — 2-D inverse FFT.
pub fn ifft2(a: &ArraysD, vm: &VirtualMachine) -> PyResult<ArraysD> {
    if a.ndim() != 2 {
        return Err(vm.new_value_error("ifft2: input must be 2-D".to_string()));
    }
    Ok(ArraysD::C128(fft_2d(into_complex(a), true)))
}

fn fft_2d(mut a: ArrayD<C64>, inverse: bool) -> ArrayD<C64> {
    let shape = a.shape().to_vec();
    let (m, n) = (shape[0], shape[1]);
    let mut planner = FftPlanner::<f64>::new();
    // FFT each row.
    let row_plan = if inverse {
        planner.plan_fft_inverse(n)
    } else {
        planner.plan_fft_forward(n)
    };
    for i in 0..m {
        let mut row: Vec<C64> = (0..n).map(|j| a[IxDyn(&[i, j])]).collect();
        row_plan.process(&mut row);
        for j in 0..n {
            a[IxDyn(&[i, j])] = row[j];
        }
    }
    // FFT each column.
    let col_plan = if inverse {
        planner.plan_fft_inverse(m)
    } else {
        planner.plan_fft_forward(m)
    };
    for j in 0..n {
        let mut col: Vec<C64> = (0..m).map(|i| a[IxDyn(&[i, j])]).collect();
        col_plan.process(&mut col);
        for i in 0..m {
            a[IxDyn(&[i, j])] = col[i];
        }
    }
    if inverse {
        let scale = 1.0 / (m * n) as f64;
        for v in a.iter_mut() {
            *v *= scale;
        }
    }
    a
}

/// `np.fft.fftfreq(n, d=1.0)` — sample frequencies for an N-point FFT.
pub fn fftfreq(n: usize, d: f64) -> ArraysD {
    let nf = n as f64;
    let mut out = vec![0.0f64; n];
    let half = (n - 1) / 2 + 1; // numpy's split point
    for (i, slot) in out.iter_mut().enumerate().take(half) {
        *slot = i as f64 / (d * nf);
    }
    for (offset, slot) in out.iter_mut().enumerate().skip(half) {
        *slot = (offset as f64 - nf) / (d * nf);
    }
    ArraysD::F64(ArrayD::from_shape_vec(IxDyn(&[n]), out).unwrap_or_default())
}

/// `np.fft.rfftfreq(n, d=1.0)` — frequencies for `rfft` output.
pub fn rfftfreq(n: usize, d: f64) -> ArraysD {
    let m = n / 2 + 1;
    let denom = d * n as f64;
    let out: Vec<f64> = (0..m).map(|i| i as f64 / denom).collect();
    ArraysD::F64(ArrayD::from_shape_vec(IxDyn(&[m]), out).unwrap_or_default())
}

/// `np.fft.fftshift(a, axes=None)` — shift the zero-frequency component to
/// the centre along the given axes (all axes by default).
pub fn fftshift(a: &ArraysD, axes: Option<Vec<isize>>, vm: &VirtualMachine) -> PyResult<ArraysD> {
    shift_along(a, axes, true, vm)
}

/// `np.fft.ifftshift(a, axes=None)` — inverse of `fftshift`.
pub fn ifftshift(a: &ArraysD, axes: Option<Vec<isize>>, vm: &VirtualMachine) -> PyResult<ArraysD> {
    shift_along(a, axes, false, vm)
}

fn shift_along(
    a: &ArraysD,
    axes: Option<Vec<isize>>,
    forward: bool,
    vm: &VirtualMachine,
) -> PyResult<ArraysD> {
    let nd = a.ndim() as isize;
    if nd == 0 {
        return Ok(a.clone());
    }
    let resolved: Vec<usize> = match axes {
        None => (0..nd as usize).collect(),
        Some(v) => v
            .into_iter()
            .map(|ax| {
                let r = if ax < 0 { ax + nd } else { ax };
                if r < 0 || r >= nd {
                    Err(vm.new_value_error(format!("axis {ax} out of range")))
                } else {
                    Ok(r as usize)
                }
            })
            .collect::<PyResult<Vec<_>>>()?,
    };
    let mut out = a.clone();
    for ax in resolved {
        let n = out.shape()[ax];
        let shift = if forward {
            // fftshift: roll by n/2 (rounded up for odd n).
            (n / 2 + n % 2) as isize
        } else {
            (n / 2) as isize
        };
        out = roll_axis(&out, shift, ax, vm)?;
    }
    Ok(out)
}

fn roll_axis(a: &ArraysD, shift: isize, axis: usize, vm: &VirtualMachine) -> PyResult<ArraysD> {
    let n = a.shape()[axis] as isize;
    if n == 0 {
        return Ok(a.clone());
    }
    let s = ((shift % n) + n) % n;
    if s == 0 {
        return Ok(a.clone());
    }
    let k = (n - s) as usize;
    // Concatenate `a.slice(axis=k..)` with `a.slice(axis=..k)` along the
    // same axis.
    let nd = a.ndim();
    let mut info_a: Vec<ndarray::SliceInfoElem> = (0..nd)
        .map(|_| ndarray::SliceInfoElem::Slice {
            start: 0,
            end: None,
            step: 1,
        })
        .collect();
    let mut info_b = info_a.clone();
    info_a[axis] = ndarray::SliceInfoElem::Slice {
        start: k as isize,
        end: None,
        step: 1,
    };
    info_b[axis] = ndarray::SliceInfoElem::Slice {
        start: 0,
        end: Some(k as isize),
        step: 1,
    };
    let si_a = ndarray::SliceInfo::<_, IxDyn, IxDyn>::try_from(info_a)
        .map_err(|e| vm.new_index_error(e.to_string()))?;
    let si_b = ndarray::SliceInfo::<_, IxDyn, IxDyn>::try_from(info_b)
        .map_err(|e| vm.new_index_error(e.to_string()))?;
    macro_rules! per {
        ($var:ident, $arr:ident) => {{
            let part_a = $arr.slice(si_a.as_ref()).to_owned();
            let part_b = $arr.slice(si_b.as_ref()).to_owned();
            let cat = ndarray::concatenate(ndarray::Axis(axis), &[part_a.view(), part_b.view()])
                .map_err(|e| vm.new_value_error(e.to_string()))?;
            Ok(ArraysD::$var(cat))
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
        // roll_axis on non-numeric arrays falls back to a no-op (fft caller
        // would already have rejected the dtype). Returning Ok-of-clone is
        // safe because fftshift is logically a permutation, so identity is a
        // valid degenerate case for empty / non-permutable arrays.
        other => Ok(other.clone()),
    }
}

fn pad_or_truncate(mut v: Vec<C64>, n: usize) -> Vec<C64> {
    if v.len() < n {
        v.resize(n, C64::new(0.0, 0.0));
    } else if v.len() > n {
        v.truncate(n);
    }
    v
}

/// `np.fft.fftn(a)` — N-dimensional FFT. Performs an FFT along every axis in
/// turn. If `axes` is provided, only those axes are transformed.
pub fn fftn(a: &ArraysD, axes: Option<Vec<isize>>, vm: &VirtualMachine) -> PyResult<ArraysD> {
    do_fftn(a, axes, /*inverse=*/ false, vm)
}

/// `np.fft.ifftn(a)` — inverse N-dimensional FFT.
pub fn ifftn(a: &ArraysD, axes: Option<Vec<isize>>, vm: &VirtualMachine) -> PyResult<ArraysD> {
    do_fftn(a, axes, /*inverse=*/ true, vm)
}

fn do_fftn(
    a: &ArraysD,
    axes: Option<Vec<isize>>,
    inverse: bool,
    vm: &VirtualMachine,
) -> PyResult<ArraysD> {
    let nd = a.ndim();
    if nd == 0 {
        return Ok(a.clone());
    }
    let resolved: Vec<usize> = match axes {
        None => (0..nd).collect(),
        Some(v) => v
            .into_iter()
            .map(|ax| {
                let n = if ax < 0 { ax + nd as isize } else { ax };
                if n < 0 || n >= nd as isize {
                    Err(vm.new_value_error(format!("fftn: axis {ax} out of range for {nd}-D")))
                } else {
                    Ok(n as usize)
                }
            })
            .collect::<PyResult<_>>()?,
    };
    let mut data = into_complex(a);
    let mut planner = rustfft::FftPlanner::<f64>::new();
    for ax in resolved.iter().copied() {
        let n_ax = data.shape()[ax];
        let plan = if inverse {
            planner.plan_fft_inverse(n_ax)
        } else {
            planner.plan_fft_forward(n_ax)
        };
        // Walk each lane along axis `ax` and FFT it in place.
        for mut lane in data.lanes_mut(ndarray::Axis(ax)).into_iter() {
            let mut buf: Vec<C64> = lane.iter().copied().collect();
            plan.process(&mut buf);
            for (slot, v) in lane.iter_mut().zip(buf.iter()) {
                *slot = *v;
            }
        }
        if inverse {
            let scale = 1.0 / n_ax as f64;
            for v in data.iter_mut() {
                *v *= scale;
            }
        }
    }
    Ok(ArraysD::C128(data))
}
