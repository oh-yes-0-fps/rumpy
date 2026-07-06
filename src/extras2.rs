//! Additional creation helpers, trig auxiliaries, and stats — the
//! "obvious-numpy" surface that wasn't in the first cut.

use crate::dtype::{ArraysD, DType};
use crate::internal::{OptionExt, internal};
use ndarray::{ArrayD, IxDyn};
use rustpython_vm::{PyResult, VirtualMachine};

/// Cast to F64 and unwrap; on the unreachable dtype-mismatch arm, return an
/// empty array as a panic-free fallback.
fn cast_f64_or_empty(a: &ArraysD) -> ArrayD<f64> {
    match a.cast(DType::F64) {
        ArraysD::F64(x) => x,
        _ => ArrayD::<f64>::zeros(IxDyn(&[0])),
    }
}

// =====================================================================
// Creation
// =====================================================================

pub fn full_like(a: &ArraysD, value: f64) -> ArraysD {
    crate::create::full_f64(a.shape(), value, a.dtype())
}

pub fn empty_like(a: &ArraysD) -> ArraysD {
    crate::create::zeros(a.shape(), a.dtype())
}

pub fn logspace(start: f64, stop: f64, num: usize, base: f64, endpoint: bool) -> ArraysD {
    let lin = lin_inner(start, stop, num, endpoint);
    let data: Vec<f64> = lin.into_iter().map(|e| base.powf(e)).collect();
    ArraysD::F64(ArrayD::from_shape_vec(IxDyn(&[data.len()]), data).unwrap_or_default())
}

pub fn geomspace(start: f64, stop: f64, num: usize, endpoint: bool) -> Option<ArraysD> {
    if start == 0.0 || stop == 0.0 || start.signum() != stop.signum() {
        return None;
    }
    let log_start = start.abs().ln();
    let log_stop = stop.abs().ln();
    let lin = lin_inner(log_start, log_stop, num, endpoint);
    let sign = start.signum();
    let data: Vec<f64> = lin.into_iter().map(|e| sign * e.exp()).collect();
    Some(ArraysD::F64(
        ArrayD::from_shape_vec(IxDyn(&[data.len()]), data).unwrap_or_default(),
    ))
}

/// Shared linspace innards — returns the `num` evenly-spaced values used by
/// `linspace`, `logspace`, `geomspace`.
pub fn lin_inner(start: f64, stop: f64, num: usize, endpoint: bool) -> Vec<f64> {
    if num == 0 {
        return Vec::new();
    }
    if num == 1 {
        return vec![start];
    }
    let denom = if endpoint {
        (num - 1) as f64
    } else {
        num as f64
    };
    let step = (stop - start) / denom;
    (0..num).map(|i| start + step * i as f64).collect()
}

/// `numpy.linspace(..., endpoint=True, retstep=False)`.
pub fn linspace_full(start: f64, stop: f64, num: usize, endpoint: bool) -> (ArraysD, f64) {
    let v = lin_inner(start, stop, num, endpoint);
    let step = if num < 2 {
        f64::NAN
    } else {
        let denom = if endpoint {
            (num - 1) as f64
        } else {
            num as f64
        };
        (stop - start) / denom
    };
    (
        ArraysD::F64(ArrayD::from_shape_vec(IxDyn(&[v.len()]), v).unwrap_or_default()),
        step,
    )
}

// =====================================================================
// Trig & math helpers
// =====================================================================

pub fn arctan2(y: &ArraysD, x: &ArraysD, vm: &VirtualMachine) -> PyResult<ArraysD> {
    let pt = crate::promote::promote(y.dtype(), x.dtype());
    let widen = if pt.is_integer() { DType::F64 } else { pt };
    let yf = y.cast(widen);
    let xf = x.cast(widen);
    let shape = broadcast(yf.shape(), xf.shape())
        .ok_or_else(|| vm.new_value_error("broadcast failure".to_string()))?;
    let s = IxDyn(&shape);
    Ok(match (&yf, &xf) {
        (ArraysD::F32(y), ArraysD::F32(x)) => {
            let yv = y.broadcast(s.clone()).or_internal(vm, "arctan2: bcast y")?;
            let xv = x.broadcast(s.clone()).or_internal(vm, "arctan2: bcast x")?;
            let mut out = ArrayD::<f32>::zeros(s.clone());
            ndarray::Zip::from(&mut out)
                .and(&yv)
                .and(&xv)
                .for_each(|o, &p, &q| *o = p.atan2(q));
            ArraysD::F32(out)
        }
        (ArraysD::F64(y), ArraysD::F64(x)) => {
            let yv = y.broadcast(s.clone()).or_internal(vm, "arctan2: bcast y")?;
            let xv = x.broadcast(s.clone()).or_internal(vm, "arctan2: bcast x")?;
            let mut out = ArrayD::<f64>::zeros(s.clone());
            ndarray::Zip::from(&mut out)
                .and(&yv)
                .and(&xv)
                .for_each(|o, &p, &q| *o = p.atan2(q));
            ArraysD::F64(out)
        }
        _ => return Err(vm.new_type_error("arctan2 needs real numeric inputs".to_string())),
    })
}

pub fn hypot(x: &ArraysD, y: &ArraysD, vm: &VirtualMachine) -> PyResult<ArraysD> {
    let pt = crate::promote::promote(x.dtype(), y.dtype());
    let widen = if pt.is_integer() { DType::F64 } else { pt };
    let xf = x.cast(widen);
    let yf = y.cast(widen);
    let shape = broadcast(xf.shape(), yf.shape())
        .ok_or_else(|| vm.new_value_error("broadcast failure".to_string()))?;
    let s = IxDyn(&shape);
    Ok(match (&xf, &yf) {
        (ArraysD::F32(x), ArraysD::F32(y)) => {
            let xv = x.broadcast(s.clone()).or_internal(vm, "hypot: bcast x")?;
            let yv = y.broadcast(s.clone()).or_internal(vm, "hypot: bcast y")?;
            let mut out = ArrayD::<f32>::zeros(s.clone());
            ndarray::Zip::from(&mut out)
                .and(&xv)
                .and(&yv)
                .for_each(|o, &p, &q| *o = p.hypot(q));
            ArraysD::F32(out)
        }
        (ArraysD::F64(x), ArraysD::F64(y)) => {
            let xv = x.broadcast(s.clone()).or_internal(vm, "hypot: bcast x")?;
            let yv = y.broadcast(s.clone()).or_internal(vm, "hypot: bcast y")?;
            let mut out = ArrayD::<f64>::zeros(s.clone());
            ndarray::Zip::from(&mut out)
                .and(&xv)
                .and(&yv)
                .for_each(|o, &p, &q| *o = p.hypot(q));
            ArraysD::F64(out)
        }
        _ => return Err(vm.new_type_error("hypot needs real numeric inputs".to_string())),
    })
}

pub fn deg2rad(a: &ArraysD) -> ArraysD {
    crate::ops::unary_real_or_complex(
        a,
        |x| x.to_radians(),
        |c| c * (std::f64::consts::PI / 180.0),
    )
}

pub fn rad2deg(a: &ArraysD) -> ArraysD {
    crate::ops::unary_real_or_complex(
        a,
        |x| x.to_degrees(),
        |c| c * (180.0 / std::f64::consts::PI),
    )
}

/// `numpy.unwrap` — adjust phase so jumps > discont (default π) get unwrapped.
pub fn unwrap(a: &ArraysD, discont: f64) -> ArraysD {
    let f = cast_f64_or_empty(a);
    let n = f.len();
    if n < 2 {
        return ArraysD::F64(f);
    }
    let mut out = vec![0.0f64; n];
    let mut shift = 0.0;
    out[0] = f[IxDyn(&[0])];
    let period = 2.0 * std::f64::consts::PI;
    for i in 1..n {
        let raw_d = f[IxDyn(&[i])] - f[IxDyn(&[i - 1])];
        let mut d = raw_d;
        if d > discont {
            d -= period;
        } else if d < -discont {
            d += period;
        }
        shift += d - raw_d;
        out[i] = f[IxDyn(&[i])] + shift;
    }
    ArraysD::F64(ArrayD::from_shape_vec(IxDyn(&[n]), out).unwrap_or_default())
}

// =====================================================================
// Stats
// =====================================================================

/// `numpy.average` — weighted mean. If `weights` is None, falls back to
/// arithmetic mean.
pub fn average(a: &ArraysD, weights: Option<&ArraysD>, vm: &VirtualMachine) -> PyResult<ArraysD> {
    let af = cast_f64_or_empty(a);
    let result = match weights {
        None => {
            let n = af.len();
            if n == 0 {
                return Err(vm.new_value_error("average of empty array".to_string()));
            }
            af.iter().copied().sum::<f64>() / n as f64
        }
        Some(w) => {
            let wf = cast_f64_or_empty(w);
            if wf.shape() != af.shape() {
                return Err(vm.new_value_error(format!(
                    "average: weights shape {:?} != input shape {:?}",
                    wf.shape(),
                    af.shape()
                )));
            }
            let mut wsum = 0.0;
            let mut acc = 0.0;
            for (x, w) in af.iter().zip(wf.iter()) {
                acc += x * w;
                wsum += w;
            }
            if wsum == 0.0 {
                return Err(vm.new_value_error("average: weights sum to zero".to_string()));
            }
            acc / wsum
        }
    };
    Ok(ArraysD::F64(ArrayD::from_elem(IxDyn(&[]), result)))
}

/// `numpy.percentile(a, q)` with linear interpolation (numpy's default).
/// `q` is in [0, 100]. Returns a 0-D array.
pub fn percentile(a: &ArraysD, q: f64, vm: &VirtualMachine) -> PyResult<ArraysD> {
    if !(0.0..=100.0).contains(&q) {
        return Err(vm.new_value_error("percentile q must be in [0, 100]".to_string()));
    }
    quantile(a, q / 100.0, vm)
}

/// `q` is in [0, 100]. Returns the raw f64 value.
pub fn percentile_scalar(a: &ArraysD, q: f64, vm: &VirtualMachine) -> PyResult<f64> {
    let res = percentile(a, q, vm)?;
    use crate::dtype::CoerceArray;
    Ok(res
        .coerce::<f64>()
        .iter()
        .next()
        .copied()
        .unwrap_or(f64::NAN))
}

pub fn quantile(a: &ArraysD, q: f64, vm: &VirtualMachine) -> PyResult<ArraysD> {
    if !(0.0..=1.0).contains(&q) {
        return Err(vm.new_value_error("quantile q must be in [0, 1]".to_string()));
    }
    let af = cast_f64_or_empty(a);
    let mut v: Vec<f64> = af.iter().copied().filter(|x| !x.is_nan()).collect();
    if v.is_empty() {
        return Err(vm.new_value_error("quantile of empty array".to_string()));
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = v.len();
    let pos = q * (n - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    let frac = pos - lo as f64;
    let result = v[lo] + frac * (v[hi] - v[lo]);
    Ok(ArraysD::F64(ArrayD::from_elem(IxDyn(&[]), result)))
}

/// `numpy.cov(m, y=None, ddof=1)` — covariance matrix. `m` is treated as
/// observations × variables when given a 2-D array (rows = variables,
/// cols = observations — numpy's default with `rowvar=True`). For a 1-D
/// input we treat it as a single variable, returning a 1×1 result.
pub fn cov(m: &ArraysD, ddof: usize, vm: &VirtualMachine) -> PyResult<ArraysD> {
    let mat = to_2d_rowvar(m);
    let (nvars, nobs) = (mat.dim().0, mat.dim().1);
    if nobs <= ddof {
        return Err(vm.new_value_error("cov: not enough observations".to_string()));
    }
    // De-mean each row.
    let mut centered = mat.clone();
    for i in 0..nvars {
        let mean = (0..nobs).map(|j| centered[(i, j)]).sum::<f64>() / nobs as f64;
        for j in 0..nobs {
            centered[(i, j)] -= mean;
        }
    }
    let scale = (nobs - ddof) as f64;
    let mut out = ndarray::Array2::<f64>::zeros((nvars, nvars));
    for i in 0..nvars {
        for j in 0..nvars {
            let mut s = 0.0;
            for k in 0..nobs {
                s += centered[(i, k)] * centered[(j, k)];
            }
            out[(i, j)] = s / scale;
        }
    }
    Ok(ArraysD::F64(out.into_dyn()))
}

/// `numpy.corrcoef` — like cov but normalized to ±1 by std.
pub fn corrcoef(m: &ArraysD, vm: &VirtualMachine) -> PyResult<ArraysD> {
    let c = cov(m, 1, vm)?;
    let mat = match c {
        ArraysD::F64(x) => x
            .into_dimensionality::<ndarray::Ix2>()
            .map_err(|e| internal(vm, format!("corrcoef: cov result not 2-D: {e}")))?,
        _ => return Err(internal(vm, "corrcoef: cov returned non-F64")),
    };
    let n = mat.dim().0;
    let mut out = ndarray::Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            let denom = (mat[(i, i)] * mat[(j, j)]).sqrt();
            out[(i, j)] = if denom == 0.0 {
                0.0
            } else {
                mat[(i, j)] / denom
            };
        }
    }
    Ok(ArraysD::F64(out.into_dyn()))
}

fn to_2d_rowvar(a: &ArraysD) -> ndarray::Array2<f64> {
    let f = cast_f64_or_empty(a);
    match f.ndim() {
        0 => {
            // 0-D: read the scalar (or 0.0 if somehow empty) and wrap as 1×1.
            let v = f.iter().copied().next().unwrap_or(0.0);
            ndarray::Array2::from_shape_vec((1, 1), vec![v]).unwrap_or_default()
        }
        1 => {
            let n = f.len();
            ndarray::Array2::from_shape_vec((1, n), f.iter().copied().collect()).unwrap_or_default()
        }
        _ => f
            .into_dimensionality::<ndarray::Ix2>()
            .unwrap_or_else(|_| ndarray::Array2::<f64>::zeros((0, 0))),
    }
}

// =====================================================================
// Internal: broadcast shape
// =====================================================================

fn broadcast(a: &[usize], b: &[usize]) -> Option<Vec<usize>> {
    let nd = a.len().max(b.len());
    let mut out = vec![1usize; nd];
    for i in 0..nd {
        let da = if i + a.len() >= nd {
            a[i + a.len() - nd]
        } else {
            1
        };
        let db = if i + b.len() >= nd {
            b[i + b.len() - nd]
        } else {
            1
        };
        out[i] = match (da, db) {
            (x, y) if x == y => x,
            (1, y) => y,
            (x, 1) => x,
            _ => return None,
        };
    }
    Some(out)
}
