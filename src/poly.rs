//! Polynomial helpers: `polyval`, `roots`, `polyfit`.
//!
//! The legacy `numpy.poly*` API uses **descending** coefficient order
//! (`p[0] x^n + p[1] x^(n-1) + … + p[n]`), so we follow that convention.

use crate::dtype::{ArraysD, DType};
use ndarray::{ArrayD, IxDyn};
use rustpython_vm::{PyResult, VirtualMachine};

/// Cast to F64 and unwrap; on the unreachable variant, return an empty array.
fn cast_f64_or_empty(a: &ArraysD) -> ArrayD<f64> {
    match a.cast(DType::F64) {
        ArraysD::F64(x) => x,
        _ => ArrayD::<f64>::zeros(IxDyn(&[0])),
    }
}

/// `numpy.polyval(p, x)` via Horner's method.
pub fn polyval(p: &ArraysD, x: &ArraysD, _vm: &VirtualMachine) -> PyResult<ArraysD> {
    let pf = cast_f64_or_empty(p);
    let xf = cast_f64_or_empty(x);
    let coeffs: Vec<f64> = pf.iter().copied().collect();
    let shape = xf.shape().to_vec();
    let data: Vec<f64> = xf
        .iter()
        .map(|&xv| {
            let mut acc = 0.0;
            for &c in coeffs.iter() {
                acc = acc * xv + c;
            }
            acc
        })
        .collect();
    Ok(ArraysD::F64(
        ArrayD::from_shape_vec(IxDyn(&shape), data).unwrap_or_default(),
    ))
}

/// `numpy.roots(p)` — eigenvalues of the companion matrix, computed via
/// faer's general eigendecomposition. Returns a complex128 array; purely
/// real roots come back with imaginary parts exactly zero.
pub fn roots(p: &ArraysD, vm: &VirtualMachine) -> PyResult<ArraysD> {
    let pf = cast_f64_or_empty(p);
    let mut coeffs: Vec<f64> = pf.iter().copied().collect();
    while coeffs.len() > 1 && coeffs[0] == 0.0 {
        coeffs.remove(0);
    }
    if coeffs.len() < 2 {
        return Ok(ArraysD::C128(
            ArrayD::from_shape_vec(IxDyn(&[0]), Vec::<num_complex::Complex<f64>>::new())
                .unwrap_or_default(),
        ));
    }
    let n = coeffs.len() - 1;
    let lead = coeffs[0];
    let normed: Vec<f64> = coeffs.iter().skip(1).map(|c| c / lead).collect();

    let companion = faer::Mat::<f64>::from_fn(n, n, |i, j| {
        if j == n - 1 {
            -normed[n - 1 - i]
        } else if i == j + 1 {
            1.0
        } else {
            0.0
        }
    });

    let eigen = companion
        .eigen()
        .map_err(|e| vm.new_value_error(format!("roots: eigendecomposition failed: {e:?}")))?;
    let s = eigen.S().column_vector();
    let mut found: Vec<num_complex::Complex<f64>> = (0..n).map(|i| s[i]).collect();

    found.sort_by(|x, y| {
        y.re.partial_cmp(&x.re)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| y.im.partial_cmp(&x.im).unwrap_or(std::cmp::Ordering::Equal))
    });
    Ok(ArraysD::C128(
        ArrayD::from_shape_vec(IxDyn(&[n]), found).unwrap_or_default(),
    ))
}

/// `numpy.polyfit(x, y, deg)` — least-squares fit of polynomial of given
/// degree, returned in numpy's descending-power order.
pub fn polyfit(x: &ArraysD, y: &ArraysD, deg: usize, vm: &VirtualMachine) -> PyResult<ArraysD> {
    let xv = cast_f64_or_empty(x);
    let yv = cast_f64_or_empty(y);
    if xv.len() != yv.len() {
        return Err(vm.new_value_error(format!(
            "polyfit: x has {} samples, y has {}",
            xv.len(),
            yv.len()
        )));
    }
    let m = xv.len();
    let n = deg + 1;
    // Vandermonde: V[i, j] = x[i]^(n-1-j)   (so column 0 holds the highest
    // power, matching numpy's polyfit convention).
    let mut vand = ndarray::Array2::<f64>::zeros((m, n));
    let xs: Vec<f64> = xv.iter().copied().collect();
    for i in 0..m {
        let mut p = 1.0;
        for j in (0..n).rev() {
            vand[(i, j)] = p;
            p *= xs[i];
        }
    }
    let a = ArraysD::F64(vand.into_dyn());
    let b = ArraysD::F64(yv);
    crate::linalg_extra::lstsq(&a, &b, vm)
}
