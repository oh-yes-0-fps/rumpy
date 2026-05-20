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

/// `numpy.roots(p)` — eigenvalues of the companion matrix via QR with
/// double-shift-style deflation. Returns a complex128 array; for purely-real
/// polynomials with real roots, the imaginary parts are exactly zero.
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
    let mut a = ndarray::Array2::<f64>::zeros((n, n));
    for i in 0..(n - 1) {
        a[(i + 1, i)] = 1.0;
    }
    for i in 0..n {
        a[(i, n - 1)] = -normed[n - 1 - i];
    }
    let mut iters = 0;
    let max_iters = 2000;
    let mut size = n;
    let mut found: Vec<num_complex::Complex<f64>> = Vec::with_capacity(n);
    while size > 1 && iters < max_iters {
        iters += 1;
        // Wilkinson-style shift from the bottom-right 2×2 block.
        let p00 = a[(size - 2, size - 2)];
        let p01 = a[(size - 2, size - 1)];
        let p10 = a[(size - 1, size - 2)];
        let p11 = a[(size - 1, size - 1)];
        let tr = p00 + p11;
        let det = p00 * p11 - p01 * p10;
        let disc = tr * tr - 4.0 * det;
        let shift = if disc >= 0.0 {
            let s = disc.sqrt();
            let l1 = 0.5 * (tr + s);
            let l2 = 0.5 * (tr - s);
            if (l1 - p11).abs() < (l2 - p11).abs() {
                l1
            } else {
                l2
            }
        } else {
            0.5 * tr
        };
        for i in 0..size {
            a[(i, i)] -= shift;
        }
        givens_qr_step(&mut a, size);
        for i in 0..size {
            a[(i, i)] += shift;
        }

        // Two deflation conditions: a single real root falls off the bottom,
        // or a 2×2 trailing block with complex eigenvalues falls off.
        let tol_factor = 1e-10;
        if size >= 1
            && (size == 1
                || a[(size - 1, size - 2)].abs()
                    < tol_factor
                        * (a[(size - 1, size - 1)].abs()
                            + a[(size - 2, size - 2)].abs()
                            + 1e-300))
        {
            found.push(num_complex::Complex::new(a[(size - 1, size - 1)], 0.0));
            size -= 1;
            continue;
        }
        if size >= 2
            && (size == 2
                || a[(size - 2, size - 3)].abs()
                    < tol_factor
                        * (a[(size - 2, size - 2)].abs()
                            + a[(size - 3, size - 3)].abs()
                            + 1e-300))
        {
            // Pull eigenvalues out of the bottom 2×2 block.
            let (e1, e2) = block_eigs(&a, size - 2);
            found.push(e1);
            found.push(e2);
            size -= 2;
            continue;
        }
    }
    if size == 1 {
        found.push(num_complex::Complex::new(a[(0, 0)], 0.0));
    } else if size == 2 {
        let (e1, e2) = block_eigs(&a, 0);
        found.push(e1);
        found.push(e2);
    }
    if found.len() != n {
        return Err(vm.new_value_error("roots: QR algorithm did not converge".to_string()));
    }
    // numpy returns roots in arbitrary order. We sort by descending real
    // part, then descending imaginary part for stable output.
    found.sort_by(|x, y| {
        y.re.partial_cmp(&x.re)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| y.im.partial_cmp(&x.im).unwrap_or(std::cmp::Ordering::Equal))
    });
    Ok(ArraysD::C128(
        ArrayD::from_shape_vec(IxDyn(&[n]), found).unwrap_or_default(),
    ))
}

fn block_eigs(
    a: &ndarray::Array2<f64>,
    i: usize,
) -> (num_complex::Complex<f64>, num_complex::Complex<f64>) {
    let p00 = a[(i, i)];
    let p01 = a[(i, i + 1)];
    let p10 = a[(i + 1, i)];
    let p11 = a[(i + 1, i + 1)];
    let tr = p00 + p11;
    let det = p00 * p11 - p01 * p10;
    let disc = tr * tr - 4.0 * det;
    if disc >= 0.0 {
        let s = disc.sqrt();
        (
            num_complex::Complex::new(0.5 * (tr + s), 0.0),
            num_complex::Complex::new(0.5 * (tr - s), 0.0),
        )
    } else {
        let s = (-disc).sqrt();
        (
            num_complex::Complex::new(0.5 * tr, 0.5 * s),
            num_complex::Complex::new(0.5 * tr, -0.5 * s),
        )
    }
}

fn givens_qr_step(a: &mut ndarray::Array2<f64>, size: usize) {
    // Apply Givens rotations to chase the sub-diagonal toward zero.
    let mut cs = Vec::<(f64, f64)>::with_capacity(size);
    // Forward: QR factorize.
    for i in 0..(size - 1) {
        let x = a[(i, i)];
        let y = a[(i + 1, i)];
        let r = (x * x + y * y).sqrt();
        if r == 0.0 {
            cs.push((1.0, 0.0));
            continue;
        }
        let c = x / r;
        let s = y / r;
        // Apply to rows i and i+1 from column i onward.
        for j in i..size {
            let xj = a[(i, j)];
            let yj = a[(i + 1, j)];
            a[(i, j)] = c * xj + s * yj;
            a[(i + 1, j)] = -s * xj + c * yj;
        }
        cs.push((c, s));
    }
    // Backward: multiply by Rᵀ to form RQ.
    for i in 0..(size - 1) {
        let (c, s) = cs[i];
        for k in 0..size {
            let xj = a[(k, i)];
            let yj = a[(k, i + 1)];
            a[(k, i)] = c * xj + s * yj;
            a[(k, i + 1)] = -s * xj + c * yj;
        }
    }
}

/// `numpy.polyfit(x, y, deg)` — least-squares fit of polynomial of given
/// degree, returned in numpy's descending-power order.
pub fn polyfit(
    x: &ArraysD,
    y: &ArraysD,
    deg: usize,
    vm: &VirtualMachine,
) -> PyResult<ArraysD> {
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
