//! `numpy.linalg` algorithms: norm, det, inv, solve, matrix_rank.
//! Pure-Rust implementations on f64 (input promoted to f64 first).

use crate::dtype::{ArraysD, DType};
use ndarray::{ArrayD, Axis, IxDyn};
use rustpython_vm::{PyResult, VirtualMachine};

fn as_f64_2d(
    a: &ArraysD,
    vm: &VirtualMachine,
) -> PyResult<ndarray::Array2<f64>> {
    if a.ndim() != 2 {
        return Err(vm.new_value_error(format!(
            "expected 2-D array, got {}-D",
            a.ndim()
        )));
    }
    use crate::dtype::CoerceArray as _CA;
    let arr = a.coerce::<f64>();
    arr.into_dimensionality::<ndarray::Ix2>()
        .map_err(|e| crate::internal::internal(vm, format!("as_f64_2d: {e}")))
}

/// The `ord` argument to `np.linalg.norm`. `None` means the default
/// (2-norm for 1-D vectors, Frobenius for 2-D matrices).
#[derive(Clone, Copy, Debug)]
pub enum NormOrd {
    /// `np.inf`
    PosInf,
    /// `-np.inf`
    NegInf,
    /// Numeric (1, 2, p, …). Special values: `0.0` → count of nonzeros for
    /// vector norms; only sensible as a vector norm.
    Num(f64),
    /// `"fro"` — Frobenius (matrix only).
    Fro,
    /// `"nuc"` — nuclear / trace norm (matrix only, needs SVD).
    Nuc,
}

/// `np.linalg.norm`.
///
/// Supported:
///   * 1-D vectors: ord=None / 1 / 2 / p (float) / inf / -inf / 0.
///   * 2-D matrices: ord=None / 'fro' / 1 / -1 / inf / -inf.
///   * `axis=int` reduces along that single axis as a vector norm.
///   * `axis=(i,j)` reduces along those two axes as a matrix norm.
///   * `keepdims=True` keeps reduced axes as length-1.
///
/// Not yet supported: ord=2 / -2 / 'nuc' on 2-D (need SVD).
pub fn norm(
    a: &ArraysD,
    ord: Option<NormOrd>,
    axis: Option<NormAxis>,
    keepdims: bool,
    vm: &VirtualMachine,
) -> PyResult<ArraysD> {
    use crate::dtype::CoerceArray as _CA;
    let arr = a.coerce::<f64>();
    // Resolve which axes participate in the reduction. `None` axis means
    // "reduce the whole array as a flat vector" (matching numpy).
    let nd = arr.ndim();
    let axes: AxesSpec = match axis {
        None => {
            // numpy: a 2-D input without an explicit axis becomes a matrix
            // norm for ord in {None, 'fro', 'nuc', 1, -1, 2, -2, inf, -inf}.
            // Everything else (e.g., p=3) is treated as a flat vector norm.
            let is_matrix_ord = match ord {
                None | Some(NormOrd::Fro) | Some(NormOrd::Nuc) => true,
                Some(NormOrd::PosInf) | Some(NormOrd::NegInf) => true,
                Some(NormOrd::Num(p)) => {
                    let ap = p.abs();
                    (ap - 1.0).abs() < f64::EPSILON || (ap - 2.0).abs() < f64::EPSILON
                }
            };
            if nd == 2 && is_matrix_ord {
                AxesSpec::Matrix(0, 1)
            } else {
                AxesSpec::All
            }
        }
        Some(NormAxis::Single(ax)) => AxesSpec::Vector(normalize_axis(ax, nd, vm)?),
        Some(NormAxis::Pair(i, j)) => {
            let ni = normalize_axis(i, nd, vm)?;
            let nj = normalize_axis(j, nd, vm)?;
            if ni == nj {
                return Err(vm.new_value_error(
                    "norm: duplicate axes for matrix norm".to_string(),
                ));
            }
            AxesSpec::Matrix(ni, nj)
        }
    };
    let result = match axes {
        AxesSpec::All => {
            // Flatten and compute a scalar vector norm.
            let flat: Vec<f64> = arr.iter().copied().collect();
            ArrayD::from_elem(IxDyn(&[]), vector_norm(&flat, ord, vm)?)
        }
        AxesSpec::Vector(ax) => {
            let out: ArrayD<f64> = arr.map_axis(Axis(ax), |row| {
                let v: Vec<f64> = row.iter().copied().collect();
                vector_norm(&v, ord, vm).unwrap_or(f64::NAN)
            });
            out
        }
        AxesSpec::Matrix(i, j) => matrix_norm_over(&arr, i, j, ord, vm)?,
    };
    let final_arr = if keepdims {
        let mut shape = result.shape().to_vec();
        // Re-insert collapsed axes (matching numpy's keepdims layout).
        let original_shape = arr.shape().to_vec();
        match axes {
            AxesSpec::All => {
                shape = vec![1; original_shape.len()];
            }
            AxesSpec::Vector(ax) => {
                shape.insert(ax, 1);
            }
            AxesSpec::Matrix(i, j) => {
                let (lo, hi) = if i < j { (i, j) } else { (j, i) };
                shape.insert(lo, 1);
                shape.insert(hi, 1);
            }
        }
        result
            .into_shape_with_order(IxDyn(&shape))
            .map_err(|e| crate::internal::internal(vm, format!("norm keepdims: {e}")))?
    } else {
        result
    };
    Ok(ArraysD::F64(final_arr))
}

/// Vector norm of a flat slice.
fn vector_norm(v: &[f64], ord: Option<NormOrd>, vm: &VirtualMachine) -> PyResult<f64> {
    if v.is_empty() {
        return Ok(0.0);
    }
    // Resolve the effective order: None / Fro / Num(2) all default to 2-norm.
    let p_opt: Option<f64> = match ord {
        None | Some(NormOrd::Fro) => None,
        Some(NormOrd::PosInf) => return Ok(v.iter().fold(0.0_f64, |a, &x| a.max(x.abs()))),
        Some(NormOrd::NegInf) => return Ok(v.iter().fold(f64::INFINITY, |a, &x| a.min(x.abs()))),
        Some(NormOrd::Nuc) => {
            return Err(vm.new_value_error(
                "norm: 'nuc' ord requires a 2-D matrix".to_string(),
            ));
        }
        Some(NormOrd::Num(p)) => Some(p),
    };
    let p = match p_opt {
        None => 2.0,
        Some(p) => p,
    };
    if p == 0.0 {
        return Ok(v.iter().filter(|&&x| x != 0.0).count() as f64);
    }
    if p == 1.0 {
        return Ok(v.iter().map(|x| x.abs()).sum());
    }
    if p == 2.0 {
        let s: f64 = v.iter().map(|x| x * x).sum();
        return Ok(s.sqrt());
    }
    if p.is_infinite() {
        return Ok(if p > 0.0 {
            v.iter().fold(0.0_f64, |a, &x| a.max(x.abs()))
        } else {
            v.iter().fold(f64::INFINITY, |a, &x| a.min(x.abs()))
        });
    }
    let s: f64 = v.iter().map(|x| x.abs().powf(p)).sum();
    Ok(s.powf(1.0 / p))
}

/// Matrix norm over the two specified axes of a higher-rank array.
fn matrix_norm_over(
    arr: &ArrayD<f64>,
    i: usize,
    j: usize,
    ord: Option<NormOrd>,
    vm: &VirtualMachine,
) -> PyResult<ArrayD<f64>> {
    let nd = arr.ndim();
    if nd < 2 {
        return Err(vm.new_value_error(
            "norm: matrix axes require at least a 2-D array".to_string(),
        ));
    }
    // Materialize the result for each combination of the non-{i,j} axes.
    let outer_axes: Vec<usize> = (0..nd).filter(|&k| k != i && k != j).collect();
    let outer_shape: Vec<usize> = outer_axes.iter().map(|&k| arr.shape()[k]).collect();
    let outer_size: usize = outer_shape.iter().product::<usize>().max(1);
    let mut out = Vec::with_capacity(outer_size);
    let mut idx = vec![0usize; nd];
    let dim_i = arr.shape()[i];
    let dim_j = arr.shape()[j];
    // Iterate outer index in C-order. For each outer position, walk the
    // (i, j) 2-D slice into a Vec and call matrix_norm_2d.
    let mut counter = vec![0usize; outer_axes.len()];
    loop {
        for (k, &ax) in outer_axes.iter().enumerate() {
            idx[ax] = counter[k];
        }
        let mut block = Vec::with_capacity(dim_i * dim_j);
        for ii in 0..dim_i {
            for jj in 0..dim_j {
                idx[i] = ii;
                idx[j] = jj;
                block.push(arr[IxDyn(&idx)]);
            }
        }
        out.push(matrix_norm_2d(&block, dim_i, dim_j, ord, vm)?);
        // Advance counter.
        if counter.is_empty() {
            break;
        }
        let mut k = counter.len();
        loop {
            if k == 0 {
                // overflow → done
                let out_arr = ArrayD::from_shape_vec(IxDyn(&outer_shape), out)
                    .map_err(|e| crate::internal::internal(vm, format!("matrix norm: {e}")))?;
                return Ok(out_arr);
            }
            k -= 1;
            counter[k] += 1;
            if counter[k] < outer_shape[k] {
                break;
            }
            counter[k] = 0;
        }
    }
    let out_arr = if outer_shape.is_empty() {
        ArrayD::from_elem(IxDyn(&[]), out[0])
    } else {
        ArrayD::from_shape_vec(IxDyn(&outer_shape), out)
            .map_err(|e| crate::internal::internal(vm, format!("matrix norm: {e}")))?
    };
    Ok(out_arr)
}

/// 2-D matrix norm on a row-major flat block of shape `(rows, cols)`.
fn matrix_norm_2d(
    data: &[f64],
    rows: usize,
    cols: usize,
    ord: Option<NormOrd>,
    vm: &VirtualMachine,
) -> PyResult<f64> {
    let at = |r: usize, c: usize| data[r * cols + c];
    match ord {
        None | Some(NormOrd::Fro) => {
            // Frobenius.
            let s: f64 = data.iter().map(|x| x * x).sum();
            Ok(s.sqrt())
        }
        Some(NormOrd::Num(p)) if (p - 1.0).abs() < f64::EPSILON => {
            // Max column sum.
            let mut best = 0.0_f64;
            for c in 0..cols {
                let s: f64 = (0..rows).map(|r| at(r, c).abs()).sum();
                if s > best {
                    best = s;
                }
            }
            Ok(best)
        }
        Some(NormOrd::Num(p)) if (p + 1.0).abs() < f64::EPSILON => {
            // Min column sum.
            let mut best = f64::INFINITY;
            for c in 0..cols {
                let s: f64 = (0..rows).map(|r| at(r, c).abs()).sum();
                if s < best {
                    best = s;
                }
            }
            Ok(best)
        }
        Some(NormOrd::PosInf) => {
            // Max row sum.
            let mut best = 0.0_f64;
            for r in 0..rows {
                let s: f64 = (0..cols).map(|c| at(r, c).abs()).sum();
                if s > best {
                    best = s;
                }
            }
            Ok(best)
        }
        Some(NormOrd::NegInf) => {
            // Min row sum.
            let mut best = f64::INFINITY;
            for r in 0..rows {
                let s: f64 = (0..cols).map(|c| at(r, c).abs()).sum();
                if s < best {
                    best = s;
                }
            }
            Ok(best)
        }
        Some(NormOrd::Num(p)) if (p - 2.0).abs() < f64::EPSILON
            || (p + 2.0).abs() < f64::EPSILON =>
        {
            Err(vm.new_not_implemented_error(
                "matrix 2-norm (largest/smallest singular value) requires SVD; not yet implemented"
                    .to_string(),
            ))
        }
        Some(NormOrd::Nuc) => Err(vm.new_not_implemented_error(
            "matrix nuclear norm requires SVD; not yet implemented".to_string(),
        )),
        Some(NormOrd::Num(p)) => Err(vm.new_value_error(format!(
            "invalid matrix norm order: {p}"
        ))),
    }
}

#[derive(Clone, Copy, Debug)]
pub enum NormAxis {
    Single(isize),
    Pair(isize, isize),
}

#[derive(Clone, Copy)]
enum AxesSpec {
    All,
    Vector(usize),
    Matrix(usize, usize),
}

fn normalize_axis(ax: isize, nd: usize, vm: &VirtualMachine) -> PyResult<usize> {
    let nd_i = nd as isize;
    let real = if ax < 0 { ax + nd_i } else { ax };
    if real < 0 || real >= nd_i {
        return Err(vm.new_value_error(format!(
            "axis {ax} is out of bounds for array of dimension {nd}"
        )));
    }
    Ok(real as usize)
}

pub fn det(a: &ArraysD, vm: &VirtualMachine) -> PyResult<ArraysD> {
    let m = as_f64_2d(a, vm)?;
    let (r, c) = m.dim();
    if r != c {
        return Err(vm.new_value_error("det: matrix must be square".to_string()));
    }
    let n = r;
    // LU with partial pivoting.
    let mut a = m;
    let mut det = 1.0f64;
    for k in 0..n {
        // Pivot.
        let mut pivot = k;
        let mut best = a[(k, k)].abs();
        for i in (k + 1)..n {
            if a[(i, k)].abs() > best {
                best = a[(i, k)].abs();
                pivot = i;
            }
        }
        if best == 0.0 {
            return Ok(ArraysD::F64(ArrayD::from_elem(IxDyn(&[]), 0.0)));
        }
        if pivot != k {
            // swap rows
            for j in 0..n {
                let t = a[(k, j)];
                a[(k, j)] = a[(pivot, j)];
                a[(pivot, j)] = t;
            }
            det = -det;
        }
        det *= a[(k, k)];
        let pivot_val = a[(k, k)];
        for i in (k + 1)..n {
            let factor = a[(i, k)] / pivot_val;
            for j in k..n {
                a[(i, j)] -= factor * a[(k, j)];
            }
        }
    }
    Ok(ArraysD::F64(ArrayD::from_elem(IxDyn(&[]), det)))
}

pub fn inv(a: &ArraysD, vm: &VirtualMachine) -> PyResult<ArraysD> {
    let m = as_f64_2d(a, vm)?;
    let (r, c) = m.dim();
    if r != c {
        return Err(vm.new_value_error("inv: matrix must be square".to_string()));
    }
    let n = r;
    // Gauss-Jordan with partial pivoting.
    let mut aug = ndarray::Array2::<f64>::zeros((n, 2 * n));
    for i in 0..n {
        for j in 0..n {
            aug[(i, j)] = m[(i, j)];
        }
        aug[(i, n + i)] = 1.0;
    }
    for k in 0..n {
        let mut pivot = k;
        let mut best = aug[(k, k)].abs();
        for i in (k + 1)..n {
            if aug[(i, k)].abs() > best {
                best = aug[(i, k)].abs();
                pivot = i;
            }
        }
        if best == 0.0 {
            return Err(vm.new_value_error("inv: singular matrix".to_string()));
        }
        if pivot != k {
            for j in 0..2 * n {
                let t = aug[(k, j)];
                aug[(k, j)] = aug[(pivot, j)];
                aug[(pivot, j)] = t;
            }
        }
        let pv = aug[(k, k)];
        for j in 0..2 * n {
            aug[(k, j)] /= pv;
        }
        for i in 0..n {
            if i == k {
                continue;
            }
            let factor = aug[(i, k)];
            for j in 0..2 * n {
                aug[(i, j)] -= factor * aug[(k, j)];
            }
        }
    }
    let inv = aug.slice(ndarray::s![.., n..]).to_owned();
    Ok(ArraysD::F64(inv.into_dyn()))
}

pub fn solve(a: &ArraysD, b: &ArraysD, vm: &VirtualMachine) -> PyResult<ArraysD> {
    let m = as_f64_2d(a, vm)?;
    let inv_m = inv(a, vm)?;
    // Compute inv(A) · b.
    let _ = m;
    let result = crate::linalg::dot(&inv_m, &b.cast(DType::F64), vm)?;
    Ok(result)
}

pub fn matrix_rank(a: &ArraysD, vm: &VirtualMachine) -> PyResult<ArraysD> {
    let m = as_f64_2d(a, vm)?;
    let (r, c) = m.dim();
    let mut a = m;
    let tol = 1e-12;
    let mut rank = 0;
    let limit = r.min(c);
    let mut row = 0usize;
    for col in 0..c {
        if row >= limit {
            break;
        }
        // Find pivot in this column.
        let mut pivot = row;
        let mut best = a[(row, col)].abs();
        for i in (row + 1)..r {
            if a[(i, col)].abs() > best {
                best = a[(i, col)].abs();
                pivot = i;
            }
        }
        if best <= tol {
            continue;
        }
        if pivot != row {
            for j in 0..c {
                let t = a[(row, j)];
                a[(row, j)] = a[(pivot, j)];
                a[(pivot, j)] = t;
            }
        }
        let pv = a[(row, col)];
        for i in (row + 1)..r {
            let factor = a[(i, col)] / pv;
            for j in col..c {
                a[(i, j)] -= factor * a[(row, j)];
            }
        }
        row += 1;
        rank += 1;
    }
    Ok(ArraysD::I64(ArrayD::from_elem(IxDyn(&[]), rank as i64)))
}

/// `np.trace` — sum of the diagonal.
pub fn trace(a: &ArraysD, vm: &VirtualMachine) -> PyResult<ArraysD> {
    let m = as_f64_2d(a, vm)?;
    let n = m.dim().0.min(m.dim().1);
    let mut s = 0.0;
    for i in 0..n {
        s += m[(i, i)];
    }
    Ok(ArraysD::F64(ArrayD::from_elem(IxDyn(&[]), s)).cast(a.dtype()))
}

/// Cholesky decomposition `L · Lᵀ = A` for a symmetric positive-definite
/// matrix. Returns the lower triangular `L`.
pub fn cholesky(a: &ArraysD, vm: &VirtualMachine) -> PyResult<ArraysD> {
    let m = as_f64_2d(a, vm)?;
    let (r, c) = m.dim();
    if r != c {
        return Err(vm.new_value_error("cholesky: matrix must be square".to_string()));
    }
    let n = r;
    let mut l = ndarray::Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..=i {
            let mut sum = 0.0;
            for k in 0..j {
                sum += l[(i, k)] * l[(j, k)];
            }
            if i == j {
                let diag = m[(i, i)] - sum;
                if diag <= 0.0 {
                    return Err(
                        vm.new_value_error("cholesky: matrix is not positive-definite".to_string()),
                    );
                }
                l[(i, j)] = diag.sqrt();
            } else {
                l[(i, j)] = (m[(i, j)] - sum) / l[(j, j)];
            }
        }
    }
    Ok(ArraysD::F64(l.into_dyn()))
}

/// Output style for [`qr`]. Mirrors numpy's `mode` kwarg.
#[derive(Copy, Clone, Debug)]
pub enum QrMode {
    /// `Q (m×k)`, `R (k×n)` where `k = min(m, n)`. numpy default.
    Reduced,
    /// `Q (m×m)`, `R (m×n)`.
    Complete,
    /// Just `R (k×n)`.
    R,
}

impl QrMode {
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "reduced" => Some(QrMode::Reduced),
            "complete" => Some(QrMode::Complete),
            "r" => Some(QrMode::R),
            _ => None,
        }
    }
}

/// QR decomposition by classical Householder reflections.
///
/// * `mode = Reduced` (numpy default) — returns `(Q[:, :k], R[:k, :])`.
/// * `mode = Complete` — returns the full m×m Q and m×n R.
/// * `mode = R` — returns `(R[:k, :],)` (a one-element vec) — caller picks
///   the convention.
pub fn qr(a: &ArraysD, mode: QrMode, vm: &VirtualMachine) -> PyResult<(ArraysD, ArraysD)> {
    let m = as_f64_2d(a, vm)?;
    let (rows, cols) = m.dim();
    let mut r = m.clone();
    let mut q = ndarray::Array2::<f64>::eye(rows);
    let k = rows.min(cols);
    for col in 0..k {
        // Build Householder vector v.
        let mut alpha = 0.0;
        for i in col..rows {
            alpha += r[(i, col)] * r[(i, col)];
        }
        alpha = alpha.sqrt();
        if r[(col, col)] > 0.0 {
            alpha = -alpha;
        }
        if alpha == 0.0 {
            continue;
        }
        let mut v = vec![0.0; rows];
        v[col] = r[(col, col)] - alpha;
        for i in (col + 1)..rows {
            v[i] = r[(i, col)];
        }
        let norm2: f64 = v.iter().map(|x| x * x).sum();
        if norm2 == 0.0 {
            continue;
        }
        let beta = 2.0 / norm2;
        // Apply H = I - β v vᵀ to R from the left: R = H · R
        for j in col..cols {
            let mut dot = 0.0;
            for i in col..rows {
                dot += v[i] * r[(i, j)];
            }
            let f = beta * dot;
            for i in col..rows {
                r[(i, j)] -= f * v[i];
            }
        }
        // And Q from the right: Q = Q · Hᵀ = Q · H
        for i in 0..rows {
            let mut dot = 0.0;
            for j in col..rows {
                dot += q[(i, j)] * v[j];
            }
            let f = beta * dot;
            for j in col..rows {
                q[(i, j)] -= f * v[j];
            }
        }
    }
    let k = rows.min(cols);
    let (q_out, r_out) = match mode {
        QrMode::Complete => (q.clone(), r.clone()),
        QrMode::Reduced => {
            let q_red = q.slice(ndarray::s![.., ..k]).to_owned();
            let r_red = r.slice(ndarray::s![..k, ..]).to_owned();
            (q_red, r_red)
        }
        QrMode::R => {
            // Caller will discard the Q. We still need to produce something
            // for the tuple slot; use a zero-shape array so it's cheap.
            let r_red = r.slice(ndarray::s![..k, ..]).to_owned();
            (
                ndarray::Array2::<f64>::zeros((0, 0)),
                r_red,
            )
        }
    };
    Ok((ArraysD::F64(q_out.into_dyn()), ArraysD::F64(r_out.into_dyn())))
}

/// numpy.linalg.lstsq returns this tuple. `residuals` is empty when the
/// system is underdetermined or full rank with k=0 columns (matching numpy);
/// `s` is empty here because we don't have SVD.
pub struct LstsqResult {
    pub solution: ArraysD,
    pub residuals: ArraysD,
    pub rank: i64,
    pub singular: ArraysD,
}

/// `numpy.linalg.lstsq(A, b)` — least-squares solution via the normal
/// equations. The single-array `lstsq` (used internally) returns just the
/// solution; the tuple form `lstsq_full` returns `(x, residuals, rank, s)`.
pub fn lstsq_full(
    a: &ArraysD,
    b: &ArraysD,
    vm: &VirtualMachine,
) -> PyResult<LstsqResult> {
    let am = as_f64_2d(a, vm)?;
    let (m, _n) = am.dim();
    let solution = lstsq(a, b, vm)?;
    // residuals = sum((A x - b)^2, axis=0) per column. numpy emits a 1-D
    // array of length nrhs when m > rank, otherwise an empty array.
    let prod = crate::linalg::dot(a, &solution, vm)?;
    let diff = crate::ops::binary_op(&prod, b, vm, crate::ops::Sub)?;
    let sq = crate::ops::binary_op(&diff, &diff, vm, crate::ops::Mul)?;
    let residuals = sum_axis0(&sq);
    let rank = matrix_rank_f64(&am);
    let residuals = if m > rank as usize {
        residuals
    } else {
        // Empty residuals when the system is square or underdetermined,
        // matching numpy.
        crate::create::zeros(&[0], DType::F64)
    };
    let singular = crate::create::zeros(&[0], DType::F64);
    Ok(LstsqResult {
        solution,
        residuals,
        rank: rank as i64,
        singular,
    })
}

fn sum_axis0(a: &ArraysD) -> ArraysD {
    match a {
        ArraysD::F64(arr) => {
            if arr.ndim() <= 1 {
                let s: f64 = arr.iter().copied().sum();
                ArraysD::F64(ndarray::ArrayD::from_elem(IxDyn(&[]), s))
            } else {
                let summed = arr.sum_axis(ndarray::Axis(0));
                ArraysD::F64(summed)
            }
        }
        // Caller always passes F64; safe fallback.
        _ => ArraysD::F64(ndarray::ArrayD::from_elem(IxDyn(&[]), 0.0)),
    }
}

fn matrix_rank_f64(m: &ndarray::Array2<f64>) -> usize {
    let (r, c) = m.dim();
    let mut a = m.clone();
    let tol = 1e-12;
    let mut row = 0usize;
    let limit = r.min(c);
    for col in 0..c {
        if row >= limit {
            break;
        }
        let mut pivot = row;
        let mut best = a[(row, col)].abs();
        for i in (row + 1)..r {
            if a[(i, col)].abs() > best {
                best = a[(i, col)].abs();
                pivot = i;
            }
        }
        if best <= tol {
            continue;
        }
        if pivot != row {
            for j in 0..c {
                let t = a[(row, j)];
                a[(row, j)] = a[(pivot, j)];
                a[(pivot, j)] = t;
            }
        }
        let pv = a[(row, col)];
        for i in (row + 1)..r {
            let factor = a[(i, col)] / pv;
            for j in col..c {
                a[(i, j)] -= factor * a[(row, j)];
            }
        }
        row += 1;
    }
    row
}

/// Bare-bones lstsq returning just the solution. Used by `pinv`,
/// `polyfit`, and as the workhorse for `lstsq_full`.
pub fn lstsq(a: &ArraysD, b: &ArraysD, vm: &VirtualMachine) -> PyResult<ArraysD> {
    let am = as_f64_2d(a, vm)?;
    let (m, n) = am.dim();
    // Reshape b to (m, k) — accept 1-D b (k=1) or 2-D.
    let bf = b.cast(DType::F64);
    let (b2, was_1d) = match bf {
        ArraysD::F64(arr) => match arr.ndim() {
            1 => {
                let v: Vec<f64> = arr.iter().copied().collect();
                let v_len = v.len();
                (
                    ndarray::Array2::from_shape_vec((v_len, 1), v).unwrap_or_default(),
                    true,
                )
            }
            2 => (
                arr.into_dimensionality::<ndarray::Ix2>()
                    .map_err(|e| crate::internal::internal(vm, format!("lstsq: {e}")))?,
                false,
            ),
            _ => {
                return Err(vm.new_value_error("lstsq: b must be 1-D or 2-D".to_string()));
            }
        },
        _ => return Err(crate::internal::internal(vm, "lstsq: cast to F64 failed")),
    };
    if b2.dim().0 != m {
        return Err(vm.new_value_error(format!(
            "lstsq: A rows ({}) != b rows ({})",
            m,
            b2.dim().0
        )));
    }
    // Form Aᵀ A and Aᵀ b.
    let at = am.t();
    let ata = at.dot(&am);
    let atb = at.dot(&b2);
    // Solve ata · x = atb by Gauss-Jordan on a (n × (n+k)) augmented matrix.
    let k = atb.dim().1;
    let mut aug = ndarray::Array2::<f64>::zeros((n, n + k));
    for i in 0..n {
        for j in 0..n {
            aug[(i, j)] = ata[(i, j)];
        }
        for j in 0..k {
            aug[(i, n + j)] = atb[(i, j)];
        }
    }
    for col in 0..n {
        let mut pivot = col;
        let mut best = aug[(col, col)].abs();
        for i in (col + 1)..n {
            if aug[(i, col)].abs() > best {
                best = aug[(i, col)].abs();
                pivot = i;
            }
        }
        if best < 1e-14 {
            return Err(
                vm.new_value_error("lstsq: AᵀA is singular (rank-deficient input)".to_string()),
            );
        }
        if pivot != col {
            for j in 0..n + k {
                let t = aug[(col, j)];
                aug[(col, j)] = aug[(pivot, j)];
                aug[(pivot, j)] = t;
            }
        }
        let pv = aug[(col, col)];
        for j in 0..n + k {
            aug[(col, j)] /= pv;
        }
        for i in 0..n {
            if i == col {
                continue;
            }
            let factor = aug[(i, col)];
            for j in 0..n + k {
                aug[(i, j)] -= factor * aug[(col, j)];
            }
        }
    }
    let x = aug.slice(ndarray::s![.., n..]).to_owned();
    Ok(if was_1d {
        // Return 1-D when b was 1-D.
        let v: Vec<f64> = x.iter().copied().collect();
        ArraysD::F64(ArrayD::from_shape_vec(IxDyn(&[n]), v).unwrap_or_default())
    } else {
        ArraysD::F64(x.into_dyn())
    })
}

/// `numpy.linalg.pinv(A)` — Moore-Penrose pseudoinverse via the normal
/// equations: `pinv(A) = (Aᵀ A)⁻¹ Aᵀ` for full column-rank A.
pub fn pinv(a: &ArraysD, vm: &VirtualMachine) -> PyResult<ArraysD> {
    let am = as_f64_2d(a, vm)?;
    let (m, n) = am.dim();
    // Build A · pinv(A) = I_m  →  pinv(A) = lstsq(A, I_m).
    let eye = ndarray::Array2::<f64>::eye(m);
    let eye_arr =
        ArraysD::F64(eye.into_dyn());
    let result = lstsq(a, &eye_arr, vm)?;
    let _ = (m, n);
    Ok(result)
}

/// `numpy.linalg.eigvalsh(A)` — eigenvalues of a symmetric/Hermitian real
/// matrix via Jacobi rotations. Returns the eigenvalues sorted ascending.
pub fn eigvalsh(a: &ArraysD, vm: &VirtualMachine) -> PyResult<ArraysD> {
    let am = as_f64_2d(a, vm)?;
    let (r, c) = am.dim();
    if r != c {
        return Err(vm.new_value_error("eigvalsh: matrix must be square".to_string()));
    }
    let n = r;
    let mut a = am;
    // Symmetrize to be safe (in case of small numerical asymmetry).
    for i in 0..n {
        for j in (i + 1)..n {
            let avg = 0.5 * (a[(i, j)] + a[(j, i)]);
            a[(i, j)] = avg;
            a[(j, i)] = avg;
        }
    }
    let tol = 1e-12;
    let max_sweeps = 100;
    for _ in 0..max_sweeps {
        // Find the largest off-diagonal element.
        let mut pi = 0usize;
        let mut pj = 1usize;
        let mut best = 0.0;
        for i in 0..n {
            for j in (i + 1)..n {
                if a[(i, j)].abs() > best {
                    best = a[(i, j)].abs();
                    pi = i;
                    pj = j;
                }
            }
        }
        if best < tol {
            break;
        }
        let app = a[(pi, pi)];
        let aqq = a[(pj, pj)];
        let apq = a[(pi, pj)];
        let theta = (aqq - app) / (2.0 * apq);
        let t = if theta >= 0.0 {
            1.0 / (theta + (1.0 + theta * theta).sqrt())
        } else {
            1.0 / (theta - (1.0 + theta * theta).sqrt())
        };
        let cos = 1.0 / (1.0 + t * t).sqrt();
        let sin = t * cos;
        // Apply the rotation to rows/cols pi, pj.
        for k in 0..n {
            if k == pi || k == pj {
                continue;
            }
            let aki = a[(k, pi)];
            let akj = a[(k, pj)];
            a[(k, pi)] = cos * aki - sin * akj;
            a[(pi, k)] = a[(k, pi)];
            a[(k, pj)] = sin * aki + cos * akj;
            a[(pj, k)] = a[(k, pj)];
        }
        a[(pi, pi)] = app - t * apq;
        a[(pj, pj)] = aqq + t * apq;
        a[(pi, pj)] = 0.0;
        a[(pj, pi)] = 0.0;
    }
    let mut eigs: Vec<f64> = (0..n).map(|i| a[(i, i)]).collect();
    eigs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    Ok(ArraysD::F64(
        ArrayD::from_shape_vec(IxDyn(&[n]), eigs).unwrap_or_default(),
    ))
}

/// `np.linalg.eigh(A)` for a symmetric matrix. Returns (eigenvalues, eigenvectors)
/// where eigenvectors are columns of the returned matrix.
pub fn eigh(a: &ArraysD, vm: &VirtualMachine) -> PyResult<(ArraysD, ArraysD)> {
    let am = as_f64_2d(a, vm)?;
    let (r, c) = am.dim();
    if r != c {
        return Err(vm.new_value_error("eigh: matrix must be square".to_string()));
    }
    let n = r;
    let mut a = am;
    // Symmetrize.
    for i in 0..n {
        for j in (i + 1)..n {
            let avg = 0.5 * (a[(i, j)] + a[(j, i)]);
            a[(i, j)] = avg;
            a[(j, i)] = avg;
        }
    }
    // Accumulator for the rotation matrix V.
    let mut v = ndarray::Array2::<f64>::eye(n);
    let tol = 1e-12;
    let max_sweeps = 100;
    for _ in 0..max_sweeps {
        let mut pi = 0usize;
        let mut pj = 1usize;
        let mut best = 0.0;
        for i in 0..n {
            for j in (i + 1)..n {
                if a[(i, j)].abs() > best {
                    best = a[(i, j)].abs();
                    pi = i;
                    pj = j;
                }
            }
        }
        if best < tol {
            break;
        }
        let app = a[(pi, pi)];
        let aqq = a[(pj, pj)];
        let apq = a[(pi, pj)];
        let theta = (aqq - app) / (2.0 * apq);
        let t = if theta >= 0.0 {
            1.0 / (theta + (1.0 + theta * theta).sqrt())
        } else {
            1.0 / (theta - (1.0 + theta * theta).sqrt())
        };
        let cos = 1.0 / (1.0 + t * t).sqrt();
        let sin = t * cos;
        for k in 0..n {
            if k == pi || k == pj {
                continue;
            }
            let aki = a[(k, pi)];
            let akj = a[(k, pj)];
            a[(k, pi)] = cos * aki - sin * akj;
            a[(pi, k)] = a[(k, pi)];
            a[(k, pj)] = sin * aki + cos * akj;
            a[(pj, k)] = a[(k, pj)];
        }
        a[(pi, pi)] = app - t * apq;
        a[(pj, pj)] = aqq + t * apq;
        a[(pi, pj)] = 0.0;
        a[(pj, pi)] = 0.0;
        // Update V columns.
        for k in 0..n {
            let vki = v[(k, pi)];
            let vkj = v[(k, pj)];
            v[(k, pi)] = cos * vki - sin * vkj;
            v[(k, pj)] = sin * vki + cos * vkj;
        }
    }
    // Sort eigenvalues ascending and reorder eigenvectors.
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&i, &j| a[(i, i)].partial_cmp(&a[(j, j)]).unwrap_or(std::cmp::Ordering::Equal));
    let eigs: Vec<f64> = order.iter().map(|&i| a[(i, i)]).collect();
    let mut vec_out = ndarray::Array2::<f64>::zeros((n, n));
    for (new_j, &old_j) in order.iter().enumerate() {
        for k in 0..n {
            vec_out[(k, new_j)] = v[(k, old_j)];
        }
    }
    Ok((
        ArraysD::F64(ArrayD::from_shape_vec(IxDyn(&[n]), eigs).unwrap_or_default()),
        ArraysD::F64(vec_out.into_dyn()),
    ))
}

/// `np.linalg.eig(A)` — general eigendecomposition. Currently we only support
/// symmetric A (delegated to eigh). Non-symmetric matrices raise NotImplementedError.
pub fn eig(a: &ArraysD, vm: &VirtualMachine) -> PyResult<(ArraysD, ArraysD)> {
    let am = as_f64_2d(a, vm)?;
    let (r, c) = am.dim();
    if r != c {
        return Err(vm.new_value_error("eig: matrix must be square".to_string()));
    }
    // Heuristic: if symmetric within tolerance, use eigh.
    let mut max_asym: f64 = 0.0;
    for i in 0..r {
        for j in 0..c {
            let d = (am[(i, j)] - am[(j, i)]).abs();
            if d > max_asym {
                max_asym = d;
            }
        }
    }
    if max_asym > 1e-10 {
        return Err(vm.new_not_implemented_error(
            "linalg.eig: non-symmetric matrices not yet supported (would need complex output)"
                .to_string(),
        ));
    }
    eigh(a, vm)
}

/// `np.linalg.eigvals(A)` — eigenvalues only (general matrix). See `eig`.
pub fn eigvals(a: &ArraysD, vm: &VirtualMachine) -> PyResult<ArraysD> {
    let (vals, _) = eig(a, vm)?;
    Ok(vals)
}

/// `np.linalg.slogdet(A)` — (sign, log(|det|)). Computed via LU.
pub fn slogdet(a: &ArraysD, vm: &VirtualMachine) -> PyResult<(ArraysD, ArraysD)> {
    let m = as_f64_2d(a, vm)?;
    let (r, c) = m.dim();
    if r != c {
        return Err(vm.new_value_error("slogdet: matrix must be square".to_string()));
    }
    let n = r;
    let mut a = m;
    let mut sign = 1.0f64;
    let mut logabs = 0.0f64;
    for k in 0..n {
        let mut pivot = k;
        let mut best = a[(k, k)].abs();
        for i in (k + 1)..n {
            if a[(i, k)].abs() > best {
                best = a[(i, k)].abs();
                pivot = i;
            }
        }
        if best == 0.0 {
            // singular
            return Ok((
                ArraysD::F64(ArrayD::from_elem(IxDyn(&[]), 0.0)),
                ArraysD::F64(ArrayD::from_elem(IxDyn(&[]), f64::NEG_INFINITY)),
            ));
        }
        if pivot != k {
            for j in 0..n {
                let t = a[(k, j)];
                a[(k, j)] = a[(pivot, j)];
                a[(pivot, j)] = t;
            }
            sign = -sign;
        }
        let pv = a[(k, k)];
        if pv < 0.0 {
            sign = -sign;
        }
        logabs += pv.abs().ln();
        for i in (k + 1)..n {
            let factor = a[(i, k)] / pv;
            for j in k..n {
                a[(i, j)] -= factor * a[(k, j)];
            }
        }
    }
    Ok((
        ArraysD::F64(ArrayD::from_elem(IxDyn(&[]), sign)),
        ArraysD::F64(ArrayD::from_elem(IxDyn(&[]), logabs)),
    ))
}

/// `np.linalg.matrix_power(A, n)` — A^n via repeated squaring. For n < 0,
/// computes inv(A)^|n|. For n == 0 returns the identity.
pub fn matrix_power(
    a: &ArraysD,
    n: isize,
    vm: &VirtualMachine,
) -> PyResult<ArraysD> {
    let m = as_f64_2d(a, vm)?;
    let (r, c) = m.dim();
    if r != c {
        return Err(vm.new_value_error("matrix_power: matrix must be square".to_string()));
    }
    let size = r;
    if n == 0 {
        return Ok(ArraysD::F64(
            ndarray::Array2::<f64>::eye(size).into_dyn(),
        ));
    }
    let base = if n < 0 {
        // Compute inverse first.
        let inv_arr = inv(a, vm)?;
        match inv_arr {
            ArraysD::F64(x) => x.into_dimensionality::<ndarray::Ix2>().map_err(|e| {
                crate::internal::internal(vm, format!("matrix_power inv: {e}"))
            })?,
            _ => return Err(crate::internal::internal(vm, "matrix_power: inv non-f64")),
        }
    } else {
        m
    };
    let mut exp = n.unsigned_abs() as u64;
    let mut acc = ndarray::Array2::<f64>::eye(size);
    let mut cur = base;
    while exp > 0 {
        if exp & 1 == 1 {
            acc = acc.dot(&cur);
        }
        exp >>= 1;
        if exp > 0 {
            cur = cur.dot(&cur);
        }
    }
    Ok(ArraysD::F64(acc.into_dyn()))
}

/// `np.linalg.svd(A, full_matrices=True)` — singular value decomposition.
/// Returns (U, Σ, V^H) where A = U Σ V^H. Computed via eigh(A^T A).
pub fn svd(
    a: &ArraysD,
    full_matrices: bool,
    vm: &VirtualMachine,
) -> PyResult<(ArraysD, ArraysD, ArraysD)> {
    let am = as_f64_2d(a, vm)?;
    let (m, n) = am.dim();
    // Compute A^T A (n×n, symmetric PSD).
    let at = am.t();
    let ata = at.dot(&am); // n×n
    let ata_arr = ArraysD::F64(ata.into_dyn());
    let (eigs, v_mat) = eigh(&ata_arr, vm)?;
    // eigh returns eigenvalues ascending; reverse to descending (numpy svd order).
    use crate::dtype::CoerceArray;
    let eig_vec: Vec<f64> = eigs.coerce::<f64>().iter().copied().collect();
    let mut idx: Vec<usize> = (0..eig_vec.len()).collect();
    idx.sort_by(|&i, &j| eig_vec[j].partial_cmp(&eig_vec[i]).unwrap_or(std::cmp::Ordering::Equal));
    let sigma: Vec<f64> = idx
        .iter()
        .map(|&i| eig_vec[i].max(0.0).sqrt())
        .collect();
    // Reorder V columns according to idx.
    let v_owned: ndarray::Array2<f64> = v_mat
        .coerce::<f64>()
        .into_dimensionality::<ndarray::Ix2>()
        .map_err(|e| crate::internal::internal(vm, format!("svd v: {e}")))?;
    let mut v_sorted = ndarray::Array2::<f64>::zeros((n, n));
    for (new_col, &old_col) in idx.iter().enumerate() {
        for row in 0..n {
            v_sorted[(row, new_col)] = v_owned[(row, old_col)];
        }
    }
    // Compute U columns = A * V[:, k] / σ_k for σ_k > 0.
    // If m < n, only m singular values exist; trim.
    let k = m.min(n);
    let mut u = ndarray::Array2::<f64>::zeros((m, if full_matrices { m } else { k }));
    for j in 0..k.min(if full_matrices { m } else { k }) {
        let s = sigma[j];
        if s > 1e-12 {
            let v_col = v_sorted.column(j);
            let av = am.dot(&v_col);
            for i in 0..m {
                u[(i, j)] = av[i] / s;
            }
        }
    }
    // Fill remaining U columns (full_matrices and k < m): use Gram-Schmidt
    // against existing columns.
    if full_matrices && k < m {
        for j in k..m {
            // Start with standard basis vector e_j, then orthogonalize.
            let mut new_col = ndarray::Array1::<f64>::zeros(m);
            new_col[j] = 1.0;
            for prev in 0..j {
                let prev_col = u.column(prev);
                let dot: f64 = prev_col.iter().zip(new_col.iter()).map(|(a, b)| a * b).sum();
                for i in 0..m {
                    new_col[i] -= dot * prev_col[i];
                }
            }
            let nrm = new_col.iter().map(|x| x * x).sum::<f64>().sqrt();
            if nrm > 1e-12 {
                for i in 0..m {
                    u[(i, j)] = new_col[i] / nrm;
                }
            }
        }
    }
    // sigma slice (only the m.min(n) entries are meaningful).
    let sigma_out: Vec<f64> = sigma.iter().take(k).copied().collect();
    // V^H is the *transpose* of V (real case).
    let vh = if full_matrices {
        v_sorted.t().to_owned()
    } else {
        // Take first k columns of V → transpose to get rows.
        let mut v_trim = ndarray::Array2::<f64>::zeros((k, n));
        for j in 0..k {
            for row in 0..n {
                v_trim[(j, row)] = v_sorted[(row, j)];
            }
        }
        v_trim
    };
    Ok((
        ArraysD::F64(u.into_dyn()),
        ArraysD::F64(ArrayD::from_shape_vec(IxDyn(&[k]), sigma_out).unwrap_or_default()),
        ArraysD::F64(vh.into_dyn()),
    ))
}

/// `np.cross` for 1-D 3-element vectors.
pub fn cross(a: &ArraysD, b: &ArraysD, vm: &VirtualMachine) -> PyResult<ArraysD> {
    let af = a.cast(DType::F64);
    let bf = b.cast(DType::F64);
    let ArraysD::F64(xa) = af else {
        return Err(crate::internal::internal(vm, "cross: cast a failed"));
    };
    let ArraysD::F64(xb) = bf else {
        return Err(crate::internal::internal(vm, "cross: cast b failed"));
    };
    if xa.len() != 3 || xb.len() != 3 {
        return Err(vm.new_value_error("cross: only 3-element vectors supported".to_string()));
    }
    let a: Vec<f64> = xa.iter().copied().collect();
    let b: Vec<f64> = xb.iter().copied().collect();
    let out = vec![
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ];
    Ok(ArraysD::F64(ArrayD::from_shape_vec(IxDyn(&[3]), out).unwrap_or_default()))
}
