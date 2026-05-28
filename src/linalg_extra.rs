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

/// Copy a row-major `ndarray::Array2<f64>` into a (column-major) `faer::Mat`.
fn nd_to_faer(arr: &ndarray::Array2<f64>) -> faer::Mat<f64> {
    let (r, c) = arr.dim();
    faer::Mat::from_fn(r, c, |i, j| arr[(i, j)])
}

/// Copy a `faer::MatRef<f64>` back into a row-major `ndarray::Array2<f64>`.
fn faer_to_nd(m: faer::MatRef<'_, f64>) -> ndarray::Array2<f64> {
    let (r, c) = (m.nrows(), m.ncols());
    ndarray::Array2::from_shape_fn((r, c), |(i, j)| m[(i, j)])
}

/// Sign of a permutation given as a forward index array (output = `(-1)^(n - cycles)`).
fn perm_sign(fwd: &[usize]) -> f64 {
    let n = fwd.len();
    let mut visited = vec![false; n];
    let mut cycles = 0usize;
    for i in 0..n {
        if !visited[i] {
            cycles += 1;
            let mut j = i;
            while !visited[j] {
                visited[j] = true;
                j = fwd[j];
            }
        }
    }
    if (n - cycles) % 2 == 0 { 1.0 } else { -1.0 }
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
        Some(NormOrd::Num(p)) if (p - 2.0).abs() < f64::EPSILON => {
            // Largest singular value.
            let sv = singular_values(data, rows, cols, vm)?;
            Ok(sv.into_iter().fold(0.0_f64, f64::max))
        }
        Some(NormOrd::Num(p)) if (p + 2.0).abs() < f64::EPSILON => {
            // Smallest singular value.
            let sv = singular_values(data, rows, cols, vm)?;
            Ok(sv.into_iter().fold(f64::INFINITY, f64::min))
        }
        Some(NormOrd::Nuc) => {
            // Sum of singular values.
            let sv = singular_values(data, rows, cols, vm)?;
            Ok(sv.into_iter().sum())
        }
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
    let mat = nd_to_faer(&m);
    let det = mat.determinant();
    Ok(ArraysD::F64(ArrayD::from_elem(IxDyn(&[]), det)))
}

pub fn inv(a: &ArraysD, vm: &VirtualMachine) -> PyResult<ArraysD> {
    use faer::linalg::solvers::DenseSolveCore;
    let m = as_f64_2d(a, vm)?;
    let (r, c) = m.dim();
    if r != c {
        return Err(vm.new_value_error("inv: matrix must be square".to_string()));
    }
    let mat = nd_to_faer(&m);
    if mat.determinant() == 0.0 {
        return Err(vm.new_value_error("inv: singular matrix".to_string()));
    }
    let inv = mat.partial_piv_lu().inverse();
    Ok(ArraysD::F64(faer_to_nd(inv.as_ref()).into_dyn()))
}

pub fn solve(a: &ArraysD, b: &ArraysD, vm: &VirtualMachine) -> PyResult<ArraysD> {
    use faer::linalg::solvers::Solve;
    let am = as_f64_2d(a, vm)?;
    let (r, c) = am.dim();
    if r != c {
        return Err(vm.new_value_error("solve: matrix must be square".to_string()));
    }
    let bf = b.cast(DType::F64);
    let (b_mat, was_1d) = match bf {
        ArraysD::F64(arr) => match arr.ndim() {
            1 => {
                let n = arr.len();
                if n != r {
                    return Err(vm.new_value_error(format!(
                        "solve: A rows ({r}) != b rows ({n})"
                    )));
                }
                (faer::Mat::<f64>::from_fn(n, 1, |i, _| arr[i]), true)
            }
            2 => {
                let arr2 = arr.into_dimensionality::<ndarray::Ix2>()
                    .map_err(|e| crate::internal::internal(vm, format!("solve: {e}")))?;
                if arr2.dim().0 != r {
                    return Err(vm.new_value_error(format!(
                        "solve: A rows ({}) != b rows ({})",
                        r, arr2.dim().0
                    )));
                }
                (nd_to_faer(&arr2), false)
            }
            _ => return Err(vm.new_value_error("solve: b must be 1-D or 2-D".to_string())),
        },
        _ => return Err(crate::internal::internal(vm, "solve: cast to F64 failed")),
    };
    let mat = nd_to_faer(&am);
    if mat.determinant() == 0.0 {
        return Err(vm.new_value_error("solve: singular matrix".to_string()));
    }
    let x = mat.partial_piv_lu().solve(&b_mat);
    if was_1d {
        let n = x.nrows();
        let v: Vec<f64> = (0..n).map(|i| x[(i, 0)]).collect();
        Ok(ArraysD::F64(ArrayD::from_shape_vec(IxDyn(&[n]), v).unwrap_or_default()))
    } else {
        Ok(ArraysD::F64(faer_to_nd(x.as_ref()).into_dyn()))
    }
}

pub fn matrix_rank(a: &ArraysD, vm: &VirtualMachine) -> PyResult<ArraysD> {
    let m = as_f64_2d(a, vm)?;
    let rank = svd_rank(&m, vm)? as i64;
    Ok(ArraysD::I64(ArrayD::from_elem(IxDyn(&[]), rank)))
}

/// Numerical rank via SVD: count singular values strictly above
/// `max(m, n) * largest_sv * eps`, matching numpy's default tolerance.
fn svd_rank(m: &ndarray::Array2<f64>, vm: &VirtualMachine) -> PyResult<usize> {
    let (rows, cols) = m.dim();
    if rows == 0 || cols == 0 {
        return Ok(0);
    }
    let mat = nd_to_faer(m);
    let svd = mat.thin_svd()
        .map_err(|e| vm.new_value_error(format!("matrix_rank svd failed: {e:?}")))?;
    let s = svd.S().column_vector();
    let k = s.nrows();
    if k == 0 {
        return Ok(0);
    }
    let smax = (0..k).map(|i| s[i]).fold(0.0_f64, f64::max);
    let tol = rows.max(cols) as f64 * smax * f64::EPSILON;
    Ok((0..k).filter(|&i| s[i] > tol).count())
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
    let mat = nd_to_faer(&m);
    let llt = mat.llt(faer::Side::Lower)
        .map_err(|_| vm.new_value_error("cholesky: matrix is not positive-definite".to_string()))?;
    // faer's L() is lower-triangular but may not zero the strict upper part —
    // numpy expects a strictly-lower-triangular layout.
    let l_ref = llt.L();
    let mut l = ndarray::Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..=i {
            l[(i, j)] = l_ref[(i, j)];
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

/// QR decomposition via faer.
///
/// * `mode = Reduced` (numpy default) — returns `(Q[:, :k], R[:k, :])`.
/// * `mode = Complete` — returns the full m×m Q and m×n R.
/// * `mode = R` — returns `(R[:k, :],)` (a one-element vec) — caller picks
///   the convention.
pub fn qr(a: &ArraysD, mode: QrMode, vm: &VirtualMachine) -> PyResult<(ArraysD, ArraysD)> {
    let m = as_f64_2d(a, vm)?;
    let (rows, cols) = m.dim();
    let k = rows.min(cols);
    let mat = nd_to_faer(&m);
    let qr = mat.qr();
    // faer's R() is full m×n with the strict-lower part holding householder
    // garbage; mask the lower triangle to zero so the output matches numpy.
    let r_full_ref = qr.R();
    let mut r_full = ndarray::Array2::<f64>::zeros((rows, cols));
    for i in 0..rows {
        for j in i..cols {
            r_full[(i, j)] = r_full_ref[(i, j)];
        }
    }
    let (q_out, r_out) = match mode {
        QrMode::Complete => {
            let q = qr.compute_Q();
            (faer_to_nd(q.as_ref()), r_full)
        }
        QrMode::Reduced => {
            let q = qr.compute_thin_Q();
            let r_red = r_full.slice(ndarray::s![..k, ..]).to_owned();
            (faer_to_nd(q.as_ref()), r_red)
        }
        QrMode::R => {
            let r_red = r_full.slice(ndarray::s![..k, ..]).to_owned();
            (ndarray::Array2::<f64>::zeros((0, 0)), r_red)
        }
    };
    Ok((ArraysD::F64(q_out.into_dyn()), ArraysD::F64(r_out.into_dyn())))
}

/// numpy.linalg.lstsq returns this tuple. `residuals` is empty when the
/// system is underdetermined or rank-deficient (matching numpy); `singular`
/// holds the singular values of A.
pub struct LstsqResult {
    pub solution: ArraysD,
    pub residuals: ArraysD,
    pub rank: i64,
    pub singular: ArraysD,
}

/// `numpy.linalg.lstsq(A, b)` — least-squares solution via faer's SVD-backed
/// `solve_lstsq`. The single-array `lstsq` (used internally) returns just
/// the solution; the tuple form `lstsq_full` returns `(x, residuals, rank, s)`.
pub fn lstsq_full(
    a: &ArraysD,
    b: &ArraysD,
    vm: &VirtualMachine,
) -> PyResult<LstsqResult> {
    let am = as_f64_2d(a, vm)?;
    let (m, n) = am.dim();
    let (b_mat, was_1d) = b_to_faer(b, m, vm)?;
    let mat = nd_to_faer(&am);
    let svd = mat.thin_svd()
        .map_err(|e| vm.new_value_error(format!("lstsq svd failed: {e:?}")))?;
    let s = svd.S().column_vector();
    let k = s.nrows();
    let smax = (0..k).map(|i| s[i]).fold(0.0_f64, f64::max);
    let tol = m.max(n) as f64 * smax * f64::EPSILON;
    let rank = (0..k).filter(|&i| s[i] > tol).count() as i64;
    // SolveLstsq's solve_lstsq returns the (n × nrhs) minimum-norm solution.
    use faer::linalg::solvers::SolveLstsq;
    let x = svd.solve_lstsq(&b_mat);
    let solution = faer_x_to_arraysd(&x, n, was_1d);
    // Residuals (per RHS column): only emitted when m > rank and the system
    // is overdetermined-and-full-rank (numpy convention).
    let residuals = if m > rank as usize && rank as usize == n {
        let nrhs = b_mat.ncols();
        let mut out = ndarray::Array1::<f64>::zeros(nrhs);
        for col in 0..nrhs {
            let mut s = 0.0_f64;
            for i in 0..m {
                let mut ax_i = 0.0_f64;
                for j in 0..n {
                    ax_i += am[(i, j)] * x[(j, col)];
                }
                let r = ax_i - b_mat[(i, col)];
                s += r * r;
            }
            out[col] = s;
        }
        ArraysD::F64(out.into_dyn())
    } else {
        crate::create::zeros(&[0], DType::F64)
    };
    let sv: Vec<f64> = (0..k).map(|i| s[i]).collect();
    let singular = ArraysD::F64(ArrayD::from_shape_vec(IxDyn(&[k]), sv).unwrap_or_default());
    Ok(LstsqResult { solution, residuals, rank, singular })
}

/// Convert `b` (1-D or 2-D, any numeric dtype) to a `faer::Mat<f64>` shaped
/// `(m, nrhs)`. Returns the matrix and whether the input was 1-D so callers
/// can reshape the output back.
fn b_to_faer(
    b: &ArraysD,
    expected_rows: usize,
    vm: &VirtualMachine,
) -> PyResult<(faer::Mat<f64>, bool)> {
    let bf = b.cast(DType::F64);
    match bf {
        ArraysD::F64(arr) => match arr.ndim() {
            1 => {
                let n = arr.len();
                if n != expected_rows {
                    return Err(vm.new_value_error(format!(
                        "A rows ({expected_rows}) != b rows ({n})"
                    )));
                }
                Ok((faer::Mat::<f64>::from_fn(n, 1, |i, _| arr[i]), true))
            }
            2 => {
                let arr2 = arr.into_dimensionality::<ndarray::Ix2>()
                    .map_err(|e| crate::internal::internal(vm, format!("b_to_faer: {e}")))?;
                let (br, bc) = arr2.dim();
                if br != expected_rows {
                    return Err(vm.new_value_error(format!(
                        "A rows ({expected_rows}) != b rows ({br})"
                    )));
                }
                Ok((faer::Mat::<f64>::from_fn(br, bc, |i, j| arr2[(i, j)]), false))
            }
            _ => Err(vm.new_value_error("b must be 1-D or 2-D".to_string())),
        },
        _ => Err(crate::internal::internal(vm, "b cast to F64 failed")),
    }
}

/// Reshape a faer (n × nrhs) solution back into either a 1-D ArraysD (when
/// the original RHS was 1-D) or a 2-D one.
fn faer_x_to_arraysd(x: &faer::Mat<f64>, n: usize, was_1d: bool) -> ArraysD {
    if was_1d {
        let v: Vec<f64> = (0..n).map(|i| x[(i, 0)]).collect();
        ArraysD::F64(ArrayD::from_shape_vec(IxDyn(&[n]), v).unwrap_or_default())
    } else {
        let cols = x.ncols();
        let mut out = ndarray::Array2::<f64>::zeros((n, cols));
        for i in 0..n {
            for j in 0..cols {
                out[(i, j)] = x[(i, j)];
            }
        }
        ArraysD::F64(out.into_dyn())
    }
}

/// Bare-bones lstsq returning just the solution. Used by `pinv`,
/// `polyfit`, and as the workhorse for `lstsq_full`.
pub fn lstsq(a: &ArraysD, b: &ArraysD, vm: &VirtualMachine) -> PyResult<ArraysD> {
    use faer::linalg::solvers::SolveLstsq;
    let am = as_f64_2d(a, vm)?;
    let (m, n) = am.dim();
    let (b_mat, was_1d) = b_to_faer(b, m, vm)?;
    let mat = nd_to_faer(&am);
    let svd = mat.thin_svd()
        .map_err(|e| vm.new_value_error(format!("lstsq svd failed: {e:?}")))?;
    let x = svd.solve_lstsq(&b_mat);
    Ok(faer_x_to_arraysd(&x, n, was_1d))
}

/// `numpy.linalg.pinv(A)` — Moore-Penrose pseudoinverse via faer SVD.
pub fn pinv(a: &ArraysD, vm: &VirtualMachine) -> PyResult<ArraysD> {
    let am = as_f64_2d(a, vm)?;
    let mat = nd_to_faer(&am);
    let svd = mat.thin_svd()
        .map_err(|e| vm.new_value_error(format!("pinv svd failed: {e:?}")))?;
    let p = svd.pseudoinverse();
    Ok(ArraysD::F64(faer_to_nd(p.as_ref()).into_dyn()))
}

/// `numpy.linalg.eigvalsh(A)` — eigenvalues of a symmetric/Hermitian real
/// matrix via faer's self-adjoint EVD. Returns eigenvalues sorted ascending.
pub fn eigvalsh(a: &ArraysD, vm: &VirtualMachine) -> PyResult<ArraysD> {
    let (vals, _) = eigh(a, vm)?;
    Ok(vals)
}

/// `np.linalg.eigh(A)` for a symmetric matrix via faer. Returns
/// (eigenvalues sorted ascending, eigenvectors as columns).
pub fn eigh(a: &ArraysD, vm: &VirtualMachine) -> PyResult<(ArraysD, ArraysD)> {
    let am = as_f64_2d(a, vm)?;
    let (r, c) = am.dim();
    if r != c {
        return Err(vm.new_value_error("eigh: matrix must be square".to_string()));
    }
    let n = r;
    let mat = nd_to_faer(&am);
    let eig = mat.self_adjoint_eigen(faer::Side::Lower)
        .map_err(|e| vm.new_value_error(format!("eigh failed: {e:?}")))?;
    let s = eig.S().column_vector();
    let u = eig.U();
    // faer returns eigenvalues ascending already; sort defensively.
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&i, &j| s[i].partial_cmp(&s[j]).unwrap_or(std::cmp::Ordering::Equal));
    let eigs: Vec<f64> = order.iter().map(|&i| s[i]).collect();
    let mut vec_out = ndarray::Array2::<f64>::zeros((n, n));
    for (new_j, &old_j) in order.iter().enumerate() {
        for k in 0..n {
            vec_out[(k, new_j)] = u[(k, old_j)];
        }
    }
    Ok((
        ArraysD::F64(ArrayD::from_shape_vec(IxDyn(&[n]), eigs).unwrap_or_default()),
        ArraysD::F64(vec_out.into_dyn()),
    ))
}

/// `np.linalg.eig(A)` — general (possibly non-symmetric) eigendecomposition
/// via faer. Always returns complex128 eigenvalues and eigenvectors, matching
/// numpy's behavior for real input that has complex eigenvalues.
pub fn eig(a: &ArraysD, vm: &VirtualMachine) -> PyResult<(ArraysD, ArraysD)> {
    let am = as_f64_2d(a, vm)?;
    let (r, c) = am.dim();
    if r != c {
        return Err(vm.new_value_error("eig: matrix must be square".to_string()));
    }
    let n = r;
    let m = faer::Mat::from_fn(n, n, |i, j| am[(i, j)]);
    let eigen = m.eigen()
        .map_err(|e| vm.new_value_error(format!("eig failed: {e:?}")))?;
    let s = eigen.S().column_vector();
    let u = eigen.U();
    let vals: Vec<crate::dtype::C64> = (0..n).map(|i| s[i]).collect();
    let mut vecs: Vec<crate::dtype::C64> = Vec::with_capacity(n * n);
    for i in 0..n {
        for j in 0..n {
            vecs.push(u[(i, j)]);
        }
    }
    Ok((
        ArraysD::C128(ArrayD::from_shape_vec(IxDyn(&[n]), vals).unwrap_or_default()),
        ArraysD::C128(ArrayD::from_shape_vec(IxDyn(&[n, n]), vecs).unwrap_or_default()),
    ))
}

/// `np.linalg.eigvals(A)` — eigenvalues only (general matrix). See `eig`.
pub fn eigvals(a: &ArraysD, vm: &VirtualMachine) -> PyResult<ArraysD> {
    let (vals, _) = eig(a, vm)?;
    Ok(vals)
}

/// `np.linalg.slogdet(A)` — (sign, log(|det|)). Computed via LU so that the
/// log-domain accumulation avoids over/underflow for large matrices.
pub fn slogdet(a: &ArraysD, vm: &VirtualMachine) -> PyResult<(ArraysD, ArraysD)> {
    let m = as_f64_2d(a, vm)?;
    let (r, c) = m.dim();
    if r != c {
        return Err(vm.new_value_error("slogdet: matrix must be square".to_string()));
    }
    let n = r;
    let mat = nd_to_faer(&m);
    let lu = mat.partial_piv_lu();
    let u = lu.U();
    let p = lu.P();
    let (fwd, _) = p.arrays();
    let mut sign = perm_sign(fwd);
    let mut logabs = 0.0f64;
    for i in 0..n {
        let d = u[(i, i)];
        if d == 0.0 {
            return Ok((
                ArraysD::F64(ArrayD::from_elem(IxDyn(&[]), 0.0)),
                ArraysD::F64(ArrayD::from_elem(IxDyn(&[]), f64::NEG_INFINITY)),
            ));
        }
        if d < 0.0 {
            sign = -sign;
        }
        logabs += d.abs().ln();
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

/// `np.linalg.svd(A, full_matrices=True)` — singular value decomposition via
/// faer. Returns (U, Σ, V^H) where A = U Σ V^H.
pub fn svd(
    a: &ArraysD,
    full_matrices: bool,
    vm: &VirtualMachine,
) -> PyResult<(ArraysD, ArraysD, ArraysD)> {
    let am = as_f64_2d(a, vm)?;
    let (m, n) = am.dim();
    let k = m.min(n);
    let mat = nd_to_faer(&am);
    let svd = if full_matrices {
        mat.svd()
    } else {
        mat.thin_svd()
    }
    .map_err(|e| vm.new_value_error(format!("svd failed: {e:?}")))?;
    let u_ref = svd.U();
    let v_ref = svd.V();
    let s_ref = svd.S().column_vector();
    let u = faer_to_nd(u_ref);
    // V is (n, k) in thin / (n, n) in full. numpy wants V^H of shape (k, n) or (n, n).
    let v_rows = v_ref.nrows();
    let v_cols = v_ref.ncols();
    let mut vh = ndarray::Array2::<f64>::zeros((v_cols, v_rows));
    for i in 0..v_rows {
        for j in 0..v_cols {
            vh[(j, i)] = v_ref[(i, j)];
        }
    }
    let sigma: Vec<f64> = (0..k).map(|i| s_ref[i]).collect();
    Ok((
        ArraysD::F64(u.into_dyn()),
        ArraysD::F64(ArrayD::from_shape_vec(IxDyn(&[k]), sigma).unwrap_or_default()),
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

/// Singular values of a row-major `rows x cols` matrix via faer's thin SVD.
fn singular_values(
    data: &[f64],
    rows: usize,
    cols: usize,
    vm: &VirtualMachine,
) -> PyResult<Vec<f64>> {
    let m = faer::Mat::<f64>::from_fn(rows, cols, |r, c| data[r * cols + c]);
    let svd = m.thin_svd()
        .map_err(|e| vm.new_value_error(format!("svd failed: {e:?}")))?;
    let s = svd.S().column_vector();
    Ok((0..s.nrows()).map(|i| s[i]).collect())
}
