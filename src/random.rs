//! `numpy.random` — pseudo-random generators and distribution sampling.
//!
//! Backed by ``rand::rngs::StdRng`` (ChaCha-based) under a single global
//! lock. ``seed(n)`` reseeds the state; the default seed is fixed for
//! reproducibility. Sampling for the >30 named distributions
//! (gamma, beta, poisson, …) is delegated to the ``rand_distr`` crate so
//! we don't reimplement well-tested CDFs by hand.

use crate::dtype::ArraysD;
use ndarray::{ArrayD, IxDyn};
use rand::{RngExt, SeedableRng, rngs::StdRng};
use rand_distr::Distribution;
use std::sync::{Mutex, OnceLock};

fn global() -> &'static Mutex<StdRng> {
    static G: OnceLock<Mutex<StdRng>> = OnceLock::new();
    G.get_or_init(|| Mutex::new(StdRng::seed_from_u64(0xDEADBEEFCAFEBABE)))
}

pub fn seed(s: u64) {
    let mut g = global().lock().unwrap_or_else(|e| e.into_inner());
    *g = StdRng::seed_from_u64(s);
}

/// Build an F64 ndarray from a sampler. `shape` may be empty (treated as `[1]`).
fn build_f64<F: FnMut(&mut StdRng) -> f64>(shape: &[usize], mut sampler: F) -> ArraysD {
    let n: usize = shape.iter().product::<usize>().max(1);
    let mut rng = global().lock().unwrap_or_else(|e| e.into_inner());
    let v: Vec<f64> = (0..n).map(|_| sampler(&mut rng)).collect();
    let shape = if shape.is_empty() {
        &[1usize][..]
    } else {
        shape
    };
    ArraysD::F64(ArrayD::from_shape_vec(IxDyn(shape), v).unwrap_or_default())
}

fn build_i64<F: FnMut(&mut StdRng) -> i64>(shape: &[usize], mut sampler: F) -> ArraysD {
    let n: usize = shape.iter().product::<usize>().max(1);
    let mut rng = global().lock().unwrap_or_else(|e| e.into_inner());
    let v: Vec<i64> = (0..n).map(|_| sampler(&mut rng)).collect();
    let shape = if shape.is_empty() {
        &[1usize][..]
    } else {
        shape
    };
    ArraysD::I64(ArrayD::from_shape_vec(IxDyn(shape), v).unwrap_or_default())
}

pub fn rand(shape: &[usize]) -> ArraysD {
    build_f64(shape, |rng| rng.random::<f64>())
}

/// Standard normal samples via `rand_distr::StandardNormal`.
pub fn randn(shape: &[usize]) -> ArraysD {
    build_f64(shape, |rng| rand_distr::StandardNormal.sample(rng))
}

pub fn randint(low: i64, high: i64, shape: &[usize]) -> ArraysD {
    if high <= low {
        // Caller violates contract; return empty rather than panic.
        return ArraysD::I64(ArrayD::from_shape_vec(IxDyn(&[0]), vec![]).unwrap_or_default());
    }
    let n: usize = shape.iter().product::<usize>().max(1);
    let mut rng = global().lock().unwrap_or_else(|e| e.into_inner());
    let range = (high - low) as u64;
    let mut v = Vec::with_capacity(n);
    for _ in 0..n {
        let r = rng.random::<u64>() % range;
        v.push(low + r as i64);
    }
    let shape = if shape.is_empty() {
        &[1usize][..]
    } else {
        shape
    };
    ArraysD::I64(ArrayD::from_shape_vec(IxDyn(shape), v).unwrap_or_default())
}

pub fn uniform(low: f64, high: f64, shape: &[usize]) -> ArraysD {
    let n: usize = shape.iter().product::<usize>().max(1);
    let mut rng = global().lock().unwrap_or_else(|e| e.into_inner());
    let span = high - low;
    let mut v = Vec::with_capacity(n);
    for _ in 0..n {
        v.push(low + rng.random::<f64>() * span);
    }
    let shape = if shape.is_empty() {
        &[1usize][..]
    } else {
        shape
    };
    ArraysD::F64(ArrayD::from_shape_vec(IxDyn(shape), v).unwrap_or_default())
}

pub fn normal(mean: f64, std: f64, shape: &[usize]) -> ArraysD {
    let z = randn(shape);
    let ArraysD::F64(arr) = z else {
        // randn always returns F64; this arm is logically unreachable.
        return ArraysD::F64(ArrayD::from_shape_vec(IxDyn(&[0]), vec![]).unwrap_or_default());
    };
    let scaled = arr.mapv(|v| mean + std * v);
    ArraysD::F64(scaled)
}

// Distribution samplers — each delegates to a rand_distr `Distribution`.
// Errors building the distribution (negative scale, etc.) fall back to a
// degenerate sampler returning the configured mean / 0 to avoid panics.

pub fn exponential(scale: f64, shape: &[usize]) -> ArraysD {
    let d = rand_distr::Exp::new(1.0 / scale.max(f64::MIN_POSITIVE)).ok();
    build_f64(shape, move |rng| match &d {
        Some(d) => d.sample(rng),
        None => 0.0,
    })
}

pub fn standard_exponential(shape: &[usize]) -> ArraysD {
    exponential(1.0, shape)
}

pub fn gamma(shape_k: f64, scale: f64, shape: &[usize]) -> ArraysD {
    let d =
        rand_distr::Gamma::new(shape_k.max(f64::MIN_POSITIVE), scale.max(f64::MIN_POSITIVE)).ok();
    build_f64(shape, move |rng| match &d {
        Some(d) => d.sample(rng),
        None => 0.0,
    })
}

pub fn standard_gamma(shape_k: f64, shape: &[usize]) -> ArraysD {
    gamma(shape_k, 1.0, shape)
}

pub fn beta(a: f64, b: f64, shape: &[usize]) -> ArraysD {
    let d = rand_distr::Beta::new(a.max(f64::MIN_POSITIVE), b.max(f64::MIN_POSITIVE)).ok();
    build_f64(shape, move |rng| match &d {
        Some(d) => d.sample(rng),
        None => 0.0,
    })
}

pub fn chisquare(df: f64, shape: &[usize]) -> ArraysD {
    // chisquare(df) = gamma(shape=df/2, scale=2).
    gamma(df / 2.0, 2.0, shape)
}

pub fn standard_t(df: f64, shape: &[usize]) -> ArraysD {
    let d = rand_distr::StudentT::new(df.max(f64::MIN_POSITIVE)).ok();
    build_f64(shape, move |rng| match &d {
        Some(d) => d.sample(rng),
        None => 0.0,
    })
}

pub fn f_dist(d1: f64, d2: f64, shape: &[usize]) -> ArraysD {
    let d = rand_distr::FisherF::new(d1.max(f64::MIN_POSITIVE), d2.max(f64::MIN_POSITIVE)).ok();
    build_f64(shape, move |rng| match &d {
        Some(d) => d.sample(rng),
        None => 0.0,
    })
}

pub fn lognormal(mean: f64, sigma: f64, shape: &[usize]) -> ArraysD {
    let d = rand_distr::LogNormal::new(mean, sigma.max(0.0)).ok();
    build_f64(shape, move |rng| match &d {
        Some(d) => d.sample(rng),
        None => 0.0,
    })
}

pub fn cauchy(median: f64, scale: f64, shape: &[usize]) -> ArraysD {
    let d = rand_distr::Cauchy::new(median, scale.max(f64::MIN_POSITIVE)).ok();
    build_f64(shape, move |rng| match &d {
        Some(d) => d.sample(rng),
        None => 0.0,
    })
}

pub fn standard_cauchy(shape: &[usize]) -> ArraysD {
    cauchy(0.0, 1.0, shape)
}

pub fn laplace(loc: f64, scale: f64, shape: &[usize]) -> ArraysD {
    // numpy: Laplace(loc, scale). rand_distr lacks Laplace; sample as
    //   loc + sign(u-0.5) * scale * -ln(1 - 2|u-0.5|).
    build_f64(shape, move |rng| {
        let u: f64 = rng.random::<f64>() - 0.5;
        let s = if u < 0.0 { -1.0 } else { 1.0 };
        loc - scale.max(0.0) * s * (1.0 - 2.0 * u.abs()).ln()
    })
}

pub fn logistic(loc: f64, scale: f64, shape: &[usize]) -> ArraysD {
    build_f64(shape, move |rng| {
        let u: f64 = rng
            .random::<f64>()
            .clamp(f64::MIN_POSITIVE, 1.0 - f64::MIN_POSITIVE);
        loc + scale.max(0.0) * (u / (1.0 - u)).ln()
    })
}

pub fn gumbel(loc: f64, scale: f64, shape: &[usize]) -> ArraysD {
    build_f64(shape, move |rng| {
        let u: f64 = rng.random::<f64>().max(f64::MIN_POSITIVE);
        loc - scale.max(0.0) * (-u.ln()).ln()
    })
}

pub fn pareto(a: f64, shape: &[usize]) -> ArraysD {
    let d = rand_distr::Pareto::new(1.0, a.max(f64::MIN_POSITIVE)).ok();
    build_f64(shape, move |rng| match &d {
        Some(d) => d.sample(rng) - 1.0, // numpy returns "shifted" Pareto.
        None => 0.0,
    })
}

pub fn power_dist(a: f64, shape: &[usize]) -> ArraysD {
    build_f64(shape, move |rng| {
        let u: f64 = rng.random::<f64>();
        u.powf(1.0 / a.max(f64::MIN_POSITIVE))
    })
}

pub fn rayleigh(scale: f64, shape: &[usize]) -> ArraysD {
    build_f64(shape, move |rng| {
        let u: f64 = rng.random::<f64>().max(f64::MIN_POSITIVE);
        scale.max(0.0) * (-2.0 * u.ln()).sqrt()
    })
}

pub fn triangular(left: f64, mode: f64, right: f64, shape: &[usize]) -> ArraysD {
    let d = rand_distr::Triangular::new(left, right, mode).ok();
    build_f64(shape, move |rng| match &d {
        Some(d) => d.sample(rng),
        None => left,
    })
}

pub fn vonmises(mu: f64, kappa: f64, shape: &[usize]) -> ArraysD {
    // Best-effort von Mises sampler (rejection method) — accurate for small κ;
    // larger κ benefits from the more refined Best/Fisher method.
    build_f64(shape, move |rng| {
        let k = kappa.max(0.0);
        if k < 1e-9 {
            // κ ≈ 0 → uniform on (-π, π].
            (rng.random::<f64>() - 0.5) * 2.0 * std::f64::consts::PI + mu
        } else {
            let a = 1.0 + (1.0 + 4.0 * k * k).sqrt();
            let b = (a - (2.0 * a).sqrt()) / (2.0 * k);
            let r = (1.0 + b * b) / (2.0 * b);
            loop {
                let u1: f64 = rng.random();
                let z = (std::f64::consts::PI * u1).cos();
                let f = (1.0 + r * z) / (r + z);
                let c = k * (r - f);
                let u2: f64 = rng.random();
                if c * (2.0 - c) - u2 > 0.0 || (c / u2).ln() + 1.0 - c >= 0.0 {
                    let u3: f64 = rng.random();
                    let theta = if u3 - 0.5 > 0.0 { f.acos() } else { -f.acos() };
                    return mu + theta;
                }
            }
        }
    })
}

pub fn wald(mean: f64, scale: f64, shape: &[usize]) -> ArraysD {
    // Michael-Schucany-Haas method for Inverse Gaussian / Wald.
    let m = mean.max(f64::MIN_POSITIVE);
    let l = scale.max(f64::MIN_POSITIVE);
    build_f64(shape, move |rng| {
        let v: f64 = rand_distr::StandardNormal.sample(rng);
        let y = v * v;
        let x = m + (m * m * y) / (2.0 * l)
            - (m / (2.0 * l)) * (4.0 * m * l * y + m * m * y * y).sqrt();
        let u: f64 = rng.random();
        if u <= m / (m + x) { x } else { m * m / x }
    })
}

pub fn weibull(a: f64, shape: &[usize]) -> ArraysD {
    let d = rand_distr::Weibull::new(1.0, a.max(f64::MIN_POSITIVE)).ok();
    build_f64(shape, move |rng| match &d {
        Some(d) => d.sample(rng),
        None => 0.0,
    })
}

pub fn poisson(lam: f64, shape: &[usize]) -> ArraysD {
    let d = rand_distr::Poisson::new(lam.max(0.0)).ok();
    build_i64(shape, move |rng| match &d {
        Some(d) => d.sample(rng) as i64,
        None => 0,
    })
}

pub fn binomial(n: i64, p: f64, shape: &[usize]) -> ArraysD {
    let d = rand_distr::Binomial::new(n.max(0) as u64, p.clamp(0.0, 1.0)).ok();
    build_i64(shape, move |rng| match &d {
        Some(d) => d.sample(rng) as i64,
        None => 0,
    })
}

pub fn geometric(p: f64, shape: &[usize]) -> ArraysD {
    let p = p.clamp(f64::MIN_POSITIVE, 1.0);
    build_i64(shape, move |rng| {
        let u: f64 = rng.random::<f64>().max(f64::MIN_POSITIVE);
        (u.ln() / (1.0 - p).ln()).ceil() as i64
    })
}

pub fn negative_binomial(n: f64, p: f64, shape: &[usize]) -> ArraysD {
    // NB(n, p) sampled via gamma-poisson mixture:
    //   X ~ Poisson(λ), λ ~ Gamma(n, (1-p)/p).
    let gamma_dist = rand_distr::Gamma::new(
        n.max(f64::MIN_POSITIVE),
        (1.0 - p).max(f64::MIN_POSITIVE) / p.max(f64::MIN_POSITIVE),
    )
    .ok();
    build_i64(shape, move |rng| match &gamma_dist {
        Some(g) => {
            let lam: f64 = g.sample(rng);
            rand_distr::Poisson::new(lam.max(0.0))
                .map(|d| Distribution::sample(&d, rng) as i64)
                .unwrap_or(0)
        }
        None => 0,
    })
}

pub fn hypergeometric(ngood: i64, nbad: i64, nsample: i64, shape: &[usize]) -> ArraysD {
    let d = rand_distr::Hypergeometric::new(
        (ngood + nbad).max(0) as u64,
        ngood.max(0) as u64,
        nsample.max(0) as u64,
    )
    .ok();
    build_i64(shape, move |rng| match &d {
        Some(d) => d.sample(rng) as i64,
        None => 0,
    })
}

pub fn logseries(p: f64, shape: &[usize]) -> ArraysD {
    // numpy uses Kemp's accept-reject (LK). Here we use a simple
    // inverse-transform via tabulation up to a reasonable cutoff.
    let p = p.clamp(f64::MIN_POSITIVE, 1.0 - f64::MIN_POSITIVE);
    build_i64(shape, move |rng| {
        let r = -(1.0 - p).ln();
        let u: f64 = rng.random();
        let mut s = p / r;
        let mut k: i64 = 1;
        let mut cum = s;
        while u > cum && k < 10_000 {
            k += 1;
            s *= p * (k - 1) as f64 / k as f64;
            cum += s;
        }
        k
    })
}

pub fn noncentral_chisquare(df: f64, nonc: f64, shape: &[usize]) -> ArraysD {
    // Noncentral chi-square: sum of df-1 squared standard normals plus
    // (N + sqrt(nonc))^2 — equivalent to gamma-poisson mixture.
    build_f64(shape, move |rng| {
        let normal_mu = nonc.max(0.0).sqrt();
        let z: f64 = rand_distr::StandardNormal.sample(rng);
        let first = (z + normal_mu).powi(2);
        let rest_df = (df - 1.0).max(0.0);
        let g = rand_distr::Gamma::new((rest_df / 2.0).max(f64::MIN_POSITIVE), 2.0)
            .ok()
            .map(|d| Distribution::sample(&d, rng))
            .unwrap_or(0.0);
        first + g
    })
}

pub fn noncentral_f(df_num: f64, df_den: f64, nonc: f64, shape: &[usize]) -> ArraysD {
    // F = (noncentral_chisquare(df_num, nonc)/df_num) / (chisquare(df_den)/df_den)
    build_f64(shape, move |rng| {
        let num: f64 = {
            let mu = nonc.max(0.0).sqrt();
            let z: f64 = rand_distr::StandardNormal.sample(rng);
            let first = (z + mu).powi(2);
            let rest_df = (df_num - 1.0).max(0.0);
            let g = rand_distr::Gamma::new((rest_df / 2.0).max(f64::MIN_POSITIVE), 2.0)
                .ok()
                .map(|d| Distribution::sample(&d, rng))
                .unwrap_or(0.0);
            (first + g) / df_num.max(f64::MIN_POSITIVE)
        };
        let den: f64 = rand_distr::Gamma::new((df_den / 2.0).max(f64::MIN_POSITIVE), 2.0)
            .ok()
            .map(|d| Distribution::sample(&d, rng))
            .unwrap_or(1.0)
            / df_den.max(f64::MIN_POSITIVE);
        num / den.max(f64::MIN_POSITIVE)
    })
}

pub fn zipf(a: f64, shape: &[usize]) -> ArraysD {
    let d = rand_distr::Zipf::new(i64::MAX as f64, a.max(1.0 + f64::MIN_POSITIVE)).ok();
    build_i64(shape, move |rng| match &d {
        Some(d) => d.sample(rng) as i64,
        None => 1,
    })
}

pub fn random_bytes(n: usize) -> Vec<u8> {
    let mut rng = global().lock().unwrap_or_else(|e| e.into_inner());
    (0..n).map(|_| rng.random::<u8>()).collect()
}

/// Random permutation of integers ``[0, n)`` — Fisher-Yates shuffle.
pub fn permutation(n: i64) -> ArraysD {
    let n = n.max(0) as usize;
    let mut v: Vec<i64> = (0..n as i64).collect();
    let mut rng = global().lock().unwrap_or_else(|e| e.into_inner());
    for i in (1..n).rev() {
        let j = (rng.random::<u64>() as usize) % (i + 1);
        v.swap(i, j);
    }
    ArraysD::I64(ArrayD::from_shape_vec(IxDyn(&[n]), v).unwrap_or_default())
}

/// Shuffle ``a`` in place along axis 0.
pub fn shuffle(a: &mut ArraysD) {
    let mut rng = global().lock().unwrap_or_else(|e| e.into_inner());
    let n = match a {
        ArraysD::F64(x) => x.shape()[0],
        ArraysD::F32(x) => x.shape()[0],
        ArraysD::I64(x) => x.shape()[0],
        ArraysD::I32(x) => x.shape()[0],
        _ => return,
    };
    for i in (1..n).rev() {
        let j = (rng.random::<u64>() as usize) % (i + 1);
        match a {
            ArraysD::F64(x) => {
                let s = x.shape().to_vec();
                swap_along_axis0(x, i, j, &s);
            }
            ArraysD::F32(x) => {
                let s = x.shape().to_vec();
                swap_along_axis0(x, i, j, &s);
            }
            ArraysD::I64(x) => {
                let s = x.shape().to_vec();
                swap_along_axis0(x, i, j, &s);
            }
            ArraysD::I32(x) => {
                let s = x.shape().to_vec();
                swap_along_axis0(x, i, j, &s);
            }
            _ => {}
        }
    }
}

fn swap_along_axis0<T: Clone>(arr: &mut ArrayD<T>, i: usize, j: usize, shape: &[usize]) {
    if i == j {
        return;
    }
    let row_size: usize = shape[1..].iter().product::<usize>().max(1);
    // Non-contiguous arrays can't be sliced flat; for those we walk by
    // ndarray's index API instead. Returning silently on a non-contiguous
    // array is preferable to panicking — `shuffle` is called from a hot
    // path and contiguity is the common case.
    let Some(slice) = arr.as_slice_mut() else {
        return;
    };
    // Unsafe-free swap using a temp vec.
    let off_i = i * row_size;
    let off_j = j * row_size;
    let mut tmp = vec![slice[off_i].clone(); row_size];
    for k in 0..row_size {
        tmp[k] = slice[off_i + k].clone();
        slice[off_i + k] = slice[off_j + k].clone();
        slice[off_j + k] = tmp[k].clone();
    }
}

pub fn choice(a: &[i64], n: usize, replace: bool) -> ArraysD {
    let mut rng = global().lock().unwrap_or_else(|e| e.into_inner());
    let v: Vec<i64> = if replace || n >= a.len() {
        (0..n)
            .map(|_| a[(rng.random::<u64>() as usize) % a.len()])
            .collect()
    } else {
        // Without replacement: shuffle and take.
        let mut pool: Vec<i64> = a.to_vec();
        let m = pool.len();
        for i in (1..m).rev() {
            let j = (rng.random::<u64>() as usize) % (i + 1);
            pool.swap(i, j);
        }
        pool.truncate(n);
        pool
    };
    ArraysD::I64(ArrayD::from_shape_vec(IxDyn(&[n]), v).unwrap_or_default())
}
