//! `numpy.random` — pseudo-random generators (PCG-XSH-RR).
//!
//! The state is a single global `OnceLock<Mutex<Pcg64>>`. Calling
//! `seed(n)` resets the state; otherwise a default seed is used.
//!
//! This isn't numpy's exact RNG (which uses BitGenerator / PCG64) but the
//! algorithm family is the same so distributions are statistically sound.

use crate::dtype::ArraysD;
use ndarray::{ArrayD, IxDyn};
use std::sync::{Mutex, OnceLock};

#[derive(Debug)]
struct Pcg64 {
    state: u128,
    inc: u128,
}

impl Pcg64 {
    fn from_seed(seed: u64) -> Self {
        let mut p = Pcg64 {
            state: 0,
            inc: (seed as u128) << 1 | 1,
        };
        p.next_u64();
        p.state = p.state.wrapping_add(seed as u128);
        p.next_u64();
        p
    }

    fn next_u64(&mut self) -> u64 {
        let old = self.state;
        self.state = old.wrapping_mul(6364136223846793005u128).wrapping_add(self.inc);
        let xorshifted = (((old >> 18) ^ old) >> 27) as u64;
        let rot = (old >> 59) as u32;
        xorshifted.rotate_right(rot)
    }

    fn next_f64(&mut self) -> f64 {
        // Convert top 53 bits to [0, 1).
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

fn global() -> &'static Mutex<Pcg64> {
    static G: OnceLock<Mutex<Pcg64>> = OnceLock::new();
    G.get_or_init(|| Mutex::new(Pcg64::from_seed(0xDEADBEEFCAFEBABE)))
}

pub fn seed(s: u64) {
    let mut g = global().lock().unwrap_or_else(|e| e.into_inner());
    *g = Pcg64::from_seed(s);
}

pub fn rand(shape: &[usize]) -> ArraysD {
    let n: usize = shape.iter().product::<usize>().max(1);
    let mut rng = global().lock().unwrap_or_else(|e| e.into_inner());
    let mut v = Vec::with_capacity(n);
    for _ in 0..n {
        v.push(rng.next_f64());
    }
    let shape = if shape.is_empty() { &[1usize][..] } else { shape };
    ArraysD::F64(ArrayD::from_shape_vec(IxDyn(shape), v).unwrap_or_default())
}

/// Standard normal — Box–Muller.
pub fn randn(shape: &[usize]) -> ArraysD {
    let n: usize = shape.iter().product::<usize>().max(1);
    let mut rng = global().lock().unwrap_or_else(|e| e.into_inner());
    let mut v = Vec::with_capacity(n);
    let mut i = 0;
    while i < n {
        let u1 = rng.next_f64().max(1e-300); // avoid log(0)
        let u2 = rng.next_f64();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f64::consts::PI * u2;
        v.push(r * theta.cos());
        i += 1;
        if i < n {
            v.push(r * theta.sin());
            i += 1;
        }
    }
    let shape = if shape.is_empty() { &[1usize][..] } else { shape };
    ArraysD::F64(ArrayD::from_shape_vec(IxDyn(shape), v).unwrap_or_default())
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
        let r = rng.next_u64() % range;
        v.push(low + r as i64);
    }
    let shape = if shape.is_empty() { &[1usize][..] } else { shape };
    ArraysD::I64(ArrayD::from_shape_vec(IxDyn(shape), v).unwrap_or_default())
}

pub fn uniform(low: f64, high: f64, shape: &[usize]) -> ArraysD {
    let n: usize = shape.iter().product::<usize>().max(1);
    let mut rng = global().lock().unwrap_or_else(|e| e.into_inner());
    let span = high - low;
    let mut v = Vec::with_capacity(n);
    for _ in 0..n {
        v.push(low + rng.next_f64() * span);
    }
    let shape = if shape.is_empty() { &[1usize][..] } else { shape };
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
