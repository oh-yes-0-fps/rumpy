//! `numpy.einsum` — the subscript-string variant.
//!
//! We parse `"<spec1>,<spec2>,...->result"` (or implicit output via
//! `"<spec1>,<spec2>,..."`), then run a naive multi-loop summation. This
//! covers the most-used cases (`ij,jk->ik`, `ii->`, `ij->ji`, `i,i->`,
//! batched matmul `bij,bjk->bik`, outer product `i,j->ij`, etc.). It is
//! not the optimal-contraction-order machine numpy implements — just the
//! straightforward Einstein-summation executor.

use crate::dtype::{ArraysD, CoerceArray, DType};
use crate::internal::internal;
use crate::promote::promote_many;
use ndarray::{ArrayD, IxDyn};
use rustpython_vm::{PyResult, VirtualMachine};

pub fn einsum(
    spec: &str,
    operands: &[ArraysD],
    vm: &VirtualMachine,
) -> PyResult<ArraysD> {
    let (input_specs, output_spec) = parse_spec(spec, vm)?;
    if input_specs.len() != operands.len() {
        return Err(vm.new_value_error(format!(
            "einsum: spec has {} operands, got {}",
            input_specs.len(),
            operands.len()
        )));
    }
    // Greedy pair contraction for 3+ operands. The naive executor below
    // touches every label, so contracting two-at-a-time keeps each pass
    // O(prod(labels-in-that-pair)) rather than O(prod-all-labels).
    if operands.len() > 2 {
        return einsum_greedy(input_specs, output_spec, operands, vm);
    }
    for (i, (s, op)) in input_specs.iter().zip(operands.iter()).enumerate() {
        if s.len() != op.ndim() {
            return Err(vm.new_value_error(format!(
                "einsum: operand {} ndim ({}) != subscript count ({})",
                i,
                op.ndim(),
                s.len()
            )));
        }
    }

    // Promote everything to a common dtype, then cast to that dtype. We work
    // in F64 for non-complex types and C128 for complex.
    let promoted = promote_many(&operands.iter().map(|a| a.dtype()).collect::<Vec<_>>());
    let work_dt = if promoted.is_complex() {
        DType::C128
    } else {
        DType::F64
    };

    // Collect the size for every distinct label.
    let mut label_size: std::collections::BTreeMap<char, usize> =
        std::collections::BTreeMap::new();
    for (s, op) in input_specs.iter().zip(operands.iter()) {
        for (j, &lbl) in s.iter().enumerate() {
            let dim = op.shape()[j];
            match label_size.get(&lbl) {
                Some(existing) if *existing != dim => {
                    return Err(vm.new_value_error(format!(
                        "einsum: label '{}' has conflicting sizes {} vs {}",
                        lbl, existing, dim
                    )));
                }
                _ => {
                    label_size.insert(lbl, dim);
                }
            }
        }
    }

    // Resolve output labels — explicit if given, otherwise labels that
    // appear exactly once across inputs, in alphabetical order.
    let resolved_output: Vec<char> = match output_spec {
        Some(o) => o,
        None => {
            let mut counts: std::collections::BTreeMap<char, usize> =
                std::collections::BTreeMap::new();
            for s in &input_specs {
                for &c in s {
                    *counts.entry(c).or_insert(0) += 1;
                }
            }
            counts
                .iter()
                .filter(|(_, c)| **c == 1)
                .map(|(l, _)| *l)
                .collect()
        }
    };

    // Distinct labels in deterministic order — we iterate over their full
    // cartesian product.
    let all_labels: Vec<char> = label_size.keys().copied().collect();
    let dims: Vec<usize> = all_labels.iter().map(|l| label_size[l]).collect();

    // Output shape.
    let out_shape: Vec<usize> = resolved_output.iter().map(|l| label_size[l]).collect();
    let out_n: usize = out_shape.iter().product::<usize>().max(1);

    match work_dt {
        DType::F64 => {
            let arrs: Vec<ArrayD<f64>> = operands.iter().map(|o| o.coerce::<f64>()).collect();
            let acc = run_einsum_f64(
                &arrs,
                &input_specs,
                &resolved_output,
                &all_labels,
                &dims,
                out_n,
            );
            let shape = if out_shape.is_empty() {
                IxDyn(&[])
            } else {
                IxDyn(&out_shape)
            };
            if out_shape.is_empty() {
                Ok(ArraysD::F64(ArrayD::from_elem(shape, acc[0])))
            } else {
                Ok(ArraysD::F64(ArrayD::from_shape_vec(shape, acc).unwrap_or_default()))
            }
        }
        DType::C128 => {
            let arrs: Vec<ArrayD<num_complex::Complex<f64>>> = operands
                .iter()
                .map(|o| o.coerce::<num_complex::Complex<f64>>())
                .collect();
            let acc = run_einsum_c128(
                &arrs,
                &input_specs,
                &resolved_output,
                &all_labels,
                &dims,
                out_n,
            );
            let shape = if out_shape.is_empty() {
                IxDyn(&[])
            } else {
                IxDyn(&out_shape)
            };
            if out_shape.is_empty() {
                Ok(ArraysD::C128(ArrayD::from_elem(shape, acc[0])))
            } else {
                Ok(ArraysD::C128(
                    ArrayD::from_shape_vec(shape, acc).unwrap_or_default(),
                ))
            }
        }
        // promote_many only emits {F64, C128} for the work dtype; any other
        // value is a logic bug, surface it as a clean Python error.
        _ => Err(internal(vm, "einsum: unexpected work dtype")),
    }
}

/// Greedy multi-operand contraction. Each iteration picks the pair whose
/// contraction yields the smallest intermediate (by product of remaining
/// labels), then folds them into a synthetic two-operand einsum.
fn einsum_greedy(
    mut specs: Vec<Vec<char>>,
    output: Option<Vec<char>>,
    operands: &[ArraysD],
    vm: &VirtualMachine,
) -> PyResult<ArraysD> {
    let mut arrs: Vec<ArraysD> = operands.to_vec();
    // Collect label → size; needed to estimate intermediate sizes.
    let mut sizes: std::collections::BTreeMap<char, usize> =
        std::collections::BTreeMap::new();
    for (s, a) in specs.iter().zip(arrs.iter()) {
        for (j, &lbl) in s.iter().enumerate() {
            sizes.insert(lbl, a.shape()[j]);
        }
    }
    let final_output: Vec<char> = match output {
        Some(o) => o,
        None => {
            // numpy: labels appearing exactly once across inputs, sorted.
            let mut counts: std::collections::BTreeMap<char, usize> =
                std::collections::BTreeMap::new();
            for s in &specs {
                for &c in s {
                    *counts.entry(c).or_insert(0) += 1;
                }
            }
            counts
                .iter()
                .filter(|(_, c)| **c == 1)
                .map(|(l, _)| *l)
                .collect()
        }
    };

    while specs.len() > 1 {
        // Choose the cheapest pair to contract.
        let mut best = (0usize, 1usize);
        let mut best_cost = usize::MAX;
        for i in 0..specs.len() {
            for j in (i + 1)..specs.len() {
                let cost = pair_cost(&specs[i], &specs[j], &specs, &final_output, &sizes);
                if cost < best_cost {
                    best_cost = cost;
                    best = (i, j);
                }
            }
        }
        let (i, j) = best;
        // Intermediate output labels: labels in (specs[i] ∪ specs[j]) that
        // appear elsewhere OR are in the final output.
        let mut used_elsewhere: std::collections::BTreeSet<char> =
            std::collections::BTreeSet::new();
        for (k, s) in specs.iter().enumerate() {
            if k == i || k == j {
                continue;
            }
            for &c in s {
                used_elsewhere.insert(c);
            }
        }
        for &c in &final_output {
            used_elsewhere.insert(c);
        }
        let mut interim_labels: Vec<char> = Vec::new();
        let mut seen = std::collections::BTreeSet::new();
        for &c in specs[i].iter().chain(specs[j].iter()) {
            if used_elsewhere.contains(&c) && seen.insert(c) {
                interim_labels.push(c);
            }
        }
        // Build the sub-spec for the pair and run.
        let pair_spec = format!(
            "{},{}->{}",
            specs[i].iter().collect::<String>(),
            specs[j].iter().collect::<String>(),
            interim_labels.iter().collect::<String>(),
        );
        let pair_result = einsum(&pair_spec, &[arrs[i].clone(), arrs[j].clone()], vm)?;
        // Remove i and j from the working set, then push the intermediate.
        let (lo, hi) = if i < j { (i, j) } else { (j, i) };
        arrs.remove(hi);
        arrs.remove(lo);
        specs.remove(hi);
        specs.remove(lo);
        arrs.push(pair_result);
        specs.push(interim_labels);
    }
    // Final pass to project the last operand to the desired output.
    let final_spec = format!(
        "{}->{}",
        specs[0].iter().collect::<String>(),
        final_output.iter().collect::<String>(),
    );
    einsum(&final_spec, &arrs, vm)
}

fn pair_cost(
    a: &[char],
    b: &[char],
    all: &[Vec<char>],
    out: &[char],
    sizes: &std::collections::BTreeMap<char, usize>,
) -> usize {
    // Cost ≈ product of dims of the surviving labels (those used elsewhere
    // or in the final output) plus contracted dims.
    let mut surviving = std::collections::BTreeSet::new();
    let mut elsewhere = std::collections::BTreeSet::new();
    for s in all {
        if std::ptr::eq(s.as_ptr(), a.as_ptr()) || std::ptr::eq(s.as_ptr(), b.as_ptr()) {
            continue;
        }
        for &c in s {
            elsewhere.insert(c);
        }
    }
    for &c in out {
        elsewhere.insert(c);
    }
    for &c in a.iter().chain(b.iter()) {
        if elsewhere.contains(&c) {
            surviving.insert(c);
        }
    }
    // Multiply all distinct dims that participate in this contraction.
    let union: std::collections::BTreeSet<char> = a.iter().chain(b.iter()).copied().collect();
    union
        .iter()
        .map(|c| sizes.get(c).copied().unwrap_or(1))
        .product::<usize>()
        .saturating_add(surviving.iter().map(|c| sizes[c]).product::<usize>())
}

/// Parsed einsum spec: per-input subscript lists and an optional output list.
type EinsumSpec = (Vec<Vec<char>>, Option<Vec<char>>);

fn parse_spec(spec: &str, vm: &VirtualMachine) -> PyResult<EinsumSpec> {
    let spec = spec.replace(' ', "");
    let (lhs, rhs) = if let Some(idx) = spec.find("->") {
        (&spec[..idx], Some(&spec[idx + 2..]))
    } else {
        (spec.as_str(), None)
    };
    let mut inputs = Vec::new();
    for part in lhs.split(',') {
        let chars: Vec<char> = part.chars().collect();
        for &c in &chars {
            if !c.is_ascii_alphabetic() {
                return Err(vm.new_value_error(format!(
                    "einsum: subscript characters must be ASCII letters, got '{c}'"
                )));
            }
        }
        inputs.push(chars);
    }
    let output = rhs.map(|r| r.chars().collect::<Vec<char>>());
    Ok((inputs, output))
}

fn run_einsum_f64(
    arrs: &[ArrayD<f64>],
    input_specs: &[Vec<char>],
    out_spec: &[char],
    all_labels: &[char],
    dims: &[usize],
    out_n: usize,
) -> Vec<f64> {
    let mut out = vec![0.0f64; out_n];
    let n_labels = all_labels.len();
    let mut idx = vec![0usize; n_labels];
    let label_index: std::collections::HashMap<char, usize> = all_labels
        .iter()
        .enumerate()
        .map(|(i, &c)| (c, i))
        .collect();
    // Precompute per-operand the index-positions in `idx` for each axis.
    let operand_idx_paths: Vec<Vec<usize>> = input_specs
        .iter()
        .map(|s| s.iter().map(|c| label_index[c]).collect())
        .collect();
    let out_idx_path: Vec<usize> = out_spec.iter().map(|c| label_index[c]).collect();
    let out_strides: Vec<usize> = {
        let mut s = vec![1usize; out_spec.len()];
        for i in (0..out_spec.len().saturating_sub(1)).rev() {
            s[i] = s[i + 1] * dims[out_idx_path[i + 1]];
        }
        s
    };

    loop {
        let mut product = 1.0f64;
        for (op_a, path) in arrs.iter().zip(operand_idx_paths.iter()) {
            let coord: Vec<usize> = path.iter().map(|&p| idx[p]).collect();
            product *= op_a[IxDyn(&coord)];
        }
        // Compute output flat offset.
        let flat = if out_spec.is_empty() {
            0
        } else {
            out_idx_path
                .iter()
                .zip(out_strides.iter())
                .map(|(&p, &s)| idx[p] * s)
                .sum()
        };
        out[flat] += product;

        // Advance the cartesian product.
        let mut k = n_labels;
        while k > 0 {
            k -= 1;
            idx[k] += 1;
            if idx[k] < dims[k] {
                break;
            }
            idx[k] = 0;
            if k == 0 {
                return out;
            }
        }
        if dims.is_empty() {
            return out;
        }
    }
}

fn run_einsum_c128(
    arrs: &[ArrayD<num_complex::Complex<f64>>],
    input_specs: &[Vec<char>],
    out_spec: &[char],
    all_labels: &[char],
    dims: &[usize],
    out_n: usize,
) -> Vec<num_complex::Complex<f64>> {
    let mut out = vec![num_complex::Complex::<f64>::new(0.0, 0.0); out_n];
    let n_labels = all_labels.len();
    let mut idx = vec![0usize; n_labels];
    let label_index: std::collections::HashMap<char, usize> = all_labels
        .iter()
        .enumerate()
        .map(|(i, &c)| (c, i))
        .collect();
    let operand_idx_paths: Vec<Vec<usize>> = input_specs
        .iter()
        .map(|s| s.iter().map(|c| label_index[c]).collect())
        .collect();
    let out_idx_path: Vec<usize> = out_spec.iter().map(|c| label_index[c]).collect();
    let out_strides: Vec<usize> = {
        let mut s = vec![1usize; out_spec.len()];
        for i in (0..out_spec.len().saturating_sub(1)).rev() {
            s[i] = s[i + 1] * dims[out_idx_path[i + 1]];
        }
        s
    };
    loop {
        let mut product = num_complex::Complex::<f64>::new(1.0, 0.0);
        for (op_a, path) in arrs.iter().zip(operand_idx_paths.iter()) {
            let coord: Vec<usize> = path.iter().map(|&p| idx[p]).collect();
            product *= op_a[IxDyn(&coord)];
        }
        let flat = if out_spec.is_empty() {
            0
        } else {
            out_idx_path
                .iter()
                .zip(out_strides.iter())
                .map(|(&p, &s)| idx[p] * s)
                .sum()
        };
        out[flat] += product;
        let mut k = n_labels;
        while k > 0 {
            k -= 1;
            idx[k] += 1;
            if idx[k] < dims[k] {
                break;
            }
            idx[k] = 0;
            if k == 0 {
                return out;
            }
        }
        if dims.is_empty() {
            return out;
        }
    }
}
