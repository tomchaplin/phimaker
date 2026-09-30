use itertools::Itertools;
use lophat::{
    algorithms::Decomposition,
    columns::{Column, VecColumn},
};

use crate::indexing::{DensePermutation, Permutation, PermuteItems};

pub fn build_d_dom(
    d_cod: &[VecColumn],
    cols_in_dom: &[usize],
    dom_first_permutation: &impl Permutation,
) -> impl Iterator<Item = VecColumn> {
    cols_in_dom.iter().map(|&idx| {
        let col = &d_cod[idx];
        let dimension = col.dimension();
        let column = col.entries().permuted(dom_first_permutation).collect_vec();
        VecColumn::from((dimension, column))
    })
}

pub fn build_d_im(
    d_cod: &[VecColumn],
    dom_first_permutation: &impl Permutation,
) -> impl Iterator<Item = VecColumn> {
    d_cod.iter().map(|col| {
        VecColumn::from((
            col.dimension(),
            col.entries().permuted(dom_first_permutation).collect_vec(),
        ))
    })
}
pub fn build_d_rel(
    d_cod: &[VecColumn],
    dom_first_permutation: &impl Permutation,
    sz_domain: usize,
) -> impl Iterator<Item = VecColumn> {
    let sz_codomain = d_cod.len();
    (sz_domain..sz_codomain).map(move |idx_dom_first| {
        let idx = dom_first_permutation.inverse_map(idx_dom_first);
        let col = &d_cod[idx];
        VecColumn::from((
            col.dimension(),
            col.entries()
                .filter_map(|row_idx| {
                    let row_idx_dom_first = dom_first_permutation.map(row_idx);
                    if row_idx_dom_first < sz_domain {
                        None
                    } else {
                        Some(row_idx_dom_first - sz_domain)
                    }
                })
                .sorted()
                .collect_vec(),
        ))
    })
}

pub fn build_d_ker<Algo: Decomposition<VecColumn>>(
    d_im_decomposition: &Algo,
    mapping: &impl Permutation,
) -> impl Iterator<Item = VecColumn> {
    decomp_cycle_idxs(d_im_decomposition).map(|idx| {
        let v_col = d_im_decomposition.get_v_col(idx).unwrap();
        VecColumn::from((
            v_col.dimension(),
            v_col.entries().permuted(mapping).collect_vec(),
        ))
    })
}

pub fn build_d_cok<Algo: Decomposition<VecColumn>>(
    d_cod: &[VecColumn],
    d_dom_decomp: &Algo,
    dom_first_mapping: &impl Permutation,
) -> impl Iterator<Item = VecColumn> {
    let sz_domain = d_dom_decomp.n_cols();
    (0..d_cod.len()).map(move |idx| {
        let idx_dom_first = dom_first_mapping.map(idx);
        let col_in_dom = idx_dom_first < sz_domain;
        if col_in_dom && d_dom_decomp.get_r_col(idx_dom_first).is_cycle() {
            VecColumn::from((
                d_cod[idx].dimension(),
                d_dom_decomp
                    .get_v_col(idx_dom_first)
                    .unwrap()
                    .entries()
                    .unpermuted(dom_first_mapping)
                    .collect_vec(),
            ))
        } else {
            d_cod[idx].clone()
        }
    })
}

/// Iterate over indices containing cycles of d_im, in sorted order.
pub fn decomp_cycle_idxs<Algo: Decomposition<VecColumn>>(
    d_im_decomposition: &Algo,
) -> impl Iterator<Item = usize> {
    (0..d_im_decomposition.n_cols()).filter(|&idx| {
        let r_col = d_im_decomposition.get_r_col(idx);
        r_col.is_cycle()
    })
}

/// Permutes row indices so that all rows of the domain appear before other rows.
pub(crate) fn compute_dom_first_permutation(
    total_size: usize,
    cols_in_dom: &[usize],
) -> DensePermutation {
    let cols_in_dom = cols_in_dom.iter().copied().sorted().collect::<Vec<_>>();
    let num_in_domain = cols_in_dom.len();
    let mut next_domain_idx = 0;
    let mut next_non_domain_idx = num_in_domain;
    let mut perm = vec![0; total_size];
    (0..total_size).for_each(|idx| {
        if next_domain_idx < num_in_domain && cols_in_dom[next_domain_idx] == idx {
            perm[idx] = next_domain_idx;
            next_domain_idx += 1;
        } else {
            perm[idx] = next_non_domain_idx;
            next_non_domain_idx += 1
        }
    });
    DensePermutation::new(perm)
}
