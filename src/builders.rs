//! Construction of the six reduction matrices for an inclusion.
//! Low-level builders assume valid filtered F2 chain complexes and consistent
//! domain-first permutations preserving order within domain and complement.

use itertools::Itertools;
use lophat::{
    algorithms::Decomposition,
    columns::{Column, VecColumn},
};

use crate::indexing::{DensePermutation, Permutation, PermuteItems};

/// Build the domain boundary matrix in local domain coordinates.
///
/// `cols_in_dom` must be sorted, distinct, in range, and boundary-closed.
/// The permutation maps these columns, in order, to the initial index block.
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

/// Build the image reduction matrix by permuting boundary rows only.
///
/// Columns retain the original codomain order. The domain-first permutation must
/// preserve relative order within the domain and its complement. This matrix
/// need not square to zero and must be reduced without clearing.
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
/// Build the quotient boundary matrix B/A, with no extra basepoint.
///
/// The first `sz_domain` permuted indices must be exactly the boundary-closed
/// domain. Remove those rows and columns and subtract `sz_domain` from remaining
/// row indices. Local quotient index i corresponds to inverse_map(i + sz_domain).
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

/// Build the kernel reduction matrix from the cycle columns of the image matrix.
///
/// Select V columns whose reduced R column is zero, in original column order,
/// then permute their rows using the same domain-first mapping. This rectangular
/// matrix has codomain height and must be reduced without clearing.
///
/// # Panics
/// Panics if the image decomposition did not retain V.
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

/// Build the cokernel reduction matrix in original codomain coordinates.
///
/// Replace each domain cycle column by its domain V column, translating local
/// domain rows back to original indices; retain other boundary columns.
/// The decomposition and permutation must describe the same domain. Reduce the
/// result without clearing.
///
/// # Panics
/// Panics if required V columns were not retained.
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

/// Iterate over column indices with zero reduced R columns, in ascending order.
/// This tests reduced columns, not whether the corresponding original boundary is zero.
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
