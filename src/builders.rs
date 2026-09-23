use lophat::{
    algorithms::Decomposition,
    columns::{Column, VecColumn},
};

use crate::indexing::{IndexMapping, ReordorableColumn, VectorMapping};

pub fn extract_columns<'a>(
    matrix: &'a [VecColumn],
    extract: &'a [bool],
) -> impl Iterator<Item = VecColumn> + 'a {
    matrix
        .iter()
        .zip(extract.iter())
        .filter(|(_, in_dom)| **in_dom)
        .map(|(col, _)| col)
        .cloned()
}

pub fn build_d_dom<'a>(
    d_cod: &'a [VecColumn],
    col_in_dom: &'a [bool],
    dom_first_mapping: &'a VectorMapping,
) -> impl Iterator<Item = VecColumn> + 'a {
    extract_columns(d_cod, col_in_dom).map(|mut col| {
        col.reorder_rows(dom_first_mapping);
        col
    })
}

pub fn build_d_im<'a>(
    d_cod: &'a [VecColumn],
    mapping: &'a impl IndexMapping,
) -> impl Iterator<Item = VecColumn> + 'a {
    d_cod.iter().cloned().map(|mut col| {
        col.reorder_rows(mapping);
        col
    })
}
// WARNING: This functions makes the following assumption:
// If the boundary of a cell is entirely contained in L then that cell is in L
// This ensures that a 1-cell not in L can have at most 1 vertex in L
// This makes it easier to map the boundary
// Also inherits assumption from build_rel_mapping
pub fn build_d_rel<'a>(
    df: &'a [VecColumn],
    g_elements: &'a [bool],
    rel_mapping: &'a VectorMapping,
    l_index: usize,
) -> impl Iterator<Item = VecColumn> + 'a {
    df.iter()
        .zip(g_elements.iter())
        .enumerate()
        .filter_map(move |(idx, (col, &in_g))| {
            if in_g && idx != l_index {
                None
            } else {
                let mut new_col = col.clone();
                new_col.reorder_rows(rel_mapping);
                Some(new_col)
            }
        })
}

pub fn build_d_ker<'a, Algo: Decomposition<VecColumn>>(
    d_im_decomposition: &'a Algo,
    mapping: &'a impl IndexMapping,
) -> impl Iterator<Item = VecColumn> + 'a {
    let paired_cols = (0..d_im_decomposition.n_cols()).map(|idx| {
        (
            d_im_decomposition.get_r_col(idx),
            d_im_decomposition.get_v_col(idx).unwrap(),
        )
    });
    paired_cols.filter_map(|(r_col, v_col)| {
        if r_col.pivot().is_none() {
            // If r_col is zero then v_col stores a cycle
            // We should add it to dker with the elements of L appearing first
            let mut new_col = v_col.clone();
            new_col.reorder_rows(mapping);
            Some(new_col)
        } else {
            // Filter this column out
            None
        }
    })
}

pub fn build_d_cok<'a, Algo: Decomposition<VecColumn>>(
    d_cod: &'a [VecColumn],
    d_dom_decomp: &'a Algo,
    col_in_dom: &'a [bool],
    dom_first_mapping: &'a impl IndexMapping,
) -> impl Iterator<Item = VecColumn> + 'a {
    (0..d_cod.len()).map(|col_idx| {
        if col_in_dom[col_idx] {
            let idx_in_d_dom = dom_first_mapping.map(col_idx).unwrap();
            let d_dom_rcol = &d_dom_decomp.get_r_col(idx_in_d_dom);
            if d_dom_rcol.pivot().is_none() {
                let mut next_col = d_dom_decomp.get_v_col(idx_in_d_dom).unwrap().clone();
                // Convert from L simplices first back to default order
                next_col.unreorder_rows(dom_first_mapping);
                next_col
            } else {
                d_cod[col_idx].clone()
            }
        } else {
            d_cod[col_idx].clone()
        }
    })
}
