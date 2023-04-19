use lophat::{
    algorithms::RVDecomposition,
    columns::{Column, VecColumn},
};

use crate::indexing::{IndexMapping, ReordorableColumn, VectorMapping};

pub fn extract_columns<'a>(
    matrix: &'a Vec<VecColumn>,
    extract: &'a Vec<bool>,
) -> impl Iterator<Item = VecColumn> + 'a {
    matrix
        .iter()
        .zip(extract.iter())
        .filter(|(_, in_g)| **in_g)
        .map(|(col, _)| col)
        .cloned()
}

pub fn build_dg<'a>(
    df: &'a Vec<VecColumn>,
    g_elements: &'a Vec<bool>,
    l_first_mapping: &'a VectorMapping,
) -> impl Iterator<Item = VecColumn> + 'a {
    extract_columns(df, g_elements).map(|mut col| {
        col.reorder_rows(l_first_mapping);
        col
    })
}

pub fn build_dim<'a>(
    df: &'a Vec<VecColumn>,
    mapping: &'a impl IndexMapping,
) -> impl Iterator<Item = VecColumn> + 'a {
    df.iter().cloned().map(|mut col| {
        col.reorder_rows(mapping);
        col
    })
}
// WARNING: This functions makes the following assumption:
// If the boundary of a cell is entirely contained in L then that cell is in L
// This ensures that a 1-cell not in L can have at most 1 vertex in L
// This makes it easier to map the boundary
// Also inherits assumption from build_rel_mapping
pub fn build_drel<'a>(
    df: &'a Vec<VecColumn>,
    g_elements: &'a Vec<bool>,
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

pub fn build_dker<'a, Algo: RVDecomposition<VecColumn>>(
    dim_decomposition: &'a Algo,
    mapping: &'a impl IndexMapping,
) -> impl Iterator<Item = VecColumn> + 'a {
    let paired_cols = (0..dim_decomposition.n_cols()).map(|idx| {
        (
            dim_decomposition.get_r_col(idx),
            dim_decomposition.get_v_col(idx).unwrap(),
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

pub fn build_dcok<'a, Algo: RVDecomposition<VecColumn>>(
    df: &'a Vec<VecColumn>,
    dg_decomposition: &'a Algo,
    g_elements: &'a Vec<bool>,
    mapping: &'a impl IndexMapping,
) -> impl Iterator<Item = VecColumn> + 'a {
    (0..df.len()).map(|col_idx| {
        let col_in_g = g_elements[col_idx];
        if col_in_g {
            let idx_in_dg = mapping.map(col_idx).unwrap();
            let dg_rcol = &dg_decomposition.get_r_col(idx_in_dg);
            if dg_rcol.pivot().is_none() {
                let mut next_col = dg_decomposition.get_v_col(idx_in_dg).unwrap().clone();
                // Convert from L simplices first back to default order
                next_col.unreorder_rows(mapping);
                next_col
            } else {
                df[col_idx].clone()
            }
        } else {
            df[col_idx].clone()
        }
    })
}
