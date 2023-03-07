use std::fmt::Debug;

use crate::Column;

use super::rv_decomposition::RVDecomposition;

use crossbeam::atomic::AtomicCell;
use pinboard::NonEmptyPinboard;
use rayon::prelude::*;

// Implements do while loop lines 6-9
fn get_col_with_pivot<C: Column + Clone>(
    l: usize,
    matrix: &Vec<NonEmptyPinboard<(C, C)>>,
    pivots: &Vec<AtomicCell<Option<usize>>>,
) -> Option<(usize, (C, C))> {
    loop {
        let piv = pivots[l].load();
        if let Some(piv) = piv {
            let cols = matrix[piv].read();
            if cols.0.pivot() != Some(l) {
                // Got a column but it now has the wrong pivot; loop again.
                continue;
            };
            // Get column with correct pivot, return to caller.
            return Some((piv, cols));
        } else {
            // There is not yet a column with this pivot, inform caller.
            return None;
        }
    }
}

fn reduce_column<C: Column>(
    j: usize,
    matrix: &Vec<NonEmptyPinboard<(C, C)>>,
    pivots: &Vec<AtomicCell<Option<usize>>>,
) {
    let mut working_j = j;
    'outer: loop {
        let mut curr_column = matrix[working_j].read();
        while let Some(l) = (&curr_column).0.pivot() {
            let piv_with_column_opt = get_col_with_pivot(l, &matrix, &pivots);
            if let Some((piv, piv_column)) = piv_with_column_opt {
                // Lines 17-24
                if piv < working_j {
                    curr_column.0.add_col(&piv_column.0);
                    curr_column.1.add_col(&piv_column.1);
                } else if piv > working_j {
                    matrix[working_j].set(curr_column);
                    if pivots[l]
                        .compare_exchange(Some(piv), Some(working_j))
                        .is_ok()
                    {
                        working_j = piv;
                    }
                    continue 'outer;
                } else {
                    panic!()
                }
            } else {
                // piv = -1 case
                matrix[working_j].set(curr_column);
                if pivots[l].compare_exchange(None, Some(working_j)).is_ok() {
                    return;
                } else {
                    continue 'outer;
                }
            }
        }
        // Lines 25-27 (curr_column = 0 clause)
        if (&curr_column.0).pivot().is_none() {
            matrix[working_j].set(curr_column);
            return;
        }
    }
}
pub fn rv_decompose_lock_free<C: Column + Debug + 'static>(
    matrix: impl ExactSizeIterator<Item = C>,
) -> RVDecomposition<C> {
    let matrix_len = matrix.len();
    // Step 0: Setup storage for pivots vector
    let pivots: Vec<_> = (0..matrix_len).map(|_| AtomicCell::new(None)).collect();
    // Step 1: Setup a vector of atomic pointers to (r_col, v_col) pairs
    let matrix: Vec<_> = matrix
        .enumerate()
        .map(|(idx, r_col)| {
            let mut v_col = C::default();
            v_col.add_entry(idx);
            NonEmptyPinboard::new((r_col, v_col))
        })
        .collect();
    // Reduce matrix
    // TODO: Can we advice rayon to split work in chunks?
    (0..matrix_len)
        .into_par_iter()
        .for_each(|j| reduce_column(j, &matrix, &pivots));
    // Wrap into RV decomposition
    let (r, v) = matrix.into_iter().map(|pinboard| pinboard.read()).unzip();
    RVDecomposition { r, v }
}
