use std::collections::HashMap;

use crate::Column;
use crate::PersistenceDiagram;

#[derive(Debug, Default)]
pub struct RVDecomposition<C: Column> {
    pub r: Vec<C>,
    pub v: Vec<C>,
}

impl<C: Column> RVDecomposition<C> {
    fn col_idx_with_same_low(&self, col: &C, low_inverse: &HashMap<usize, usize>) -> Option<usize> {
        let pivot = col.pivot()?;
        low_inverse.get(&pivot).copied()
    }
    // Receives column, reduces it with left-to-right addition from R
    // Adds reduction to self
    fn reduce_column(&mut self, mut column: C, low_inverse: &mut HashMap<usize, usize>) {
        // v_col tracks how the final reduced column is built up
        // Currently column contains 1 lot of the latest column in D
        let mut v_col = C::default();
        v_col.add_entry(self.r.len());
        // Reduce the column, keeping track of how we do this in V
        while let Some(col_idx) = self.col_idx_with_same_low(&column, &low_inverse) {
            column.add_col(&self.r[col_idx]);
            v_col.add_col(&self.v[col_idx]);
        }
        // Update low inverse
        let final_pivot = column.pivot();
        if let Some(final_pivot) = final_pivot {
            // This column has a lowest 1 and is being inserted at the end of R
            low_inverse.insert(final_pivot, self.r.len());
        }
        // Push to decomposition
        self.r.push(column);
        self.v.push(v_col);
    }

    pub fn diagram(&self) -> PersistenceDiagram {
        let mut diagram = PersistenceDiagram::default();
        for (idx, col) in self.r.iter().enumerate() {
            if let Some(lowest_idx) = col.pivot() {
                // Negative column
                diagram.unpaired.remove(&lowest_idx);
                diagram.paired.push((lowest_idx, idx))
            } else {
                // Positive column
                diagram.unpaired.insert(idx);
            }
        }
        diagram
    }
}

pub fn rv_decompose<C: Column>(matrix: Vec<C>) -> RVDecomposition<C> {
    let mut low_inverse = HashMap::new();
    matrix
        .into_iter()
        .fold(RVDecomposition::default(), |mut accum, next_col| {
            accum.reduce_column(next_col, &mut low_inverse);
            accum
        })
}
