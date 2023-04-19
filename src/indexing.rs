use lophat::{
    algorithms::RVDecomposition,
    columns::{Column, VecColumn},
    utils::PersistenceDiagram,
};

pub trait ReordorableColumn: Send + Sync + Clone + Default {
    fn reorder_rows(&mut self, mapping: &impl IndexMapping);
    fn unreorder_rows(&mut self, mapping: &impl IndexMapping);
}
#[derive(Debug, Default, Clone)]
pub struct AnnotatedColumn<T> {
    pub col: T,
    pub in_g: bool,
}

impl ReordorableColumn for VecColumn {
    // TODO: Reimplement so that this happens in-place?
    fn reorder_rows(&mut self, mapping: &impl IndexMapping) {
        let mut new_col: Vec<usize> = self
            .entries()
            .filter_map(|row_idx| mapping.map(row_idx))
            .collect();
        new_col.sort();
        self.set_entries(new_col);
    }

    // TODO: Reimplement so that this happens in-place?
    fn unreorder_rows(&mut self, mapping: &impl IndexMapping) {
        let mut new_col: Vec<usize> = self
            .entries()
            .filter_map(|row_idx| mapping.inverse_map(row_idx))
            .collect();
        new_col.sort();
        self.set_entries(new_col);
    }
}

pub trait IndexMapping {
    fn map(&self, index: usize) -> Option<usize>;
    fn inverse_map(&self, index: usize) -> Option<usize>;
}

#[derive(Debug)]
pub struct VectorMapping {
    internal: Vec<Option<usize>>,
    internal_inverse: Option<Vec<usize>>,
}

impl IndexMapping for VectorMapping {
    fn map(&self, index: usize) -> Option<usize> {
        self.internal[index]
    }

    fn inverse_map(&self, index: usize) -> Option<usize> {
        let inv = self.internal_inverse.as_ref().unwrap();
        Some(inv[index])
    }
}

pub fn compute_l_first_mapping(matrix: &Vec<AnnotatedColumn<VecColumn>>) -> VectorMapping {
    let total_size = matrix.len();
    let num_in_g = matrix.iter().filter(|col| col.in_g).count();
    let mut next_g_index = 0;
    let mut next_f_index = num_in_g;
    let mut mapping = Vec::with_capacity(total_size);
    let mut inverse_mapping = vec![0; total_size];
    for col in matrix {
        if col.in_g {
            inverse_mapping[next_g_index] = mapping.len();
            mapping.push(Some(next_g_index));
            next_g_index += 1;
        } else {
            inverse_mapping[next_f_index] = mapping.len();
            mapping.push(Some(next_f_index));
            next_f_index += 1
        }
    }
    VectorMapping {
        internal: mapping,
        internal_inverse: Some(inverse_mapping),
    }
}

pub fn build_kernel_mapping<Algo: RVDecomposition<VecColumn>>(
    dim_decomposition: &Algo,
) -> VectorMapping {
    let mut counter = 0;
    let mut idx_list: Vec<Option<usize>> = vec![];
    let r_col_iter = (0..dim_decomposition.n_cols()).map(|idx| dim_decomposition.get_r_col(idx));
    for r_col in r_col_iter {
        if r_col.pivot().is_none() {
            idx_list.push(Some(counter));
            counter += 1;
        } else {
            idx_list.push(None);
        }
    }
    VectorMapping {
        internal: idx_list,
        internal_inverse: None,
    }
}

pub fn build_rel_mapping(
    matrix: &Vec<VecColumn>,
    g_elements: &Vec<bool>,
    size_of_l: usize,
    size_of_k: usize,
) -> (VectorMapping, usize) {
    let mut l_index: Option<usize> = None;
    // Keeps track of next index in new ordering
    // We only increment counter when we encounter a cell not in L or the first L-cell
    let mut counter = 0;
    let size_of_quo = size_of_k - size_of_l + 1;
    let mut idx_list: Vec<Option<usize>> = vec![];
    let mut inverse_mapping = vec![0; size_of_quo];
    for (idx, (r_col, &in_g)) in matrix.iter().zip(g_elements.iter()).enumerate() {
        if in_g {
            if l_index.is_none() {
                l_index = Some(counter);
                // Counter should idx up to now
                inverse_mapping[counter] = idx;
                counter += 1;
            }
            if r_col.dimension() == 0 {
                // This is a vertex in L, should get_mapped to l_index
                idx_list.push(l_index)
            } else {
                // This is a higher-dimensional cell
                idx_list.push(None)
            }
        } else {
            idx_list.push(Some(counter));
            inverse_mapping[counter] = idx;
            counter += 1;
        }
    }
    (
        VectorMapping {
            internal: idx_list,
            internal_inverse: Some(inverse_mapping),
        },
        l_index.unwrap(),
    )
}

pub fn unreorder_idxs(diagram: &mut PersistenceDiagram, mapping: &impl IndexMapping) {
    diagram.unpaired = diagram
        .unpaired
        .iter()
        .cloned()
        .map(|idx| mapping.inverse_map(idx).unwrap())
        .collect();
    diagram.paired = diagram
        .paired
        .iter()
        .cloned()
        .map(|(b_idx, d_idx)| {
            (
                mapping.inverse_map(b_idx).unwrap(),
                mapping.inverse_map(d_idx).unwrap(),
            )
        })
        .collect();
}
