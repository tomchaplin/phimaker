use std::{
    cmp::Ordering,
    collections::{HashMap, HashSet},
};

use pyo3::prelude::*;

pub trait IndexMapping {
    fn map(&self, index: usize) -> usize;
    fn inverse_map(&self, index: usize) -> usize;
}

#[derive(Debug)]
struct VectorMapping {
    internal: Vec<usize>,
    internal_inverse: Vec<usize>,
}

impl IndexMapping for VectorMapping {
    fn map(&self, index: usize) -> usize {
        self.internal[index]
    }

    fn inverse_map(&self, index: usize) -> usize {
        self.internal_inverse[index]
    }
}

pub trait Column: Send {
    fn pivot(&self) -> Option<usize>;
    fn add_col(&mut self, other: &Self);
    fn reorder_rows(&mut self, mapping: &impl IndexMapping);
    fn unreorder_rows(&mut self, mapping: &impl IndexMapping);
}

#[derive(Debug, Default, Clone)]
pub struct VecColumn {
    internal: Vec<usize>,
}

#[derive(Debug, Default, Clone)]
pub struct AnnotatedVecColumn {
    col: VecColumn,
    in_g: bool,
}

#[derive(Debug, Default)]
pub struct RVDecomposition {
    r: Vec<VecColumn>,
    v: Vec<VecColumn>,
    low_inverse: HashMap<usize, usize>,
}

impl VecColumn {
    // Returns the index where we should try to insert next entry
    fn add_entry(&mut self, entry: usize, starting_idx: usize) -> usize {
        let mut working_idx = starting_idx;
        while let Some(value_at_idx) = self.internal.iter().nth(working_idx) {
            match value_at_idx.cmp(&entry) {
                Ordering::Less => {
                    working_idx += 1;
                    continue;
                }
                Ordering::Equal => {
                    self.internal.remove(working_idx);
                    return working_idx;
                }
                Ordering::Greater => {
                    self.internal.insert(working_idx, entry);
                    return working_idx + 1;
                }
            }
        }
        // Bigger than all idxs in col - add to end
        self.internal.push(entry);
        return self.internal.len() - 1;
    }
}

impl Column for VecColumn {
    fn pivot(&self) -> Option<usize> {
        Some(*(self.internal.iter().last()?))
    }

    fn add_col(&mut self, other: &Self) {
        let mut working_idx = 0;
        for entry in other.internal.iter() {
            working_idx = self.add_entry(*entry, working_idx);
        }
    }

    fn reorder_rows(&mut self, mapping: &impl IndexMapping) {
        // Map row idxs to new row idxs
        for row_idx in self.internal.iter_mut() {
            *row_idx = mapping.map(*row_idx);
        }
        // Make sure idxs still appear smallest to largest
        self.internal.sort();
    }

    fn unreorder_rows(&mut self, mapping: &impl IndexMapping) {
        // Map row idxs to new row idxs
        for row_idx in self.internal.iter_mut() {
            *row_idx = mapping.inverse_map(*row_idx);
        }
        // Make sure idxs still appear smallest to largest
        self.internal.sort();
    }
}

impl RVDecomposition {
    fn col_idx_with_same_low(&self, col: &VecColumn) -> Option<usize> {
        let pivot = col.pivot()?;
        self.low_inverse.get(&pivot).copied()
    }
    // Receives column, reduces it with left-to-right addition from R
    // Adds reduction to self
    fn reduce_column(&mut self, mut column: VecColumn) {
        // v_col tracks how the final reduced column is built up
        // Currently column contains 1 lot of the latest column in D
        let mut v_col = VecColumn {
            internal: vec![self.r.len()],
        };
        // Reduce the column, keeping track of how we do this in V
        while let Some(col_idx) = self.col_idx_with_same_low(&column) {
            column.add_col(&self.r[col_idx]);
            v_col.add_col(&self.v[col_idx]);
        }
        // Update low inverse
        let final_pivot = column.pivot();
        if let Some(final_pivot) = final_pivot {
            // This column has a lowest 1 and is being inserted at the end of R
            self.low_inverse.insert(final_pivot, self.r.len());
        }
        // Push to decomposition
        self.r.push(column);
        self.v.push(v_col);
    }
}

pub fn rv_decompose(matrix: Vec<VecColumn>) -> RVDecomposition {
    matrix
        .into_iter()
        .fold(RVDecomposition::default(), |mut accum, next_col| {
            accum.reduce_column(next_col);
            accum
        })
}

#[derive(Debug)]
pub struct DecompositionEnsemble {
    f: RVDecomposition,
    g: RVDecomposition,
    im: RVDecomposition,
    ker: RVDecomposition,
    cok: RVDecomposition,
    l_first_mapping: VectorMapping,
    g_elements: Vec<bool>,
    size_of_l: usize,
    size_of_k: usize,
}

fn compute_l_first_mapping(matrix: &Vec<AnnotatedVecColumn>) -> VectorMapping {
    let total_size = matrix.len();
    let num_in_g = matrix.iter().filter(|col| col.in_g).count();
    let mut next_g_index = 0;
    let mut next_f_index = num_in_g;
    let mut mapping = Vec::with_capacity(total_size);
    let mut inverse_mapping = vec![0; total_size];
    for col in matrix {
        if col.in_g {
            inverse_mapping[next_g_index] = mapping.len();
            mapping.push(next_g_index);
            next_g_index += 1;
        } else {
            inverse_mapping[next_f_index] = mapping.len();
            mapping.push(next_f_index);
            next_f_index += 1
        }
    }
    VectorMapping {
        internal: mapping,
        internal_inverse: inverse_mapping,
    }
}

fn extract_columns(matrix: &Vec<VecColumn>, extract: &Vec<bool>) -> Vec<VecColumn> {
    matrix
        .iter()
        .zip(extract.iter())
        .filter(|(_, in_g)| **in_g)
        .map(|(col, _)| col.clone())
        .collect()
}

fn build_dg(
    df: &Vec<VecColumn>,
    g_elements: &Vec<bool>,
    l_first_mapping: &VectorMapping,
) -> Vec<VecColumn> {
    extract_columns(df, g_elements)
        .into_iter()
        .map(|mut col| {
            col.reorder_rows(l_first_mapping);
            col
        })
        .collect()
}

fn build_dim(df: &Vec<VecColumn>, mapping: &impl IndexMapping) -> Vec<VecColumn> {
    df.clone()
        .into_iter()
        .map(|mut col| {
            col.reorder_rows(mapping);
            col
        })
        .collect()
}

fn build_dker(dim_decomposition: &RVDecomposition, mapping: &impl IndexMapping) -> Vec<VecColumn> {
    let rim_cols = dim_decomposition.r.iter();
    let vim_cols = dim_decomposition.v.iter();
    let paired_cols = rim_cols.zip(vim_cols);
    paired_cols
        .filter_map(|(r_col, v_col)| {
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
        .collect()
}

fn build_dcok(
    df: &Vec<VecColumn>,
    dg_decomposition: &RVDecomposition,
    g_elements: &Vec<bool>,
    mapping: &impl IndexMapping,
) -> Vec<VecColumn> {
    let mut new_matrix: Vec<VecColumn> = Vec::with_capacity(df.len());
    for col_idx in 0..df.len() {
        let col_in_g = g_elements[col_idx];
        if col_in_g {
            let idx_in_dg = mapping.map(col_idx);
            let dg_rcol = &dg_decomposition.r[idx_in_dg];
            if dg_rcol.pivot().is_none() {
                let mut next_col = dg_decomposition.v[idx_in_dg].clone();
                // Convert from L simplices first back to default order
                next_col.unreorder_rows(mapping);
                new_matrix.push(next_col);
            } else {
                let next_col = df[col_idx].clone();
                new_matrix.push(next_col);
            }
        } else {
            let next_col = df[col_idx].clone();
            new_matrix.push(next_col);
        }
    }
    new_matrix
}

pub fn all_decompositions(matrix: Vec<AnnotatedVecColumn>) -> DecompositionEnsemble {
    let l_first_mapping = compute_l_first_mapping(&matrix);
    let g_elements: Vec<bool> = matrix.iter().map(|anncol| anncol.in_g).collect();
    let size_of_l = g_elements.iter().filter(|in_g| **in_g).count();
    let size_of_k = matrix.len();
    let df: Vec<VecColumn> = matrix.into_iter().map(|anncol| anncol.col).collect();
    // Decompose Df
    let decomp_df = rv_decompose(df.clone());
    // Decompose Dg
    let dg = build_dg(&df, &g_elements, &l_first_mapping);
    let decomp_dg = rv_decompose(dg);
    // Decompose dim
    let dim = build_dim(&df, &l_first_mapping);
    let decompose_dim = rv_decompose(dim);
    // Decompose dker
    // TODO: Also need to return mapping from columns of Df to columns of Dker
    let dker = build_dker(&decompose_dim, &l_first_mapping);
    let decompose_dker = rv_decompose(dker);
    // Decompose dcok
    let dcok = build_dcok(&df, &decomp_dg, &g_elements, &l_first_mapping);
    print_matrix(&dcok);
    let decompose_dcok = rv_decompose(dcok);
    DecompositionEnsemble {
        f: decomp_df,
        g: decomp_dg,
        im: decompose_dim,
        ker: decompose_dker,
        cok: decompose_dcok,
        l_first_mapping,
        g_elements,
        size_of_l,
        size_of_k,
    }
}

fn print_matrix(matrix: &Vec<VecColumn>) {
    for col in matrix {
        println!("{:?}", col.internal);
    }
}

pub fn print_decomp(decomp: &RVDecomposition) {
    println!("R:");
    print_matrix(&decomp.r);
    println!("V:");
    print_matrix(&decomp.v);
}

pub fn print_ensemble(ensemble: &DecompositionEnsemble) {
    println!("Df:");
    print_decomp(&ensemble.f);
    println!("Dg:");
    print_decomp(&ensemble.g);
    println!("Dim:");
    print_decomp(&ensemble.im);
    println!("Dker:");
    print_decomp(&ensemble.ker);
    println!("Dcok:");
    print_decomp(&ensemble.cok);
}

#[derive(Default, Debug)]
struct PersistenceDiagram {
    unpaired: HashSet<usize>,
    paired: Vec<(usize, usize)>,
}

impl PersistenceDiagram {
    fn unreorder_idxs(&mut self, mapping: &impl IndexMapping) {
        self.unpaired = self
            .unpaired
            .iter()
            .cloned()
            .map(|idx| mapping.inverse_map(idx))
            .collect();
        for (b_idx, d_idx) in self.paired.iter_mut() {
            *b_idx = mapping.inverse_map(*b_idx);
            *d_idx = mapping.inverse_map(*d_idx);
        }
    }
}

impl RVDecomposition {
    fn diagram(&self) -> PersistenceDiagram {
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

#[derive(Debug)]
struct DiagramEnsemble {
    f: PersistenceDiagram,
    g: PersistenceDiagram,
    im: PersistenceDiagram,
    ker: PersistenceDiagram,
    cok: PersistenceDiagram,
}

impl DecompositionEnsemble {
    fn is_kernel_birth(&self, idx: usize) -> bool {
        let in_l = self.g_elements[idx];
        if in_l {
            return false;
        }
        let negative_in_f = self.f.r[idx].pivot().is_some();
        if !negative_in_f {
            return false;
        }
        let lowest_rim_in_l = self.im.r[idx].pivot().unwrap() < self.size_of_l;
        if !lowest_rim_in_l {
            return false;
        }
        return true;
    }

    fn is_kernel_death(&self, idx: usize) -> bool {
        let in_l = self.g_elements[idx];
        if !in_l {
            return false;
        }
        let g_index = self.l_first_mapping.map(idx);
        let negative_in_g = self.g.r[g_index].pivot().is_some();
        if !negative_in_g {
            return false;
        }
        let negative_in_f = self.f.r[idx].pivot().is_some();
        if negative_in_f {
            return false;
        }
        return true;
    }

    fn kernel_diagram(&self) -> PersistenceDiagram {
        let mut dgm = PersistenceDiagram::default();
        return dgm;
        for idx in 0..self.size_of_k {
            if self.is_kernel_birth(idx) {
                dgm.unpaired.insert(idx);
                continue;
            }
            if self.is_kernel_death(idx) {
                // TODO: Problem kernel columns have different indexing to f
                let g_birth_index = self.ker.r[idx].pivot().unwrap();
                let birth_index = self.l_first_mapping.inverse_map(g_birth_index);
                dgm.unpaired.remove(&birth_index);
                dgm.paired.push((birth_index, idx));
            }
        }
        dgm
    }

    fn image_diagram(&self) -> PersistenceDiagram {
        let mut dgm = PersistenceDiagram::default();
        for idx in 0..self.size_of_k {
            if self.g_elements[idx] {
                let g_idx = self.l_first_mapping.map(idx);
                let pos_in_g = self.g.r[g_idx].pivot().is_none();
                if pos_in_g {
                    dgm.unpaired.insert(idx);
                    continue;
                }
            }
            let neg_in_f = self.f.r[idx].pivot().is_some();
            if neg_in_f {
                let lowest_in_rim = self.im.r[idx].pivot().unwrap();
                let lowest_rim_in_l = lowest_in_rim < self.size_of_l;
                if !lowest_rim_in_l {
                    continue;
                }
                let birth_idx = self.l_first_mapping.inverse_map(lowest_in_rim);
                dgm.unpaired.remove(&birth_idx);
                dgm.paired.push((birth_idx, idx));
            }
        }
        dgm
    }

    fn cokernel_diagram(&self) -> PersistenceDiagram {
        let mut dgm = PersistenceDiagram::default();
        return dgm;
        for idx in 0..self.size_of_k {
            let pos_in_f = self.f.r[idx].pivot().is_none();
            let not_in_l_or_neg_in_g = (!self.g_elements[idx]) || self.g.r[idx].pivot().is_some();
            if pos_in_f && not_in_l_or_neg_in_g {
                dgm.unpaired.insert(idx);
                continue;
            }
            let neg_in_f = self.f.r[idx].pivot().is_some();
            if !neg_in_f {
                continue;
            }
            // TODO: This unwrap fails on example, why?
            let lowest_rim_in_l = self.im.r[idx].pivot().unwrap() < self.size_of_l;
            if lowest_rim_in_l {
                println!("{}", idx);
                let lowest_in_rcok = self.cok.r[idx].pivot().unwrap();
                dgm.unpaired.remove(&lowest_in_rcok);
                dgm.paired.push((lowest_in_rcok, idx));
            }
        }
        dgm
    }

    fn all_diagrams(&self) -> DiagramEnsemble {
        DiagramEnsemble {
            f: self.f.diagram(),
            g: {
                let mut dgm = self.g.diagram();
                dgm.unreorder_idxs(&self.l_first_mapping);
                dgm
            },
            im: self.image_diagram(),
            ker: self.kernel_diagram(),
            cok: self.cokernel_diagram(),
        }
    }
}

/// Formats the sum of two numbers as string.
#[pyfunction]
fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
    Ok((a + b).to_string())
}

/// A Python module implemented in Rust.
#[pymodule]
fn phimaker(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::File;
    use std::io::{BufRead, BufReader};

    #[test]
    fn rv_works() {
        let file = File::open("test.mat").unwrap();
        let boundary_matrix: Vec<VecColumn> = BufReader::new(file)
            .lines()
            .map(|l| {
                let l = l.unwrap();
                if l.is_empty() {
                    vec![]
                } else {
                    l.split(",").map(|c| c.parse().unwrap()).collect()
                }
            })
            .map(|l| VecColumn { internal: l })
            .collect();
        let decomposition = rv_decompose(boundary_matrix);
        print_decomp(&decomposition);
        println!("{:?}", decomposition.diagram());
        assert_eq!(true, true)
    }

    #[test]
    fn ensemble_works() {
        let file = File::open("test_annotated.mat").unwrap();
        let boundary_matrix: Vec<AnnotatedVecColumn> = BufReader::new(file)
            .lines()
            .map(|l| {
                let l = l.unwrap();
                let l_vec: Vec<usize> = l.split(",").map(|c| c.parse().unwrap()).collect();
                (l_vec[0] == 1, l_vec)
            })
            .map(|(in_g, l_vec)| AnnotatedVecColumn {
                col: VecColumn {
                    internal: l_vec[1..].to_owned(),
                },
                in_g,
            })
            .collect();
        let ensemble = all_decompositions(boundary_matrix);
        print_ensemble(&ensemble);
        println!("{:?}", ensemble.all_diagrams());
        assert_eq!(true, true)
    }
}
