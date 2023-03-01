use std::{
    cmp::Ordering,
    collections::{HashMap, HashSet},
    thread,
};

use pyo3::prelude::*;

pub trait IndexMapping {
    fn map(&self, index: usize) -> Option<usize>;
    fn inverse_map(&self, index: usize) -> Option<usize>;
}

#[derive(Debug)]
struct VectorMapping {
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

    // TODO: Reimplement so that this happens in-place?
    fn reorder_rows(&mut self, mapping: &impl IndexMapping) {
        let mut new_col: Vec<usize> = self
            .internal
            .iter()
            .filter_map(|&row_idx| mapping.map(row_idx))
            .collect();
        new_col.sort();
        self.internal = new_col;
    }

    // TODO: Reimplement so that this happens in-place?
    fn unreorder_rows(&mut self, mapping: &impl IndexMapping) {
        let mut new_col: Vec<usize> = self
            .internal
            .iter()
            .filter_map(|&row_idx| mapping.inverse_map(row_idx))
            .collect();
        new_col.sort();
        self.internal = new_col;
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
    rel: RVDecomposition,
    l_first_mapping: VectorMapping,
    kernel_mapping: VectorMapping,
    rel_mapping: VectorMapping,
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

fn build_kernel_mapping(dim_decomposition: &RVDecomposition) -> VectorMapping {
    let mut counter = 0;
    let mut idx_list: Vec<Option<usize>> = vec![];
    for r_col in dim_decomposition.r.iter() {
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

// WARNING: This functions makes the following assumption:
// The 0-cells are precisely the cells with empty boundaries
fn build_rel_mapping(
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
            if r_col.internal.len() == 0 {
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

// WARNING: This functions makes the following assumption:
// If the boundary of a cell is entirely contained in L then that cell is in L
// This ensures that a 1-cell not in L can have at most 1 vertex in L
// This makes it easier to map the boundary
// Also inherits assumption from build_rel_mapping
fn build_drel(
    matrix: &Vec<VecColumn>,
    g_elements: &Vec<bool>,
    rel_mapping: &VectorMapping,
    l_index: usize,
    size_of_l: usize,
    size_of_k: usize,
) -> Vec<VecColumn> {
    let mut new_matrix = Vec::with_capacity(size_of_k - size_of_l + 1);
    for (idx, (col, &in_g)) in matrix.iter().zip(g_elements.iter()).enumerate() {
        if in_g && idx != l_index {
            // Don't add elements of L, unless its the first one
            continue;
        }
        let mut new_col = col.clone();
        new_col.reorder_rows(rel_mapping);
        new_matrix.push(new_col);
    }
    new_matrix
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
            let idx_in_dg = mapping.map(col_idx).unwrap();
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
    let (f, (g, cok), (im, ker, kernel_mapping), (rel, rel_mapping)) = thread::scope(|s| {
        let thread1 = s.spawn(|| {
            // Decompose Df
            let out = rv_decompose(df.clone());
            println!("Decomposed f");
            out
        });
        let thread2 = s.spawn(|| {
            // Decompose Dg
            let dg = build_dg(&df, &g_elements, &l_first_mapping);
            let decomp_dg = rv_decompose(dg);
            println!("Decomposed g");
            // Decompose dcok
            let dcok = build_dcok(&df, &decomp_dg, &g_elements, &l_first_mapping);
            let decompose_dcok = rv_decompose(dcok);
            println!("Decomposed cok");
            (decomp_dg, decompose_dcok)
        });
        let thread3 = s.spawn(|| {
            // Decompose dim
            let dim = build_dim(&df, &l_first_mapping);
            let decompose_dim = rv_decompose(dim);
            println!("Decomposed im");
            // Decompose dker
            // TODO: Also need to return mapping from columns of Df to columns of Dker
            let dker = build_dker(&decompose_dim, &l_first_mapping);
            let decompose_dker = rv_decompose(dker);
            println!("Decomposed ker");
            let kernel_mapping = build_kernel_mapping(&decompose_dim);
            (decompose_dim, decompose_dker, kernel_mapping)
        });
        let thread4 = s.spawn(|| {
            let (rel_mapping, l_index) = build_rel_mapping(&df, &g_elements, size_of_l, size_of_k);
            let drel = build_drel(
                &df,
                &g_elements,
                &rel_mapping,
                l_index,
                size_of_l,
                size_of_k,
            );
            let decompose_drel = rv_decompose(drel);
            println!("Decomposed rel");
            (decompose_drel, rel_mapping)
        });
        (
            thread1.join().unwrap(),
            thread2.join().unwrap(),
            thread3.join().unwrap(),
            thread4.join().unwrap(),
        )
    });
    DecompositionEnsemble {
        f,
        g,
        im,
        ker,
        cok,
        rel,
        g_elements,
        l_first_mapping,
        kernel_mapping,
        rel_mapping,
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

#[pyclass]
#[derive(Default, Debug, Clone)]
struct PersistenceDiagram {
    #[pyo3(get)]
    unpaired: HashSet<usize>,
    #[pyo3(get)]
    paired: Vec<(usize, usize)>,
}

impl PersistenceDiagram {
    fn unreorder_idxs(&mut self, mapping: &impl IndexMapping) {
        self.unpaired = self
            .unpaired
            .iter()
            .cloned()
            .map(|idx| mapping.inverse_map(idx).unwrap())
            .collect();
        for (b_idx, d_idx) in self.paired.iter_mut() {
            *b_idx = mapping.inverse_map(*b_idx).unwrap();
            *d_idx = mapping.inverse_map(*d_idx).unwrap();
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

#[pyclass]
#[derive(Debug, Clone)]
struct DiagramEnsemble {
    #[pyo3(get)]
    f: PersistenceDiagram,
    #[pyo3(get)]
    g: PersistenceDiagram,
    #[pyo3(get)]
    im: PersistenceDiagram,
    #[pyo3(get)]
    ker: PersistenceDiagram,
    #[pyo3(get)]
    cok: PersistenceDiagram,
    #[pyo3(get)]
    rel: PersistenceDiagram,
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
        let g_index = self.l_first_mapping.map(idx).unwrap();
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
        for idx in 0..self.size_of_k {
            if self.is_kernel_birth(idx) {
                dgm.unpaired.insert(idx);
                continue;
            }
            if self.is_kernel_death(idx) {
                // TODO: Problem kernel columns have different indexing to f
                let ker_idx = self.kernel_mapping.map(idx).unwrap();
                let g_birth_index = self.ker.r[ker_idx].pivot().unwrap();
                let birth_index = self.l_first_mapping.inverse_map(g_birth_index).unwrap();
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
                let g_idx = self.l_first_mapping.map(idx).unwrap();
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
                let birth_idx = self.l_first_mapping.inverse_map(lowest_in_rim).unwrap();
                dgm.unpaired.remove(&birth_idx);
                dgm.paired.push((birth_idx, idx));
            }
        }
        dgm
    }

    fn cokernel_diagram(&self) -> PersistenceDiagram {
        let mut dgm = PersistenceDiagram::default();
        for idx in 0..self.size_of_k {
            let pos_in_f = self.f.r[idx].pivot().is_none();
            let g_idx = self.l_first_mapping.map(idx).unwrap();
            let not_in_l_or_neg_in_g = (!self.g_elements[idx]) || self.g.r[g_idx].pivot().is_some();
            if pos_in_f && not_in_l_or_neg_in_g {
                dgm.unpaired.insert(idx);
                continue;
            }
            let neg_in_f = self.f.r[idx].pivot().is_some();
            if !neg_in_f {
                continue;
            }
            let lowest_rim_in_l = self.im.r[idx].pivot().unwrap() < self.size_of_l;
            if !lowest_rim_in_l {
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
            rel: {
                let mut dgm = self.rel.diagram();
                dgm.unreorder_idxs(&self.rel_mapping);
                dgm
            },
            im: self.image_diagram(),
            ker: self.kernel_diagram(),
            cok: self.cokernel_diagram(),
        }
    }
}

#[pyfunction]
fn compute_ensemble(matrix: Vec<(bool, Vec<usize>)>) -> DiagramEnsemble {
    let annotated_matrix = matrix
        .into_iter()
        .map(|(in_g, bdry)| AnnotatedVecColumn {
            in_g,
            col: VecColumn { internal: bdry },
        })
        .collect();
    let decomps = all_decompositions(annotated_matrix);
    decomps.all_diagrams()
}

/// A Python module implemented in Rust.
#[pymodule]
fn phimaker(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compute_ensemble, m)?)?;
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
