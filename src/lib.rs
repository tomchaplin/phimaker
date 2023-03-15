use std::thread;

use lophat::*;

use pyo3::prelude::*;

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

#[derive(Debug)]
pub struct DecompositionEnsemble {
    f: RVDecomposition<VecColumn>,
    g: RVDecomposition<VecColumn>,
    im: RVDecomposition<VecColumn>,
    ker: RVDecomposition<VecColumn>,
    cok: RVDecomposition<VecColumn>,
    rel: RVDecomposition<VecColumn>,
    l_first_mapping: VectorMapping,
    kernel_mapping: VectorMapping,
    rel_mapping: VectorMapping,
    g_elements: Vec<bool>,
    size_of_l: usize,
    size_of_k: usize,
}

fn compute_l_first_mapping(matrix: &Vec<AnnotatedColumn<VecColumn>>) -> VectorMapping {
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

fn extract_columns<'a>(
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

fn build_dg<'a>(
    df: &'a Vec<VecColumn>,
    g_elements: &'a Vec<bool>,
    l_first_mapping: &'a VectorMapping,
) -> impl Iterator<Item = VecColumn> + 'a {
    extract_columns(df, g_elements).map(|mut col| {
        col.reorder_rows(l_first_mapping);
        col
    })
}

fn build_dim<'a>(
    df: &'a Vec<VecColumn>,
    mapping: &'a impl IndexMapping,
) -> impl Iterator<Item = VecColumn> + 'a {
    df.iter().cloned().map(|mut col| {
        col.reorder_rows(mapping);
        col
    })
}

fn build_kernel_mapping(dim_decomposition: &RVDecomposition<VecColumn>) -> VectorMapping {
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
fn build_drel<'a>(
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

fn build_dker<'a>(
    dim_decomposition: &'a RVDecomposition<VecColumn>,
    mapping: &'a impl IndexMapping,
) -> impl Iterator<Item = VecColumn> + 'a {
    let rim_cols = dim_decomposition.r.iter();
    let vim_cols = dim_decomposition.v.as_ref().unwrap().iter();
    let paired_cols = rim_cols.zip(vim_cols);
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

fn build_dcok<'a>(
    df: &'a Vec<VecColumn>,
    dg_decomposition: &'a RVDecomposition<VecColumn>,
    g_elements: &'a Vec<bool>,
    mapping: &'a impl IndexMapping,
) -> impl Iterator<Item = VecColumn> + 'a {
    (0..df.len()).map(|col_idx| {
        let col_in_g = g_elements[col_idx];
        if col_in_g {
            let idx_in_dg = mapping.map(col_idx).unwrap();
            let dg_rcol = &dg_decomposition.r[idx_in_dg];
            if dg_rcol.pivot().is_none() {
                let mut next_col = dg_decomposition.v.as_ref().unwrap()[idx_in_dg].clone();
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

fn run_decomposition(
    matrix: impl Iterator<Item = VecColumn>,
    column_height: Option<usize>,
    options: LoPhatOptions,
) -> RVDecomposition<VecColumn> {
    if options.num_threads == 1 {
        rv_decompose(matrix, options)
    } else {
        rv_decompose_lock_free(matrix, column_height, options)
    }
}

pub fn all_decompositions(
    matrix: Vec<AnnotatedColumn<VecColumn>>,
    num_threads: usize,
) -> DecompositionEnsemble {
    let options = LoPhatOptions {
        maintain_v: true,
        num_threads,
    };
    // TODO: Clean this up so we aren't collecting the matrix again.
    let l_first_mapping = compute_l_first_mapping(&matrix);
    let g_elements: Vec<bool> = matrix.iter().map(|anncol| anncol.in_g).collect();
    let size_of_l = g_elements.iter().filter(|in_g| **in_g).count();
    let size_of_k = matrix.len();
    let df: Vec<VecColumn> = matrix.into_iter().map(|anncol| anncol.col).collect();
    let (f, (g, cok), (im, ker, kernel_mapping), (rel, rel_mapping)) = thread::scope(|s| {
        let thread1 = s.spawn(|| {
            // Decompose Df
            let out = run_decomposition(df.iter().cloned(), Some(size_of_k), options);
            println!("Decomposed f");
            out
        });
        let thread2 = s.spawn(|| {
            // Decompose Dg
            let dg = build_dg(&df, &g_elements, &l_first_mapping);
            let decomp_dg = run_decomposition(dg, Some(size_of_l), options);
            println!("Decomposed g");
            // Decompose dcok
            let dcok = build_dcok(&df, &decomp_dg, &g_elements, &l_first_mapping);
            let decompose_dcok = run_decomposition(dcok, Some(size_of_k), options);
            println!("Decomposed cok");
            (decomp_dg, decompose_dcok)
        });
        let thread3 = s.spawn(|| {
            // Decompose dim
            let dim = build_dim(&df, &l_first_mapping);
            let decompose_dim = run_decomposition(dim, Some(size_of_k), options);
            println!("Decomposed im");
            // Decompose dker
            // TODO: Also need to return mapping from columns of Df to columns of Dker
            let dker = build_dker(&decompose_dim, &l_first_mapping);
            let decompose_dker = run_decomposition(dker, Some(size_of_k), options);
            println!("Decomposed ker");
            let kernel_mapping = build_kernel_mapping(&decompose_dim);
            (decompose_dim, decompose_dker, kernel_mapping)
        });
        let thread4 = s.spawn(|| {
            let (rel_mapping, l_index) = build_rel_mapping(&df, &g_elements, size_of_l, size_of_k);
            let drel = build_drel(&df, &g_elements, &rel_mapping, l_index);
            let decompose_drel = run_decomposition(drel, Some(size_of_k - size_of_l + 1), options);
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

pub fn print_decomp(decomp: &RVDecomposition<VecColumn>) {
    println!("R:");
    print_matrix(&decomp.r);
    println!("V:");
    print_matrix(&decomp.v.as_ref().unwrap());
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

fn unreorder_idxs(diagram: &mut PersistenceDiagram, mapping: &impl IndexMapping) {
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
                dgm.paired.insert((birth_index, idx));
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
                dgm.paired.insert((birth_idx, idx));
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
                dgm.paired.insert((lowest_in_rcok, idx));
            }
        }
        dgm
    }

    fn all_diagrams(&self) -> DiagramEnsemble {
        DiagramEnsemble {
            f: self.f.diagram(),
            g: {
                let mut dgm = self.g.diagram();
                unreorder_idxs(&mut dgm, &self.l_first_mapping);
                dgm
            },
            rel: {
                let mut dgm = self.rel.diagram();
                unreorder_idxs(&mut dgm, &self.rel_mapping);
                dgm
            },
            im: self.image_diagram(),
            ker: self.kernel_diagram(),
            cok: self.cokernel_diagram(),
        }
    }
}

#[pyfunction]
#[pyo3(signature = (matrix, num_threads=0))]
fn compute_ensemble(matrix: Vec<(bool, Vec<usize>)>, num_threads: usize) -> DiagramEnsemble {
    let annotated_matrix = matrix
        .into_iter()
        .map(|(in_g, bdry)| AnnotatedColumn {
            in_g,
            col: VecColumn { internal: bdry },
        })
        .collect();
    let decomps = all_decompositions(annotated_matrix, num_threads);
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
    fn ensemble_works() {
        let file = File::open("examples/test_annotated.mat").unwrap();
        let boundary_matrix: Vec<AnnotatedColumn<VecColumn>> = BufReader::new(file)
            .lines()
            .map(|l| {
                let l = l.unwrap();
                let l_vec: Vec<usize> = l.split(",").map(|c| c.parse().unwrap()).collect();
                (l_vec[0] == 1, l_vec)
            })
            .map(|(in_g, l_vec)| AnnotatedColumn {
                col: VecColumn {
                    internal: l_vec[1..].to_owned(),
                },
                in_g,
            })
            .collect();
        let ensemble = all_decompositions(boundary_matrix, 0);
        print_ensemble(&ensemble);
        println!("{:?}", ensemble.all_diagrams());
        assert_eq!(true, true)
    }
}
