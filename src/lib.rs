pub mod builders;
pub mod cylinder;
pub mod diagrams;
pub mod ensemble;
pub mod indexing;
pub mod overlap;
pub mod utils;

use cylinder::{build_cylinder, CylinderMetadata};
use diagrams::DiagramEnsemble;
use ensemble::{all_decompositions, all_decompositions_slow};
use indexing::AnnotatedColumn;

use lophat::{algorithms::LockFreeAlgorithm, columns::VecColumn};
use overlap::compute_zero_overlap;
use pyo3::prelude::*;

/// Compute the six-pack of persistence diagrams for an inclusion of filtered cell complexes.
/// `matrix` is a vector of tuples corresponding to the cells of the codomain,
/// where each tuple is of the form
/// `(is_simplex_in_domain, dimension_of_simplex, indices_of_boundary_simplices)`.
#[pyfunction]
#[pyo3(signature = (matrix, num_threads=0, slow=false))]
fn compute_ensemble(
    py: Python<'_>,
    matrix: Vec<(bool, usize, Vec<usize>)>,
    num_threads: usize,
    slow: bool,
) -> DiagramEnsemble {
    let annotated_matrix: Vec<_> = matrix
        .into_iter()
        .map(|(in_g, dimension, boundary)| AnnotatedColumn {
            in_g,
            col: VecColumn::from((dimension, boundary)),
        })
        .collect();
    if slow {
        let decomps =
            all_decompositions_slow::<LockFreeAlgorithm<_>>(annotated_matrix, num_threads);
        decomps.all_diagrams()
    } else {
        let decomps = py.allow_threads(|| {
            all_decompositions::<LockFreeAlgorithm<_>>(annotated_matrix, num_threads)
        });
        decomps.all_diagrams()
    }
}

/// Compute the six-pack of persistence diagrams for an arbitrary map $f$
/// of filtered chain complexes over $\mathbb{F}_2$.
/// `domain_matrix` and `codomain_matrix` are vectors whose i^th^ entry represents
/// the i^th^ column of the boundary matrix of the domain and codomain, respectively.
/// Each such entry corresponds to a generator of the chain complex and is a tuple of the form
/// `(entrance_time, degree of i^th^ generator, indices of generators in the boundary)`.
/// The columns must be sorted by entrance time, and the matrices must be strictly upper-triangular.
/// Similarly, entries of `map` represent columns in the matrix of $f$.
/// The i^th^ entry of `map` is vector of indices corresponding to the
/// non-zero entries of the i^th^ column of the matrix of $f$.
/// `map` must have at least as many entries as there are domain cells.
///
/// # Panics:
/// - If the domain and codomain matrices are not sorted by entrance time.
/// - If the domain and codomain matrices are not strictly upper-triangular.
/// - If `map` is not compatible with the domain matrix, i.e., if the entrance time of any
///   generator in the domain is less than the entrance time of any generators in its image under
///   $f$.
#[pyfunction]
#[pyo3(signature = (domain_matrix, codomain_matrix, map, num_threads=0, slow=false))]
fn compute_ensemble_cylinder(
    py: Python<'_>,
    domain_matrix: Vec<(f64, usize, Vec<usize>)>,
    codomain_matrix: Vec<(f64, usize, Vec<usize>)>,
    map: Vec<Vec<usize>>,
    num_threads: usize,
    slow: bool,
) -> (DiagramEnsemble, CylinderMetadata) {
    // We mark each map with the dimension of the domain column
    let map = map
        .into_iter()
        .zip(domain_matrix.iter())
        .map(|(image, domain_col)| VecColumn::from((domain_col.1, image)))
        .collect();
    let domain_matrix = domain_matrix
        .into_iter()
        .map(|(time, dimension, boundary)| (time, VecColumn::from((dimension, boundary))))
        .collect();
    let codomain_matrix = codomain_matrix
        .into_iter()
        .map(|(time, dimension, boundary)| (time, VecColumn::from((dimension, boundary))))
        .collect();
    let (cylinder, metadata) = build_cylinder(domain_matrix, codomain_matrix, map);
    if slow {
        let decomps =
            all_decompositions_slow::<LockFreeAlgorithm<_>>(cylinder, num_threads);
        (decomps.all_diagrams(), metadata)
    } else {
        let decomps = py.allow_threads(|| {
            all_decompositions::<LockFreeAlgorithm<_>>(cylinder, num_threads)
        });
        (decomps.all_diagrams(), metadata)
    }
}

#[pyfunction]
fn zero_overlap(matrix: Vec<(bool, usize, Vec<usize>)>) -> Vec<(usize, usize)> {
    let annotated_matrix: Vec<AnnotatedColumn<VecColumn>> = matrix
        .into_iter()
        .map(|(in_g, dimension, boundary)| AnnotatedColumn {
            in_g,
            col: VecColumn::from((dimension, boundary)),
        })
        .collect();
    compute_zero_overlap(&annotated_matrix)
}

/// A Python module implemented in Rust.
#[pymodule]
fn phimaker(_py: Python, m: &PyModule) -> PyResult<()> {
    pyo3_log::init();
    m.add_function(wrap_pyfunction!(compute_ensemble, m)?)?;
    m.add_function(wrap_pyfunction!(compute_ensemble_cylinder, m)?)?;
    m.add_function(wrap_pyfunction!(zero_overlap, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::utils::print_ensemble;

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
                (l_vec[0] == 1, l_vec[1], l_vec)
            })
            .map(|(in_g, dimension, l_vec)| AnnotatedColumn {
                col: VecColumn::from((dimension, l_vec[2..].to_owned())),
                in_g,
            })
            .collect();
        let ensemble = all_decompositions::<LockFreeAlgorithm<_>>(boundary_matrix, 0);
        print_ensemble(&ensemble);
        println!("{:?}", ensemble.all_diagrams());
        assert_eq!(true, true)
    }
}
