pub mod builders;
pub mod cylinder;
pub mod diagrams;
pub mod ensemble;
pub mod indexing;
pub mod utils;

use cylinder::{build_cylinder, CylinderMetadata};
use diagrams::DiagramEnsemble;
use ensemble::{all_decompositions, all_decompositions_slow};

use lophat::algorithms::LockFreeAlgorithm;
use pyo3::prelude::*;

/// Compute the six-pack of persistence diagrams for an inclusion of filtered chain complexes.
/// Requires a generating set for the codomain that extends a generating set for the domain.
/// `boundary_matrix`: vector such that `boundary_matrix[i]` is the vector
/// of indices of generators in the boundary of generator `i`.
/// `dimensions`: list of dimensions of the generators in the codomain.
/// `cols_in_domain`: list of indices of generators of the domain.
/// `num_threads`: the maximum number of threads used in individual decompositions.
/// `slow`: whether the decompositions should be performed in memory or streamed from disk.
#[pyfunction]
#[pyo3(signature = (boundary_matrix, dimensions, cols_in_domain, num_threads=0, slow=false))]
fn sixpack_from_inclusion(
    py: Python<'_>,
    boundary_matrix: Vec<Vec<usize>>,
    dimensions: Vec<usize>,
    cols_in_domain: Vec<usize>,
    num_threads: usize,
    slow: bool,
) -> DiagramEnsemble {
    py.detach(|| {
        if slow {
            let decomps = all_decompositions_slow::<LockFreeAlgorithm<_>, _>(
                &boundary_matrix,
                &dimensions,
                &cols_in_domain,
                num_threads,
            );
            decomps.all_diagrams()
        } else {
            let decomps = all_decompositions::<LockFreeAlgorithm<_>, _>(
                &boundary_matrix,
                &dimensions,
                &cols_in_domain,
                num_threads,
            );
            // TODO: get the matrix of the map on persistence modules
            // as well as a basis for the matrix
            decomps.all_diagrams()
        }
    })
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
/// `map` must have at least as many entries as there are domain cells,
/// and must satisfy the requirements of a filtered chain map.
///
/// # Panics:
/// - If the domain and codomain matrices are not sorted by entrance time.
/// - If the domain and codomain matrices are not strictly upper-triangular.
/// - If `map` is not compatible with the domain matrix, i.e., if the entrance time of any
///   generator in the domain is less than the entrance time of any generators in its image under
///   $f$.
#[pyfunction]
#[pyo3(signature = (domain_matrix, codomain_matrix, map, num_threads=0, slow=false))]
fn sixpack(
    py: Python<'_>,
    domain_matrix: Vec<(f64, usize, Vec<usize>)>,
    codomain_matrix: Vec<(f64, usize, Vec<usize>)>,
    map: Vec<Vec<usize>>,
    num_threads: usize,
    slow: bool,
) -> (DiagramEnsemble, CylinderMetadata) {
    // We mark each map with the dimension of the domain column
    py.detach(|| {
        let (cylinder_boundary_matrix, metadata) =
            build_cylinder(&domain_matrix, &codomain_matrix, &map);
        if slow {
            let decomps = all_decompositions_slow::<LockFreeAlgorithm<_>, _>(
                &cylinder_boundary_matrix,
                &metadata.dimensions,
                &metadata.domain_indices,
                num_threads,
            );
            (decomps.all_diagrams(), metadata)
        } else {
            let decomps = all_decompositions::<LockFreeAlgorithm<_>, _>(
                &cylinder_boundary_matrix,
                &metadata.dimensions,
                &metadata.domain_indices,
                num_threads,
            );
            (decomps.all_diagrams(), metadata)
        }
    })
}

/// A Python module implemented in Rust.
#[pymodule]
fn phimaker(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    pyo3_log::init();
    m.add_function(wrap_pyfunction!(sixpack_from_inclusion, m)?)?;
    m.add_function(wrap_pyfunction!(sixpack, m)?)?;
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
        let mut boundary_matrix = Vec::<Vec<usize>>::new();
        let mut cols_in_domain = Vec::<usize>::new();
        let mut dimensions = Vec::<usize>::new();
        BufReader::new(file)
            .lines()
            .enumerate()
            .for_each(|(idx, line)| {
                let line = line.unwrap();
                let line_items: Vec<usize> = line
                    .split(",")
                    .map(|number_string| number_string.parse().unwrap())
                    .collect();
                let in_domain = line_items[0] == 1;
                let dimension = line_items[1];
                let boundary = line_items.into_iter().skip(2).collect::<Vec<usize>>();
                if in_domain {
                    cols_in_domain.push(idx);
                }
                dimensions.push(dimension);
                boundary_matrix.push(boundary);
            });
        let ensemble = all_decompositions::<LockFreeAlgorithm<_>, _>(
            &boundary_matrix,
            &dimensions,
            &cols_in_domain,
            0,
        );
        print_ensemble(&ensemble);
        println!("{:?}", ensemble.all_diagrams());
        assert_eq!(true, true)
    }
}
