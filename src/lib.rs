//! Persistent homology of filtered chain maps over F2.
//!
//! Use [`ensemble::all_decompositions`] and its `all_diagrams` method from Rust,
//! or [`sixpack_from_inclusion`] and [`sixpack`] from Python. Diagrams use column
//! indices, with one entry per birth; they do not store chain representatives.
//! Inclusion inputs must satisfy the assumptions in [`ensemble::all_decompositions`].
//! General maps are converted to inclusions using [`cylinder::build_cylinder`].
#![warn(missing_docs)]

pub mod builders;
pub mod cylinder;
pub mod diagrams;
pub mod ensemble;
pub mod indexing;
pub mod utils;

use cylinder::{CylinderMetadata, build_cylinder};
use diagrams::DiagramEnsemble;
use ensemble::{all_decompositions, all_decompositions_slow};

use lophat::algorithms::LockFreeAlgorithm;
use pyo3::prelude::*;

/// Compute six persistence diagrams for an inclusion A into B over F2.
///
/// Parameters
/// ----------
/// boundary_matrix : `list[list[int]]`
///     Boundary of each generator of B, as nonzero row indices over F2.
///     Column order defines filtration order; no filtration values are accepted.
/// dimensions : `list[int]`
///     Nonnegative degree of each generator, one per boundary column.
/// cols_in_domain : `list[int]`
///     Distinct indices of the generators of A in B. Any order is accepted.
/// num_threads : int, default 0
///     Thread limit per reduction, not for the whole call. Zero selects the
///     reduction library's automatic thread count.
/// slow : bool, default False
///     Reduce sequentially and save intermediate decompositions to temporary files.
///     Diagram extraction currently loads all six decompositions into memory.
///
/// Returns
/// -------
/// DiagramEnsemble
///     domain, codomain, image, kernel, cokernel, and relative diagrams.
///     Each Python property returns a fresh dict mapping birth indices to death
///     indices, with None for essential classes. All indices refer to B.
///     Relative means H(B/A), without an additional basepoint generator.
///
/// Assumptions
/// -----------
/// Boundary columns must be sorted, duplicate-free, and strictly upper triangular
/// (each row index is smaller than its column index). Boundaries lower degree by
/// one and square to zero. Domain generators must be closed under the boundary.
/// Inputs are assumed valid; these conditions are not comprehensively checked.
/// The homological degree is `dimensions[birth]`, except for the kernel, where it is
///  `dimensions[birth]` - 1 because its birth cell kills a domain class.
/// The Python interpreter is released during computation.
///
/// Limitations
/// -----------
/// Invalid inputs may panic or give incorrect results. With LoPhat 0.11.0,
/// slow mode can panic when serializing an empty decomposition, including an empty
/// domain or the empty relative complex of an identity inclusion.
///
/// Examples
/// --------
/// ```python
/// >>> from phimaker import sixpack_from_inclusion
/// >>> result = sixpack_from_inclusion([[], [], [0, 1]], [0, 0, 1], [1])
/// >>> result.codomain == {0: None, 1: 2}
/// True
/// ```
#[pyfunction]
#[pyo3(signature = (boundary_matrix, dimensions, cols_in_domain, num_threads=0, slow=false))]
pub fn sixpack_from_inclusion(
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

/// Compute six persistence diagrams for a filtered chain map over F2.
///
/// Parameters
/// ----------
/// domain_matrix, codomain_matrix : `list[tuple[float, int, list[int]]]`
///     Columns (entrance_time, degree, boundary_indices) of the two complexes.
///     Boundary indices are local to the corresponding complex.
/// map : `list[list[int]]`
///     Column i lists the codomain generators in the image of domain generator i.
///     Supply one column per domain generator; additional columns are ignored.
/// num_threads : int, default 0
///     Thread limit per reduction; zero selects automatic thread counts.
/// slow : bool, default False
///     Use sequential reductions and temporary files. Extraction reloads all six
///     decompositions; empty decompositions may fail with LoPhat 0.11.0.
///
/// Returns
/// -------
/// `tuple[DiagramEnsemble, CylinderMetadata]`
///     Diagrams for the domain inclusion into the mapping cylinder and metadata.
///     Birth/death indices refer to cylinder columns, NOT input column indices.
///     Use `metadata.times[index]` to recover filtration times and the index arrays
///     to locate input generators. Infinite deaths become None in Python.
///     Intervals with equal birth and death times are retained. Relative is the
///     homology of the cylinder modulo the domain (the mapping cone of the map).
///     For kernel intervals the degree is `metadata.dimensions[birth]` - 1;
///     for other intervals it is `metadata.dimensions[birth]`.
///
/// Assumptions
/// -----------
/// Each matrix is ordered by nondecreasing, non-NaN entrance time, with sorted,
/// duplicate-free boundary columns strictly above the diagonal. Boundaries lower
/// degree by one and square to zero. The map columns are sorted, duplicate-free,
/// and define a degree-preserving chain map commuting with the boundaries.
/// Every image generator must enter no later than its domain generator.
/// Equal times are allowed. Cylinder ties put domain generators before codomain
/// generators before shifted domain generators, preserving each input order.
/// These algebraic conditions are not comprehensively validated; invalid input
/// may panic or give incorrect results. Computation releases the Python interpreter.
///
/// Examples
/// --------
/// ```python
/// >>> from phimaker import sixpack
/// >>> diagrams, metadata = sixpack([(0.0, 0, [])], [(0.0, 0, [])], [[0]])
/// >>> len(diagrams.domain)
/// 1
/// ```
#[pyfunction]
#[pyo3(signature = (domain_matrix, codomain_matrix, map, num_threads=0, slow=false))]
pub fn sixpack(
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
