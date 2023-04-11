pub mod builders;
pub mod cylinder;
pub mod diagrams;
pub mod ensemble;
pub mod indexing;
pub mod utils;

use cylinder::{build_cylinder, CylinderMetadata};
use diagrams::DiagramEnsemble;
use ensemble::all_decompositions;
use indexing::AnnotatedColumn;
use lophat::VecColumn;

use pyo3::prelude::*;

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

#[pyfunction]
#[pyo3(signature = (domain_matrix, codomain_matrix, map, num_threads=0))]
fn compute_ensemble_cylinder(
    domain_matrix: Vec<(f64, Vec<usize>)>,
    codomain_matrix: Vec<(f64, Vec<usize>)>,
    map: Vec<Vec<usize>>,
    num_threads: usize,
) -> (DiagramEnsemble, CylinderMetadata) {
    let domain_matrix = domain_matrix
        .into_iter()
        .map(|(time, bdry)| (time, bdry.into()))
        .collect();
    let codomain_matrix = codomain_matrix
        .into_iter()
        .map(|(time, bdry)| (time, bdry.into()))
        .collect();
    let map = map.into_iter().map(VecColumn::from).collect();
    let (cylinder, metadata) = build_cylinder(domain_matrix, codomain_matrix, map);
    let decomps = all_decompositions(cylinder, num_threads);
    (decomps.all_diagrams(), metadata)
}

/// A Python module implemented in Rust.
#[pymodule]
fn phimaker(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compute_ensemble, m)?)?;
    m.add_function(wrap_pyfunction!(compute_ensemble_cylinder, m)?)?;
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
