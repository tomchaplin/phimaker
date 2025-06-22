use lophat::columns::{Column, VecColumn};
use pyo3::prelude::*;
use log::debug;

use std::cmp::Ordering;

use itertools::Itertools;

use crate::AnnotatedColumn;

#[derive(PartialEq, Eq, Clone, Copy, Debug)]
enum CylinderColType {
    Domain,
    Codomain,
    DomainShifted,
}

impl CylinderColType {
    fn type_int(&self) -> usize {
        match self {
            CylinderColType::Domain => 1,
            CylinderColType::Codomain => 2,
            CylinderColType::DomainShifted => 3,
        }
    }
}
impl PartialOrd for CylinderColType {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for CylinderColType {
    fn cmp(&self, other: &Self) -> Ordering {
        self.type_int().cmp(&other.type_int())
    }
}

#[pyclass(get_all)]
pub struct CylinderMetadata {
    pub times: Vec<f64>,
    pub domain: Vec<usize>,
    pub codomain: Vec<usize>,
    pub domain_shift: Vec<usize>,
}

// Build the filtered mapping cylinder of a map between filtered chain complexes.
pub fn build_cylinder(
    domain_matrix: Vec<(f64, VecColumn)>,
    codomain_matrix: Vec<(f64, VecColumn)>,
    map: Vec<VecColumn>,
) -> (Vec<AnnotatedColumn<VecColumn>>, CylinderMetadata) {
    let domain_size = domain_matrix.len();
    let codomain_size = codomain_matrix.len();
    let cylinder_size = 2 * domain_size + codomain_size;
    let mut domain_idxs: Vec<usize> = Vec::with_capacity(domain_size);
    let mut codomain_idxs: Vec<usize> = Vec::with_capacity(codomain_size);
    let mut domain_shift_idxs: Vec<usize> = Vec::with_capacity(domain_size);
    let mut cylinder_matrix: Vec<AnnotatedColumn<VecColumn>> = Vec::with_capacity(cylinder_size);
    let mut times: Vec<f64> = Vec::with_capacity(cylinder_size);

    // Compare times then compare cell type
    let cell_ordering =
        |x: &(usize, f64, CylinderColType, VecColumn),
         y: &(usize, f64, CylinderColType, VecColumn)| { (x.1, x.2) <= (y.1, y.2) };

    let domain_iter = domain_matrix
        .clone()
        .into_iter()
        .enumerate()
        .map(|(idx, (time, col))| {
            // First copy of domain columns
            (idx, time, CylinderColType::Domain, col.clone())
        });

    let domain_shift_iter = domain_matrix
        .into_iter()
        .enumerate()
        .map(|(idx, (time, col))| {
            // Second copy of domain columns
            (idx, time, CylinderColType::DomainShifted, col)
        });

    let codomain_iter = codomain_matrix
        .into_iter()
        .enumerate()
        .map(|(idx, (time, col))| (idx, time, CylinderColType::Codomain, col));

    let cylinder_iter = domain_iter
        .merge_by(codomain_iter, cell_ordering)
        .merge_by(domain_shift_iter, cell_ordering);

    for (cylinder_idx, (original_idx, time, col_cell_type, col)) in cylinder_iter.enumerate() {
        // Build column
        let cylinder_col = match col_cell_type {
            CylinderColType::Domain => {
                // Take normal boundary but translate to new idxs
                let new_boundary = col
                    .entries()
                    .map(|row_idx| {
                        domain_idxs
                            .get(row_idx)
                            .expect("Domain matrix should be strict upper triangular")
                    })
                    .copied()
                    .collect();
                AnnotatedColumn {
                    col: VecColumn::from((col.dimension(), new_boundary)),
                    in_g: true,
                }
            }
            CylinderColType::Codomain => {
                // Take normal boundary but translate to new idxs
                let new_boundary = col
                    .entries()
                    .map(|row_idx| {
                        codomain_idxs
                            .get(row_idx)
                            .expect("Codomain matrix should be strict upper triangular")
                    })
                    .copied()
                    .collect();
                AnnotatedColumn {
                    col: VecColumn::from((col.dimension(), new_boundary)),
                    in_g: false,
                }
            }
            CylinderColType::DomainShifted => {
                // Weird column
                let domain_part = vec![domain_idxs
                    .get(original_idx)
                    .expect("Map should have one column per column of domain matrix")]
                .into_iter()
                .copied();
                let codomain_part = map
                    .get(original_idx)
                    .unwrap() // Original_idx is already an index into map
                    .entries()
                    .map(|row_idx|
                        codomain_idxs.get(row_idx)
                        .expect("Map must be compatible with both filtrations i.e. entrance time of f(c) <= entrance time of c")
                    )
                    .copied();
                let domain_shift_part = col
                    .entries()
                    .map(|row_idx| {
                        domain_shift_idxs
                            .get(row_idx)
                            .expect("Domain matrix should be strict upper triangular")
                    })
                    .copied();
                let new_boundary = domain_part
                    .chain(codomain_part)
                    .chain(domain_shift_part)
                    .sorted() // Different parts might be interleaved; need to sort idxs
                    .collect();
                AnnotatedColumn {
                    // The shifted col is in the dimension of the domain column + 1
                    col: VecColumn::from((col.dimension() + 1, new_boundary)),
                    in_g: false,
                }
            }
        };
        cylinder_matrix.push(cylinder_col);
        // Add to appropriate indexing vector
        match col_cell_type {
            CylinderColType::Domain => {
                domain_idxs.push(cylinder_idx);
            }
            CylinderColType::Codomain => {
                codomain_idxs.push(cylinder_idx);
            }
            CylinderColType::DomainShifted => {
                domain_shift_idxs.push(cylinder_idx);
            }
        }
        times.push(time)
    }
    let metadata = CylinderMetadata {
        times,
        domain: domain_idxs,
        codomain: codomain_idxs,
        domain_shift: domain_shift_idxs,
    };
    debug!("Built mapping cylinder with {} simplices.", cylinder_matrix.len());
    (cylinder_matrix, metadata)
}

#[cfg(test)]
mod tests {
    use lophat::algorithms::LockFreeAlgorithm;

    use crate::all_decompositions;

    use super::*;

    fn build_inputs(matrix: Vec<(f64, usize, Vec<usize>)>) -> Vec<(f64, VecColumn)> {
        matrix
            .into_iter()
            .map(|(t, dimension, boundary)| (t, VecColumn::from((dimension, boundary))))
            .collect()
    }

    #[test]
    fn cylinder_works() {
        let domain_matrix = build_inputs(vec![
            (0.0, 0, vec![]),
            (0.0, 0, vec![]),
            (0.0, 0, vec![]),
            (0.0, 0, vec![]),
            (0.0, 1, vec![0, 1]),
            (0.0, 1, vec![1, 3]),
            (0.0, 1, vec![0, 2]),
            (0.0, 1, vec![2, 3]),
            (3.0, 2, vec![4, 5, 6, 7]),
            (4.0, 1, vec![0, 3]),
            (4.0, 2, vec![4, 5, 9]),
        ]);
        let codomain_matrix = build_inputs(vec![
            (0.0, 0, vec![]),
            (0.0, 0, vec![]),
            (0.0, 0, vec![]),
            (0.0, 0, vec![]),
            (0.0, 1, vec![0, 1]),
            (0.0, 1, vec![1, 3]),
            (0.0, 1, vec![0, 2]),
            (0.0, 1, vec![2, 3]),
            (0.4, 1, vec![0, 3]),
            (0.4, 2, vec![4, 5, 8]),
            (1.0, 2, vec![6, 7, 8]),
        ]);
        let map = vec![
            (0, vec![0]),
            (0, vec![1]),
            (0, vec![2]),
            (0, vec![3]),
            (1, vec![4]),
            (1, vec![5]),
            (1, vec![6]),
            (1, vec![7]),
            (2, vec![9, 10]), // Long square gets mapped to sum of directed triangles
            (1, vec![8]),
            (2, vec![9]),
        ]
        .into_iter()
        .map(VecColumn::from)
        .collect();
        let (cyl_matrix, metadata) = build_cylinder(domain_matrix, codomain_matrix, map);
        for (idx, (col, t)) in cyl_matrix.iter().zip(metadata.times.iter()).enumerate() {
            println!("{}:{} -> {:?}", idx, t, col);
        }
        let ensemble =
            all_decompositions::<LockFreeAlgorithm<VecColumn>>(cyl_matrix, 0).all_diagrams();
        let pairings: Vec<_> = ensemble.ker.paired.iter().collect();
        for pairing in &pairings {
            println!("{:?}", pairing);
        }
        assert_eq!(pairings.len(), 1);
        let first_pairing = pairings[0];
        let dgm_pt = (
            metadata.times[first_pairing.0],
            metadata.times[first_pairing.1],
        );
        assert_eq!(dgm_pt, (1.0, 3.0))
    }
}
