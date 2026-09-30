//! Pure Rust regression tests: no Python interpreter or external fixture files.

use crate::{
    builders::{build_d_dom, build_d_im, build_d_rel, compute_dom_first_permutation},
    cylinder::build_cylinder,
    diagrams::{DiagramEnsemble, ExtendedUsize, ExtendedUsize::*, PersistenceDiagram, from_file},
    ensemble::{all_decompositions, all_decompositions_slow, to_file},
    indexing::Permutation,
};
use lophat::{
    algorithms::{DecompositionAlgo, LockFreeAlgorithm, SerialAlgorithm},
    columns::{Column, VecColumn},
};
use std::io::{Seek, SeekFrom};

fn assert_diagrams(actual: &DiagramEnsemble, expected: [&[(usize, ExtendedUsize)]; 6]) {
    for (name, diagram, pairs) in [
        ("domain", &actual.domain, expected[0]),
        ("codomain", &actual.codomain, expected[1]),
        ("image", &actual.image, expected[2]),
        ("kernel", &actual.kernel, expected[3]),
        ("cokernel", &actual.cokernel, expected[4]),
        ("relative", &actual.relative, expected[5]),
    ] {
        assert_eq!(*diagram, pairs.iter().copied().collect(), "{name}");
    }
}

fn diagrams(
    matrix: &[Vec<usize>],
    dimensions: &[usize],
    domain: &[usize],
    slow: bool,
) -> DiagramEnsemble {
    if slow {
        all_decompositions_slow::<LockFreeAlgorithm<_>, _>(matrix, dimensions, domain, 1)
            .all_diagrams()
    } else {
        all_decompositions::<LockFreeAlgorithm<_>, _>(matrix, dimensions, domain, 1).all_diagrams()
    }
}

/// Putting the younger vertex first in row order must not change codomain pairing.
#[test]
fn either_endpoint_into_interval() {
    for slow in [false, true] {
        for vertex in [0, 1] {
            assert_diagrams(
                &diagrams(&[vec![], vec![], vec![0, 1]], &[0, 0, 1], &[vertex], slow),
                [
                    &[(vertex, Infinity)],
                    &[(0, Infinity), (1, Finite(2))],
                    &[(vertex, Infinity)],
                    &[],
                    &[(1 - vertex, Finite(2))],
                    &[(1 - vertex, Finite(2))],
                ],
            );
        }
    }
}

/// Domain columns and permutation must agree even when input domain indices are unsorted.
#[test]
fn unsorted_domain_with_finite_classes() {
    for slow in [false, true] {
        for domain in [vec![0, 1], vec![1, 0]] {
            assert_diagrams(
                &diagrams(
                    &[vec![], vec![0], vec![], vec![2]],
                    &[0, 1, 0, 1],
                    &domain,
                    slow,
                ),
                [
                    &[(0, Finite(1))],
                    &[(0, Finite(1)), (2, Finite(3))],
                    &[(0, Finite(1))],
                    &[],
                    &[(2, Finite(3))],
                    &[(2, Finite(3))],
                ],
            );
        }
    }
}

/// Relative homology has no extra basepoint; empty and identity inclusions remain valid.
#[test]
fn empty_and_identity_inclusions() {
    assert_diagrams(&diagrams(&[], &[], &[], false), [&[]; 6]);
    assert_diagrams(
        &diagrams(&[vec![]], &[0], &[], false),
        [
            &[],
            &[(0, Infinity)],
            &[],
            &[],
            &[(0, Infinity)],
            &[(0, Infinity)],
        ],
    );
    assert_diagrams(
        &diagrams(&[vec![]], &[0], &[0], false),
        [
            &[(0, Infinity)],
            &[(0, Infinity)],
            &[(0, Infinity)],
            &[],
            &[],
            &[],
        ],
    );
    assert_diagrams(
        &diagrams(&[vec![], vec![0]], &[0, 1], &[1, 0], false),
        [
            &[(0, Finite(1))],
            &[(0, Finite(1))],
            &[(0, Finite(1))],
            &[],
            &[],
            &[],
        ],
    );
}

/// Filling a boundary circle kills its image and creates an essential kernel class.
#[test]
fn essential_kernel_of_boundary_inclusion() {
    let matrix = vec![
        vec![],
        vec![],
        vec![],
        vec![0, 1],
        vec![1, 2],
        vec![0, 2],
        vec![3, 4, 5],
    ];
    for slow in [false, true] {
        assert_diagrams(
            &diagrams(&matrix, &[0, 0, 0, 1, 1, 1, 2], &[0, 1, 2, 3, 4, 5], slow),
            [
                &[(0, Infinity), (1, Finite(3)), (2, Finite(4)), (5, Infinity)],
                &[
                    (0, Infinity),
                    (1, Finite(3)),
                    (2, Finite(4)),
                    (5, Finite(6)),
                ],
                &[
                    (0, Infinity),
                    (1, Finite(3)),
                    (2, Finite(4)),
                    (5, Finite(6)),
                ],
                &[(6, Infinity)],
                &[],
                &[(6, Infinity)],
            ],
        );
    }
}

/// The tetrahedron face dies later than its image, producing a finite kernel interval.
#[test]
fn finite_kernel_and_essential_cokernel() {
    let matrix = vec![
        vec![],
        vec![],
        vec![],
        vec![],
        vec![0, 1],
        vec![0, 2],
        vec![1, 2],
        vec![0, 3],
        vec![1, 3],
        vec![2, 3],
        vec![4, 7, 8],
        vec![5, 7, 9],
        vec![6, 8, 9],
        vec![4, 5, 6],
    ];
    for slow in [false, true] {
        assert_diagrams(
            &diagrams(
                &matrix,
                &[0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2],
                &[0, 1, 2, 4, 5, 6, 13],
                slow,
            ),
            [
                &[
                    (0, Infinity),
                    (1, Finite(4)),
                    (2, Finite(5)),
                    (6, Finite(13)),
                ],
                &[
                    (0, Infinity),
                    (1, Finite(4)),
                    (2, Finite(5)),
                    (3, Finite(7)),
                    (6, Finite(12)),
                    (8, Finite(10)),
                    (9, Finite(11)),
                    (13, Infinity),
                ],
                &[
                    (0, Infinity),
                    (1, Finite(4)),
                    (2, Finite(5)),
                    (6, Finite(12)),
                ],
                &[(12, Finite(13))],
                &[
                    (3, Finite(7)),
                    (8, Finite(10)),
                    (9, Finite(11)),
                    (13, Infinity),
                ],
                &[
                    (3, Finite(7)),
                    (8, Finite(10)),
                    (9, Finite(11)),
                    (12, Infinity),
                ],
            ],
        );
    }
}

/// Builders must distinguish permuted rows, local domain columns, and quotient columns.
#[test]
fn builder_coordinates_and_permutation_roundtrip() {
    let matrix: Vec<VecColumn> = [(0, vec![]), (0, vec![]), (1, vec![0, 1]), (1, vec![1])]
        .into_iter()
        .map(VecColumn::from)
        .collect();
    let perm = compute_dom_first_permutation(4, &[1]);
    for i in 0..4 {
        assert_eq!(perm.inverse_map(perm.map(i)), i);
    }
    let entries = |columns: Vec<VecColumn>| {
        columns
            .iter()
            .map(|c| (c.dimension(), c.entries().collect::<Vec<_>>()))
            .collect::<Vec<_>>()
    };
    assert_eq!(
        entries(build_d_dom(&matrix, &[1], &perm).collect()),
        vec![(0, vec![])]
    );
    assert_eq!(
        entries(build_d_im(&matrix, &perm).collect()),
        vec![(0, vec![]), (0, vec![]), (1, vec![0, 1]), (1, vec![0])]
    );
    assert_eq!(
        entries(build_d_rel(&matrix, &perm, 1).collect()),
        vec![(0, vec![]), (1, vec![0]), (1, vec![])]
    );
    let identity = compute_dom_first_permutation(4, &[0, 1, 2, 3]);
    assert_eq!(build_d_rel(&matrix, &identity, 4).count(), 0);
}

/// A nonempty decomposition survives bincode storage; rereading requires an explicit rewind.
#[test]
fn serialization_roundtrip() {
    let reduction = SerialAlgorithm::<VecColumn>::init(None)
        .add_cols([VecColumn::from((0, vec![])), VecColumn::from((1, vec![0]))].into_iter())
        .decompose();
    let mut file = to_file(reduction);
    for _ in 0..2 {
        file.seek(SeekFrom::Start(0)).unwrap();
        assert_eq!(
            PersistenceDiagram::from_decomposition(&from_file(&file)),
            [(0, Finite(1))].into_iter().collect()
        );
    }
}

/// Check the cylinder differential itself, including the shifted boundary term and D squared zero.
#[test]
fn identity_interval_cylinder_boundary() {
    let interval = vec![(0.0, 0, vec![]), (0.0, 0, vec![]), (1.0, 1, vec![0, 1])];
    let (matrix, metadata) = build_cylinder(&interval, &interval, &[vec![0], vec![1], vec![2]]);
    assert_eq!(metadata.domain_indices, vec![0, 1, 6]);
    assert_eq!(metadata.codomain_indices, vec![2, 3, 7]);
    assert_eq!(metadata.domain_shift, vec![4, 5, 8]);
    assert_eq!(metadata.times, vec![0., 0., 0., 0., 0., 0., 1., 1., 1.]);
    assert_eq!(metadata.dimensions, vec![0, 0, 0, 0, 1, 1, 1, 1, 2]);
    assert_eq!(
        matrix,
        vec![
            vec![],
            vec![],
            vec![],
            vec![],
            vec![0, 2],
            vec![1, 3],
            vec![0, 1],
            vec![2, 3],
            vec![4, 5, 6, 7]
        ]
    );
    for (j, col) in matrix.iter().enumerate() {
        let mut twice = vec![false; matrix.len()];
        for &row in col {
            assert!(row < j);
            assert_eq!(metadata.dimensions[row] + 1, metadata.dimensions[j]);
            for &entry in &matrix[row] {
                twice[entry] ^= true;
            }
        }
        assert!(twice.iter().all(|&entry| !entry));
    }
}

/// The zero map's cone splits as codomain plus suspended domain.
#[test]
fn zero_map_cylinder() {
    let point = vec![(0.0, 0, vec![])];
    let (matrix, metadata) = build_cylinder(&point, &point, &[vec![]]);
    assert_eq!(matrix, vec![vec![], vec![], vec![0]]);
    assert_eq!(metadata.dimensions, vec![0, 0, 1]);
    let result = diagrams(
        &matrix,
        &metadata.dimensions,
        &metadata.domain_indices,
        false,
    );
    assert_diagrams(
        &result,
        [
            &[(0, Infinity)],
            &[(0, Finite(2)), (1, Infinity)],
            &[(0, Finite(2))],
            &[(2, Infinity)],
            &[(1, Infinity)],
            &[(1, Infinity), (2, Infinity)],
        ],
    );
}

/// Empty general maps produce no cells or artificial homology.
#[test]
fn empty_cylinder() {
    let empty: Vec<(f64, usize, Vec<usize>)> = vec![];
    let (matrix, metadata) = build_cylinder(&empty, &empty, &[]);
    assert!(matrix.is_empty());
    assert!(metadata.times.is_empty());
    assert!(metadata.domain_indices.is_empty());
    assert!(metadata.codomain_indices.is_empty());
    assert!(metadata.domain_shift.is_empty());
    assert!(metadata.dimensions.is_empty());
}
