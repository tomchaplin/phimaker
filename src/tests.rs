//! Pure Rust regression tests: no Python interpreter or external fixture files.

use std::io::{Seek, SeekFrom};

use lophat::{
	algorithms::{Decomposition, DecompositionAlgo, LockFreeAlgorithm, SerialAlgorithm},
	columns::{Column, VecColumn},
};

use crate::{
	builders::{build_d_dom, build_d_im, build_d_rel, compute_dom_first_permutation},
	cylinder::build_cylinder,
	diagrams::{DiagramEnsemble, ExtendedUsize, ExtendedUsize::*, PersistenceDiagram, from_file},
	ensemble::{all_decompositions, all_decompositions_slow, to_file},
	indexing::Permutation,
};

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
	degrees: &[usize],
	domain: &[usize],
	slow: bool,
) -> DiagramEnsemble {
	if slow {
		all_decompositions_slow::<LockFreeAlgorithm<_>, _>(matrix, degrees, domain, 1)
			.unwrap()
			.all_diagrams()
	} else {
		all_decompositions::<LockFreeAlgorithm<_>, _>(matrix, degrees, domain, 1)
			.unwrap()
			.all_diagrams()
	}
}

/// Putting the younger vertex first in row order must not change codomain
/// pairing.
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

/// Forward codomain R and retained V use original generator indices in memory
/// and after serialization. Parallel edges give a nontrivial essential cycle.
#[test]
fn codomain_coordinates_and_representatives_in_both_modes() {
	fn check_codomain(
		codomain: &impl Decomposition<VecColumn>,
		matrix: &[Vec<usize>],
		degrees: &[usize],
	) {
		assert_eq!(codomain.n_cols(), matrix.len());
		assert_eq!(codomain.has_v(), Ok(()));
		for (column, &degree) in degrees.iter().enumerate() {
			let v = codomain.get_v_col(column).unwrap();
			let mut reconstructed = VecColumn::new_with_degree(degree);
			for index in v.entries() {
				reconstructed.add_entries(matrix[index].iter().copied());
			}
			assert_eq!(
				*codomain.get_r_col(column),
				reconstructed,
				"R = D V at {column}"
			);
		}
		assert_eq!(
			codomain.get_r_col(2).entries().collect::<Vec<_>>(),
			vec![0, 1]
		);
		assert_eq!(
			codomain.get_v_col(3).unwrap().entries().collect::<Vec<_>>(),
			vec![2, 3]
		);
	}

	let matrix = vec![vec![], vec![], vec![0, 1], vec![0, 1]];
	let degrees = [0, 0, 1, 1];
	let expected: PersistenceDiagram = [(0, Infinity), (1, Finite(2)), (3, Infinity)]
		.into_iter()
		.collect();
	let ensemble =
		all_decompositions::<LockFreeAlgorithm<_>, _>(&matrix, &degrees, &[1], 1).unwrap();
	check_codomain(&ensemble.d_cod, &matrix, &degrees);
	assert_eq!(ensemble.all_diagrams().codomain, expected);

	let mut stored =
		all_decompositions_slow::<LockFreeAlgorithm<_>, _>(&matrix, &degrees, &[1], 1).unwrap();
	check_codomain(&from_file(&stored.d_cod), &matrix, &degrees);
	stored.d_cod.rewind().unwrap();
	assert_eq!(stored.all_diagrams().codomain, expected);
}

/// Domain columns and permutation must agree even when input domain indices are
/// unsorted.
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

/// Relative homology has no extra basepoint; empty and identity inclusions
/// remain valid.
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

/// Filling a boundary circle kills its image and creates an essential kernel
/// class.
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

/// The tetrahedron face dies later than its image, producing a finite kernel
/// interval.
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

/// Builders must distinguish permuted rows, local domain columns, and quotient
/// columns.
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
			.map(|c| (c.degree(), c.entries().collect::<Vec<_>>()))
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

/// A nonempty decomposition survives bincode storage; rereading requires an
/// explicit rewind.
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

/// Check the cylinder differential itself, including the shifted boundary term
/// and D squared zero.
#[test]
fn identity_interval_cylinder_boundary() {
	let interval = vec![(0.0, 0, vec![]), (0.0, 0, vec![]), (1.0, 1, vec![0, 1])];
	let (matrix, metadata) = build_cylinder(&interval, &interval, &[vec![0], vec![1], vec![2]]);
	assert_eq!(metadata.domain_indices, vec![0, 1, 6]);
	assert_eq!(metadata.codomain_indices, vec![2, 3, 7]);
	assert_eq!(metadata.domain_shift, vec![4, 5, 8]);
	assert_eq!(metadata.times, vec![0., 0., 0., 0., 0., 0., 1., 1., 1.]);
	assert_eq!(metadata.degrees, vec![0, 0, 0, 0, 1, 1, 1, 1, 2]);
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
			assert_eq!(metadata.degrees[row] + 1, metadata.degrees[j]);
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
	assert_eq!(metadata.degrees, vec![0, 0, 1]);
	let result = diagrams(&matrix, &metadata.degrees, &metadata.domain_indices, false);
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
	assert!(metadata.degrees.is_empty());
}

/// All Rust entry points reject malformed structure with the same typed error,
/// before slow mode can reach the upstream empty-decomposition serializer.
#[test]
fn inclusion_validation_errors_are_shared() {
	use InclusionError::*;

	use crate::{InclusionError, sixpack_from_inclusion};

	let cases = [
		(
			vec![vec![]],
			vec![],
			vec![],
			DegreeCount {
				expected: 1,
				actual: 0,
			},
		),
		(
			vec![vec![]],
			vec![0, 1],
			vec![],
			DegreeCount {
				expected: 1,
				actual: 2,
			},
		),
		(
			vec![vec![]],
			vec![0],
			vec![1],
			DomainIndexOutOfBounds {
				index: 1,
				columns: 1,
			},
		),
		(
			vec![],
			vec![],
			vec![0],
			DomainIndexOutOfBounds {
				index: 0,
				columns: 0,
			},
		),
		(
			vec![vec![0]],
			vec![0],
			vec![],
			NonUpperTriangular { column: 0, row: 0 },
		),
		(
			vec![vec![], vec![usize::MAX]],
			vec![0, 1],
			vec![],
			NonUpperTriangular {
				column: 1,
				row: usize::MAX,
			},
		),
		(
			vec![vec![], vec![0]],
			vec![0, 0],
			vec![],
			InvalidBoundaryDegree {
				column: 1,
				row: 0,
				degree: 0,
				boundary_degree: 0,
			},
		),
		(
			vec![vec![], vec![0]],
			vec![usize::MAX, 0],
			vec![],
			InvalidBoundaryDegree {
				column: 1,
				row: 0,
				degree: 0,
				boundary_degree: usize::MAX,
			},
		),
		(
			vec![vec![], vec![0]],
			vec![0, 1],
			vec![1],
			DomainNotClosed { column: 1, row: 0 },
		),
	];
	for (matrix, degrees, domain, expected) in cases {
		let fast = all_decompositions::<LockFreeAlgorithm<_>, _>(&matrix, &degrees, &domain, 1)
			.err()
			.expect("Invalid input must fail before reduction");
		let slow =
			all_decompositions_slow::<LockFreeAlgorithm<_>, _>(&matrix, &degrees, &domain, 1)
				.err()
				.expect("Invalid input must fail before serialization");
		assert_eq!(fast, expected);
		assert_eq!(slow, expected);
		assert!(!expected.to_string().is_empty());
		for slow in [false, true] {
			assert_eq!(
				sixpack_from_inclusion(&matrix, &degrees, &domain, 1, slow).unwrap_err(),
				expected
			);
		}
	}
}

/// Unsorted and repeated support indices mean the same nonzero entry, rather
/// than repeated summands over F2. Normalization never mutates borrowed inputs.
#[test]
fn rust_inclusion_normalizes_supports_without_mutating_inputs() {
	let matrix = vec![vec![], vec![], vec![1, 0, 1]];
	let domain = vec![1, 1];
	for slow in [false, true] {
		let result = crate::sixpack_from_inclusion(&matrix, &[0, 0, 1], &domain, 1, slow).unwrap();
		assert_diagrams(
			&result,
			[
				&[(1, Infinity)],
				&[(0, Infinity), (1, Finite(2))],
				&[(1, Infinity)],
				&[],
				&[(0, Finite(2))],
				&[(0, Finite(2))],
			],
		);
	}
	assert_eq!(matrix, vec![vec![], vec![], vec![1, 0, 1]]);
	assert_eq!(domain, vec![1, 1]);
	let ensemble =
		all_decompositions::<LockFreeAlgorithm<_>, _>(&matrix, &[0, 0, 1], &domain, 1).unwrap();
	assert_eq!(ensemble.metadata.cols_in_dom, vec![1]);
	assert_eq!(ensemble.metadata.sz_domain, 1);
}

/// The general-map Rust API delegates structural errors from the cylinder
/// inclusion.
#[test]
fn rust_general_map_propagates_validation_errors() {
	let malformed = vec![(0.0, 0, vec![]), (1.0, 0, vec![0])];
	for slow in [false, true] {
		// This input deliberately violates the degree requirement of a chain
		// complex.
		let error = crate::sixpack(&malformed, &malformed, &[vec![0], vec![1]], 1, slow)
			.err()
			.expect("The malformed cylinder must be rejected");
		assert!(matches!(
			error,
			crate::InclusionError::InvalidBoundaryDegree { .. }
		));
	}
}

/// Rust callers can use the general-map convenience function without Python.
#[test]
fn rust_general_map_identity_in_both_modes() {
	let point = vec![(0.0, 0, vec![])];
	for slow in [false, true] {
		let (result, metadata) = crate::sixpack(&point, &point, &[vec![0]], 1, slow).unwrap();
		assert_eq!(metadata.domain_indices, vec![0]);
		assert_eq!(metadata.codomain_indices, vec![1]);
		assert_eq!(metadata.domain_shift, vec![2]);
		assert_diagrams(
			&result,
			[
				&[(0, Infinity)],
				&[(0, Infinity), (1, Finite(2))],
				&[(0, Infinity)],
				&[],
				&[(1, Finite(2))],
				&[(1, Finite(2))],
			],
		);
	}
}
