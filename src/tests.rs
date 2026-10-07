//! Pure Rust regression tests: no Python interpreter or external fixture files.

use std::{
	collections::HashMap,
	io::{Seek, SeekFrom},
};

use lophat::{
	algorithms::{Decomposition, DecompositionAlgo, LockFreeAlgorithm, SerialAlgorithm},
	columns::{Column, VecColumn},
};

use crate::{
	builders::{build_d_dom, build_d_im, build_d_rel, compute_dom_first_permutation},
	cylinder::build_cylinder,
	diagrams::{DiagramEnsemble, ExtendedUsize, ExtendedUsize::*, PersistenceDiagram, from_file},
	ensemble::{FileEnsemble, all_decompositions, all_decompositions_slow, to_file},
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
	for (diagram, representatives) in [
		(&actual.domain, &actual.domain_representatives),
		(&actual.codomain, &actual.codomain_representatives),
		(&actual.image, &actual.image_representatives),
	] {
		assert_eq!(representatives.len(), diagram.len());
		for (&birth, support) in representatives {
			assert!(diagram.contains_key(&birth));
			assert_eq!(support.iter().max(), Some(&birth));
		}
	}
	let (to_codomain, to_image) = actual.get_persistence_map().unwrap();
	for (mapping, target) in [(&to_codomain, &actual.codomain), (&to_image, &actual.image)] {
		assert_eq!(mapping.len(), actual.domain.len());
		for (&birth, coefficients) in mapping {
			assert!(actual.domain.contains_key(&birth));
			for &target_birth in coefficients {
				assert!(target_birth <= birth);
				assert!(Finite(birth) < target[&target_birth]);
				assert!(target[&target_birth] <= actual.domain[&birth]);
			}
		}
	}
}

/// A younger domain vertex maps to both a finite and an essential codomain
/// feature. The next domain class maps to two finite codomain features.
#[test]
fn persistence_map_sums_codomain_features_in_both_modes() {
	for slow in [false, true] {
		let result = diagrams(
			&[vec![], vec![], vec![], vec![0, 1], vec![0, 2], vec![1, 2]],
			&[0, 0, 0, 1, 1, 1],
			&[5, 2, 1],
			slow,
		);
		let (to_codomain, to_image) = result.get_persistence_map().unwrap();
		assert_eq!(
			to_codomain,
			HashMap::from([(1, vec![0, 1]), (2, vec![1, 2])])
		);
		assert_eq!(to_image, HashMap::from([(1, vec![1]), (2, vec![2])]));
		assert_eq!(result.image[&2], Finite(4));
	}
}

/// Domain and image death reductions choose different cycle bases. A domain
/// finite feature can map to a sum of two image features, both dying earlier.
#[test]
fn persistence_map_sums_image_features_in_both_modes() {
	for slow in [false, true] {
		let result = diagrams(
			&[
				vec![],
				vec![],
				vec![],
				vec![],
				vec![1, 2],
				vec![2, 3],
				vec![1, 3],
				vec![2, 3],
			],
			&[0, 0, 0, 0, 1, 1, 1, 1],
			&[7, 3, 1, 6, 2],
			slow,
		);
		let expected = HashMap::from([(1, vec![1]), (2, vec![2]), (3, vec![2, 3])]);
		assert_eq!(
			result.get_persistence_map().unwrap(),
			(expected.clone(), expected)
		);
		assert_eq!(result.domain[&3], Finite(6));
		assert_eq!(result.image[&2], Finite(4));
		assert_eq!(result.image[&3], Finite(5));
	}
}

/// A dead target cycle is needed for exact chain expansion, then removed from
/// the homology coefficients at the source birth. Removing it before solving
/// would make the equation unsolvable.
#[test]
fn persistence_map_discards_previously_dead_targets() {
	for slow in [false, true] {
		let result = diagrams(
			&[vec![], vec![], vec![0, 1], vec![], vec![0, 3], vec![1, 3]],
			&[0, 0, 1, 0, 1, 1],
			&[5, 3, 1],
			slow,
		);
		let (to_codomain, to_image) = result.get_persistence_map().unwrap();
		assert_eq!(to_codomain, HashMap::from([(1, vec![0, 1]), (3, vec![3])]));
		assert_eq!(to_image, HashMap::from([(1, vec![1]), (3, vec![3])]));
	}
}

/// Identity maps preserve finite and essential generators. Empty domains give
/// empty maps even when their representatives are unavailable.
#[test]
fn persistence_maps_identity_and_empty() {
	for slow in [false, true] {
		let result = diagrams(&[vec![], vec![], vec![0, 1]], &[0, 0, 1], &[2, 0, 1], slow);
		let expected = HashMap::from([(0, vec![0]), (1, vec![1])]);
		assert_eq!(
			result.get_persistence_map().unwrap(),
			(expected.clone(), expected)
		);
		for result in [
			diagrams(&[], &[], &[], slow),
			diagrams(&[vec![]], &[0], &[], slow),
		] {
			assert_eq!(
				result.get_persistence_map().unwrap(),
				(HashMap::new(), HashMap::new())
			);
		}
	}
}

/// Mutable Rust ensembles must report unavailable, incomplete, non-spanning,
/// or incompatible data instead of returning a misleading persistence map.
#[test]
fn persistence_map_errors_and_normalized_supports() {
	use crate::errors::PersistenceMapError::*;
	let result = diagrams(&[vec![], vec![], vec![0, 1]], &[0, 0, 1], &[0, 1, 2], false);
	let mut modified = result.clone();
	modified.domain_representatives.clear();
	assert_eq!(
		modified.get_persistence_map(),
		Err(MissingRepresentatives { diagram: "domain" })
	);
	for target in ["codomain", "image"] {
		let mut modified = result.clone();
		let reps = if target == "codomain" {
			&mut modified.codomain_representatives
		} else {
			&mut modified.image_representatives
		};
		reps.clear();
		assert_eq!(
			modified.get_persistence_map(),
			Err(MissingRepresentatives { diagram: target })
		);
	}
	let mut modified = result.clone();
	modified.codomain_representatives.remove(&0);
	assert_eq!(
		modified.get_persistence_map(),
		Err(InconsistentData {
			diagram: "codomain"
		})
	);
	let mut modified = result.clone();
	modified.domain_representatives.insert(1, vec![0]);
	assert_eq!(
		modified.get_persistence_map(),
		Err(InconsistentData { diagram: "domain" })
	);
	let mut modified = result.clone();
	modified.codomain.remove(&0);
	modified.codomain_representatives.remove(&0);
	assert_eq!(
		modified.get_persistence_map(),
		Err(Unsolvable {
			target: "codomain",
			birth: 0
		})
	);
	let mut modified = result.clone();
	modified.codomain.insert(1, Infinity);
	assert_eq!(
		modified.get_persistence_map(),
		Err(IncompatibleIntervals {
			target: "codomain",
			birth: 1,
			target_birth: 1
		})
	);
	let mut modified = result.clone();
	for reps in [
		&mut modified.domain_representatives,
		&mut modified.codomain_representatives,
		&mut modified.image_representatives,
	] {
		for support in reps.values_mut() {
			support.reverse();
			support.push(support[0]);
		}
	}
	assert_eq!(modified.get_persistence_map(), result.get_persistence_map());
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

/// A domain with gaps and unsorted indices has local birth 3 at original index
/// 4. Its essential parallel-edge cycle becomes finite in the image. All
/// representative keys and supports must be translated, and remain cycles.
#[test]
fn representative_coordinates_and_cycles_in_both_modes() {
	let matrix = vec![vec![], vec![], vec![], vec![1, 2], vec![1, 2], vec![3, 4]];
	let expected = HashMap::from([(1, vec![1]), (2, vec![1, 2]), (4, vec![3, 4])]);
	for slow in [false, true] {
		let result = diagrams(&matrix, &[0, 0, 0, 1, 1, 2], &[4, 2, 1, 3], slow);
		assert_eq!(result.domain_representatives, expected);
		assert_eq!(result.image_representatives, expected);
		assert_eq!(result.domain[&4], Infinity);
		assert_eq!(result.image[&4], Finite(5));
		let mut codomain_expected = expected.clone();
		codomain_expected.insert(0, vec![0]);
		assert_eq!(result.codomain_representatives, codomain_expected);
		for representatives in [
			result.domain_representatives,
			result.codomain_representatives,
			result.image_representatives,
		] {
			for (birth, support) in representatives {
				assert_eq!(support.iter().max(), Some(&birth));
				let mut boundary = VecColumn::new_with_degree(0);
				for index in support {
					boundary.add_entries(matrix[index].iter().copied());
				}
				assert!(boundary.is_cycle());
			}
		}
	}
}

/// The image essential representative must remain the younger domain vertex,
/// even though image R pairs its row index and image V can use external cells.
#[test]
fn essential_image_representative_uses_domain_cycle() {
	for slow in [false, true] {
		let result = diagrams(&[vec![], vec![], vec![0, 1]], &[0, 0, 1], &[1], slow);
		assert_eq!(result.image_representatives, HashMap::from([(1, vec![1])]));
	}
}

/// Empty diagrams have empty representative maps, including when the
/// decomposition has no columns and cannot report V availability.
#[test]
fn empty_representative_maps_in_both_modes() {
	for slow in [false, true] {
		let empty = diagrams(&[], &[], &[], slow);
		assert!(empty.domain_representatives.is_empty());
		assert!(empty.codomain_representatives.is_empty());
		assert!(empty.image_representatives.is_empty());
		let result = diagrams(&[vec![]], &[0], &[], slow);
		assert!(result.domain_representatives.is_empty());
		assert_eq!(
			result.codomain_representatives,
			HashMap::from([(0, vec![0])])
		);
		assert_eq!(result.image_representatives, HashMap::new());
	}
}

/// Nonempty decompositions without V violate the extraction precondition in
/// both modes. Do not silently return diagrams with missing representatives.
#[test]
fn representatives_unavailable_without_v_in_both_modes() {
	let without_v = || {
		SerialAlgorithm::<VecColumn>::init(Some(lophat::options::LoPhatOptions {
			maintain_v: false,
			..Default::default()
		}))
		.add_cols([VecColumn::from((0, vec![]))].into_iter())
		.decompose()
	};
	let mut ensemble =
		all_decompositions::<SerialAlgorithm<_>, _>(&[vec![]], &[0], &[0], 1).unwrap();
	ensemble.d_cod = without_v();
	ensemble.d_dom = without_v();
	ensemble.d_im = without_v();
	assert!(std::panic::catch_unwind(|| ensemble.all_diagrams()).is_err());
	let stored = FileEnsemble {
		d_cod: to_file(ensemble.d_cod),
		d_dom: to_file(ensemble.d_dom),
		d_im: to_file(ensemble.d_im),
		d_cok: to_file(ensemble.d_cok),
		d_ker: to_file(ensemble.d_ker),
		d_rel: to_file(ensemble.d_rel),
		metadata: ensemble.metadata,
	};
	assert!(std::panic::catch_unwind(|| stored.all_diagrams()).is_err());
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
