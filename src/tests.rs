//! Pure Rust regression tests: no Python interpreter or external fixture files.

use std::{
	collections::{HashMap, HashSet},
	io::{Seek, SeekFrom},
};

use lophat::{
	algorithms::{Decomposition, DecompositionAlgo, LockFreeAlgorithm, SerialAlgorithm},
	columns::{Column, VecColumn},
};

use crate::{
	builders::{build_d_dom, build_d_im, build_d_rel, compute_dom_first_permutation},
	cylinder::build_cylinder,
	diagrams::{
		DiagramEnsemble,
		ExtendedUsize,
		ExtendedUsize::*,
		PersistenceDiagram,
		PersistenceMap,
		from_file,
		get_persistence_map,
	},
	ensemble::{FileEnsemble, all_decompositions, all_decompositions_slow, to_file},
	errors::PersistenceMapError,
	indexing::Permutation,
};

// Compare both inclusion targets using the private calculation and identity
// generator images.
fn persistence_maps(
	result: &DiagramEnsemble,
) -> Result<(PersistenceMap, PersistenceMap), PersistenceMapError> {
	let generator_map = result
		.domain_representatives
		.as_ref()
		.unwrap()
		.values()
		.flatten()
		.map(|&index| (index, vec![index]))
		.collect();
	Ok((
		get_persistence_map(
			&result.domain,
			result.domain_representatives.as_ref().unwrap(),
			&result.codomain,
			result.codomain_representatives.as_ref().unwrap(),
			&generator_map,
		)?,
		get_persistence_map(
			&result.domain,
			result.domain_representatives.as_ref().unwrap(),
			&result.image,
			result.image_representatives.as_ref().unwrap(),
			&generator_map,
		)?,
	))
}

// Evaluate the interval maps at each index, rather than multiplying their
// complete coefficient tables without accounting for interval lifetimes.
fn assert_image_inclusion(result: &DiagramEnsemble, n_columns: usize, expected: PersistenceMap) {
	let representatives = result.image_representatives.as_ref().unwrap();
	let identity = representatives
		.values()
		.flatten()
		.map(|&index| (index, vec![index]))
		.collect();
	let image_to_codomain = get_persistence_map(
		&result.image,
		representatives,
		&result.codomain,
		result.codomain_representatives.as_ref().unwrap(),
		&identity,
	)
	.unwrap();
	assert_eq!(image_to_codomain, expected);
	let (domain_to_codomain, domain_to_image) = persistence_maps(result).unwrap();
	let alive = |diagram: &PersistenceDiagram, birth: usize, index: usize| {
		birth <= index && Finite(index) < diagram[&birth]
	};
	for index in 0..=n_columns {
		let active_image: Vec<_> = result
			.image
			.keys()
			.copied()
			.filter(|&birth| alive(&result.image, birth, index))
			.collect();
		// Independence of the active columns verifies the monomorphism, even
		// when individual image features map to sums of codomain features.
		let mut pivots: HashMap<usize, HashSet<usize>> = HashMap::new();
		for birth in &active_image {
			let mut residual: HashSet<_> = image_to_codomain[birth]
				.iter()
				.copied()
				.filter(|&target| alive(&result.codomain, target, index))
				.collect();
			while let Some(&pivot) = residual.iter().max() {
				if let Some(column) = pivots.get(&pivot) {
					residual = residual.symmetric_difference(column).copied().collect();
				} else {
					pivots.insert(pivot, residual);
					break;
				}
			}
		}
		assert_eq!(
			pivots.len(),
			active_image.len(),
			"Image inclusion at {index}"
		);
		for &birth in result.domain.keys() {
			if !alive(&result.domain, birth, index) {
				continue;
			}
			let mut composed = HashSet::new();
			for &image_birth in &domain_to_image[&birth] {
				if !alive(&result.image, image_birth, index) {
					continue;
				}
				for &target_birth in &image_to_codomain[&image_birth] {
					if alive(&result.codomain, target_birth, index)
						&& !composed.insert(target_birth)
					{
						composed.remove(&target_birth);
					}
				}
			}
			let direct: HashSet<_> = domain_to_codomain[&birth]
				.iter()
				.copied()
				.filter(|&target| alive(&result.codomain, target, index))
				.collect();
			assert_eq!(composed, direct, "Composition for birth {birth} at {index}");
		}
	}
}

/// Image inclusion has the expected coefficients and is pointwise injective.
/// Its pointwise composition with domain-to-image equals domain-to-codomain,
/// including different cycle bases, dead targets, and an empty image.
#[test]
fn image_to_codomain_inclusions_in_both_modes() {
	for slow in [false, true] {
		for (matrix, degrees, domain, expected) in [
			(
				vec![vec![], vec![], vec![], vec![0, 1], vec![0, 2], vec![1, 2]],
				vec![0, 0, 0, 1, 1, 1],
				vec![5, 2, 1],
				HashMap::from([(1, vec![0, 1]), (2, vec![1, 2])]),
			),
			(
				vec![
					vec![],
					vec![],
					vec![],
					vec![],
					vec![1, 2],
					vec![2, 3],
					vec![1, 3],
					vec![2, 3],
				],
				vec![0, 0, 0, 0, 1, 1, 1, 1],
				vec![7, 3, 1, 6, 2],
				HashMap::from([(1, vec![1]), (2, vec![2]), (3, vec![3])]),
			),
			(
				vec![vec![], vec![], vec![0, 1], vec![], vec![0, 3], vec![1, 3]],
				vec![0, 0, 1, 0, 1, 1],
				vec![5, 3, 1],
				HashMap::from([(1, vec![0, 1]), (3, vec![3])]),
			),
			(vec![vec![]], vec![0], vec![], HashMap::new()),
		] {
			let result = diagrams(&matrix, &degrees, &domain, slow);
			assert_image_inclusion(&result, matrix.len(), expected);
		}
	}
}

/// Identity and zero general maps both factor through their cylinder image.
/// Check every cylinder index, including equal-time refinement intervals.
#[test]
fn image_to_codomain_cylinder_inclusions_in_both_modes() {
	let point = vec![(0.0, 0, vec![])];
	for slow in [false, true] {
		for mapping in [vec![vec![0]], vec![vec![]]] {
			let (result, metadata) =
				crate::sixpack(&point, &point, &mapping, true, 1, slow).unwrap();
			let birth = metadata.domain_indices[0];
			assert_image_inclusion(
				&result,
				metadata.times.len(),
				HashMap::from([(birth, vec![birth])]),
			);
		}
	}
}

/// Nonzero interval maps can compose to zero when the outer intervals do not
/// overlap. Multiplying the complete coefficient tables would retain a false
/// coefficient in a target already dead at the source birth.
#[test]
fn interval_composition_discards_dead_final_target() {
	let source = [(2, Finite(4))].into_iter().collect();
	let middle = [(1, Finite(3))].into_iter().collect();
	let target = [(0, Finite(2))].into_iter().collect();
	let source_reps = HashMap::from([(2, vec![2])]);
	let middle_reps = HashMap::from([(1, vec![1])]);
	let target_reps = HashMap::from([(0, vec![0])]);
	let first = get_persistence_map(
		&source,
		&source_reps,
		&middle,
		&middle_reps,
		&HashMap::from([(2, vec![1])]),
	)
	.unwrap();
	let second = get_persistence_map(
		&middle,
		&middle_reps,
		&target,
		&target_reps,
		&HashMap::from([(1, vec![0])]),
	)
	.unwrap();
	let direct = get_persistence_map(
		&source,
		&source_reps,
		&target,
		&target_reps,
		&HashMap::from([(2, vec![0])]),
	)
	.unwrap();
	assert_eq!(first, HashMap::from([(2, vec![1])]));
	assert_eq!(second, HashMap::from([(1, vec![0])]));
	assert_eq!(direct, HashMap::from([(2, vec![])]));
	for index in 0..=4 {
		if index < 2 || index >= 4 {
			continue;
		}
		let mut composed = HashSet::new();
		for &mid in &first[&2] {
			if Finite(index) >= middle[&mid] {
				continue;
			}
			for &end in &second[&mid] {
				if Finite(index) < target[&end] {
					assert!(composed.insert(end));
				}
			}
		}
		assert!(composed.is_empty());
	}
}

// Independent small-matrix rank oracle for homology assertions: boundary
// columns and representatives are supplied directly in the original basis.
pub(super) fn binary_rank(columns: impl IntoIterator<Item = HashSet<usize>>) -> usize {
	let mut pivots: HashMap<usize, HashSet<usize>> = HashMap::new();
	for mut residual in columns {
		while let Some(&pivot) = residual.iter().max() {
			if let Some(column) = pivots.get(&pivot) {
				residual = residual.symmetric_difference(column).copied().collect();
			} else {
				pivots.insert(pivot, residual);
				break;
			}
		}
	}
	pivots.len()
}

fn assert_kernel_cycles(
	result: &DiagramEnsemble,
	matrix: &[Vec<usize>],
	degrees: &[usize],
	domain: &[usize],
) {
	let representatives = result.kernel_representatives.as_ref().unwrap();
	assert_eq!(representatives.len(), result.kernel.len());
	for (&birth, support) in representatives {
		assert!(result.kernel.contains_key(&birth));
		assert!(!support.is_empty());
		assert!(
			support
				.iter()
				.all(|&idx| idx < birth && domain.contains(&idx))
		);
		assert!(
			support
				.iter()
				.all(|&idx| degrees[idx] + 1 == degrees[birth])
		);
		let mut boundary = HashSet::new();
		for &idx in support {
			for &row in &matrix[idx] {
				if !boundary.insert(row) {
					boundary.remove(&row);
				}
			}
		}
		assert!(boundary.is_empty(), "Kernel cycle at birth {birth}");
	}
	let identity = representatives
		.values()
		.flatten()
		.map(|&idx| (idx, vec![idx]))
		.collect();
	let kernel_to_domain = get_persistence_map(
		&result.kernel,
		representatives,
		&result.domain,
		result.domain_representatives.as_ref().unwrap(),
		&identity,
	)
	.unwrap();
	let (domain_to_codomain, _) = persistence_maps(result).unwrap();
	let alive = |diagram: &PersistenceDiagram, birth: usize, index: usize| {
		birth <= index && Finite(index) < diagram[&birth]
	};
	for index in 0..=matrix.len() {
		let kernel_columns: Vec<HashSet<_>> = representatives
			.iter()
			.filter(|&(&birth, _)| alive(&result.kernel, birth, index))
			.map(|(_, support)| support.iter().copied().collect())
			.collect();
		let domain_boundaries: Vec<HashSet<_>> = domain
			.iter()
			.filter(|&&idx| idx <= index)
			.map(|&idx| matrix[idx].iter().copied().collect())
			.collect();
		let codomain_boundaries: Vec<HashSet<_>> = matrix
			.iter()
			.take(index + 1)
			.map(|column| column.iter().copied().collect())
			.collect();
		assert_eq!(
			binary_rank(domain_boundaries.iter().chain(&kernel_columns).cloned()),
			binary_rank(domain_boundaries) + kernel_columns.len(),
			"Independent domain homology classes at {index}",
		);
		assert_eq!(
			binary_rank(codomain_boundaries.iter().chain(&kernel_columns).cloned()),
			binary_rank(codomain_boundaries),
			"Kernel cycles bound in codomain at {index}",
		);
		let active_count = |diagram: &PersistenceDiagram| {
			diagram
				.keys()
				.filter(|&&birth| alive(diagram, birth, index))
				.count()
		};
		assert_eq!(
			kernel_columns.len(),
			active_count(&result.domain) - active_count(&result.image),
		);
		let mut inclusion_columns = Vec::new();
		for (&birth, coefficients) in &kernel_to_domain {
			if !alive(&result.kernel, birth, index) {
				continue;
			}
			let column: HashSet<_> = coefficients
				.iter()
				.copied()
				.filter(|&idx| alive(&result.domain, idx, index))
				.collect();
			let mut composed = HashSet::new();
			for &domain_birth in &column {
				for &target_birth in &domain_to_codomain[&domain_birth] {
					if alive(&result.codomain, target_birth, index)
						&& !composed.insert(target_birth)
					{
						composed.remove(&target_birth);
					}
				}
			}
			assert!(composed.is_empty(), "Kernel composite at {index}");
			inclusion_columns.push(column);
		}
		assert_eq!(binary_rank(inclusion_columns), kernel_columns.len());
	}
}

/// Kernel cycles live in the domain and bound in the codomain throughout
/// their intervals. Finite representatives follow kernel death pairings,
/// including a death cycle differing from the image R at its kernel birth.
#[test]
fn kernel_representatives_and_domain_inclusion_in_both_modes() {
	for slow in [false, true] {
		for (matrix, degrees, domain, expected) in [
			(
				vec![vec![], vec![], vec![], vec![0, 1], vec![0, 2], vec![1, 2]],
				vec![0, 0, 0, 1, 1, 1],
				vec![5, 2, 1],
				HashMap::from([(4, vec![1, 2])]),
			),
			(
				vec![
					vec![],
					vec![],
					vec![],
					vec![0, 1],
					vec![0, 2],
					vec![1, 2],
					vec![0, 1],
				],
				vec![0, 0, 0, 1, 1, 1, 1],
				vec![6, 2, 0, 5, 1],
				HashMap::from([(3, vec![0, 1]), (4, vec![1, 2])]),
			),
			(
				vec![
					vec![],
					vec![],
					vec![],
					vec![0, 1],
					vec![1, 2],
					vec![0, 2],
					vec![3, 4, 5],
				],
				vec![0, 0, 0, 1, 1, 1, 2],
				vec![0, 1, 2, 3, 4, 5],
				HashMap::from([(6, vec![3, 4, 5])]),
			),
		] {
			let result = diagrams(&matrix, &degrees, &domain, slow);
			assert_eq!(result.kernel_representatives.as_ref().unwrap(), &expected);
			assert_kernel_cycles(&result, &matrix, &degrees, &domain);
		}
		let point = vec![(0.0, 0, vec![])];
		let (matrix, metadata) = build_cylinder(&point, &point, &[vec![]]);
		let result = diagrams(&matrix, &metadata.degrees, &metadata.domain_indices, slow);
		assert_eq!(
			result.kernel_representatives,
			Some(HashMap::from([(2, vec![0])]))
		);
		assert_kernel_cycles(
			&result,
			&matrix,
			&metadata.degrees,
			&metadata.domain_indices,
		);
	}
}

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
	if actual.domain_representatives.is_none() {
		assert!(actual.codomain_representatives.is_none());
		assert!(actual.image_representatives.is_none());
		assert!(actual.kernel_representatives.is_none());
		assert!(actual.cokernel_representatives.is_none());
		return;
	}
	for (diagram, representatives) in [
		(
			&actual.domain,
			actual.domain_representatives.as_ref().unwrap(),
		),
		(
			&actual.codomain,
			actual.codomain_representatives.as_ref().unwrap(),
		),
		(
			&actual.image,
			actual.image_representatives.as_ref().unwrap(),
		),
		(
			&actual.kernel,
			actual.kernel_representatives.as_ref().unwrap(),
		),
		(
			&actual.cokernel,
			actual.cokernel_representatives.as_ref().unwrap(),
		),
	] {
		assert_eq!(representatives.len(), diagram.len());
		for (&birth, support) in representatives {
			assert!(diagram.contains_key(&birth));
			if std::ptr::eq(diagram, &actual.kernel) {
				assert!(support.iter().max() < Some(&birth));
			} else {
				assert_eq!(support.iter().max(), Some(&birth));
			}
		}
	}
	let (to_codomain, to_image) = persistence_maps(actual).unwrap();
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
	let representatives = actual.kernel_representatives.as_ref().unwrap();
	let identity = representatives
		.values()
		.flatten()
		.map(|&idx| (idx, vec![idx]))
		.collect();
	get_persistence_map(
		&actual.kernel,
		representatives,
		&actual.domain,
		actual.domain_representatives.as_ref().unwrap(),
		&identity,
	)
	.unwrap();
	let zero = get_persistence_map(
		&actual.kernel,
		representatives,
		&actual.codomain,
		actual.codomain_representatives.as_ref().unwrap(),
		&identity,
	)
	.unwrap();
	assert!(zero.values().all(Vec::is_empty));
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
		let (to_codomain, to_image) = persistence_maps(&result).unwrap();
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
			persistence_maps(&result).unwrap(),
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
		let (to_codomain, to_image) = persistence_maps(&result).unwrap();
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
			persistence_maps(&result).unwrap(),
			(expected.clone(), expected)
		);
		for result in [
			diagrams(&[], &[], &[], slow),
			diagrams(&[vec![]], &[0], &[], slow),
		] {
			assert_eq!(
				persistence_maps(&result).unwrap(),
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
	modified.domain_representatives.as_mut().unwrap().clear();
	assert_eq!(
		persistence_maps(&modified),
		Err(MissingRepresentatives { diagram: "source" })
	);
	for target in ["codomain", "image"] {
		let mut modified = result.clone();
		let reps = if target == "codomain" {
			modified.codomain_representatives.as_mut().unwrap()
		} else {
			modified.image_representatives.as_mut().unwrap()
		};
		reps.clear();
		assert_eq!(
			persistence_maps(&modified),
			Err(MissingRepresentatives { diagram: "target" })
		);
	}
	let mut modified = result.clone();
	modified
		.codomain_representatives
		.as_mut()
		.unwrap()
		.remove(&0);
	assert_eq!(
		persistence_maps(&modified),
		Err(InconsistentData { diagram: "target" })
	);
	let mut modified = result.clone();
	modified
		.domain_representatives
		.as_mut()
		.unwrap()
		.insert(1, vec![2]);
	assert_eq!(
		persistence_maps(&modified),
		Err(InconsistentData { diagram: "source" })
	);
	let mut modified = result.clone();
	modified.codomain.remove(&0);
	modified
		.codomain_representatives
		.as_mut()
		.unwrap()
		.remove(&0);
	assert_eq!(
		persistence_maps(&modified),
		Err(Unsolvable {
			target: "target",
			birth: 0
		})
	);
	let mut modified = result.clone();
	modified.codomain.insert(1, Infinity);
	assert_eq!(
		persistence_maps(&modified),
		Err(IncompatibleIntervals {
			target: "target",
			birth: 1,
			target_birth: 1
		})
	);
	let mut modified = result.clone();
	for reps in [
		modified.domain_representatives.as_mut().unwrap(),
		modified.codomain_representatives.as_mut().unwrap(),
		modified.image_representatives.as_mut().unwrap(),
	] {
		for support in reps.values_mut() {
			support.reverse();
			support.push(support[0]);
		}
	}
	assert_eq!(persistence_maps(&modified), persistence_maps(&result));
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
			.all_diagrams(true)
	} else {
		all_decompositions::<LockFreeAlgorithm<_>, _>(matrix, degrees, domain, 1)
			.unwrap()
			.all_diagrams(true)
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
	assert_eq!(ensemble.all_diagrams(true).codomain, expected);

	let mut stored =
		all_decompositions_slow::<LockFreeAlgorithm<_>, _>(&matrix, &degrees, &[1], 1).unwrap();
	check_codomain(&from_file(&stored.d_cod), &matrix, &degrees);
	stored.d_cod.rewind().unwrap();
	assert_eq!(stored.all_diagrams(true).codomain, expected);
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
		assert_eq!(result.domain_representatives, Some(expected.clone()));
		assert_eq!(result.image_representatives, Some(expected.clone()));
		assert_eq!(result.domain[&4], Infinity);
		assert_eq!(result.image[&4], Finite(5));
		let mut codomain_expected = expected.clone();
		codomain_expected.insert(0, vec![0]);
		assert_eq!(result.codomain_representatives, Some(codomain_expected));
		for representatives in [
			result.domain_representatives.as_ref().unwrap(),
			result.codomain_representatives.as_ref().unwrap(),
			result.image_representatives.as_ref().unwrap(),
		] {
			for (&birth, support) in representatives {
				assert_eq!(support.iter().max(), Some(&birth));
				let mut boundary = VecColumn::new_with_degree(0);
				for &index in support {
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
		assert_eq!(
			result.image_representatives,
			Some(HashMap::from([(1, vec![1])]))
		);
	}
}

/// Empty diagrams have empty representative maps, including when the
/// decomposition has no columns and cannot report V availability.
#[test]
fn empty_representative_maps_in_both_modes() {
	for slow in [false, true] {
		let empty = diagrams(&[], &[], &[], slow);
		assert!(empty.domain_representatives.as_ref().unwrap().is_empty());
		assert!(empty.codomain_representatives.as_ref().unwrap().is_empty());
		assert!(empty.image_representatives.as_ref().unwrap().is_empty());
		assert!(empty.kernel_representatives.as_ref().unwrap().is_empty());
		assert!(empty.cokernel_representatives.as_ref().unwrap().is_empty());
		let result = diagrams(&[vec![]], &[0], &[], slow);
		assert!(result.domain_representatives.as_ref().unwrap().is_empty());
		assert_eq!(
			result.codomain_representatives,
			Some(HashMap::from([(0, vec![0])]))
		);
		assert_eq!(result.image_representatives, Some(HashMap::new()));
		assert_eq!(result.kernel_representatives, Some(HashMap::new()));
		assert_eq!(
			result.cokernel_representatives,
			Some(HashMap::from([(0, vec![0])]))
		);
	}
}

/// Disabling extraction works without V in each relevant decomposition, both
/// in memory and after serialization. Opting in must still reject missing V.
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
	for missing in ["domain", "codomain", "image"] {
		let mut ensemble =
			all_decompositions::<SerialAlgorithm<_>, _>(&[vec![]], &[0], &[0], 1).unwrap();
		match missing {
			"domain" => ensemble.d_dom = without_v(),
			"codomain" => ensemble.d_cod = without_v(),
			"image" => ensemble.d_im = without_v(),
			_ => unreachable!(),
		}
		let diagrams = ensemble.all_diagrams(false);
		assert_eq!(diagrams.domain[&0], Infinity);
		assert!(diagrams.domain_representatives.is_none());
		assert!(diagrams.codomain_representatives.is_none());
		assert!(diagrams.image_representatives.is_none());
		assert!(diagrams.kernel_representatives.is_none());
		assert!(diagrams.cokernel_representatives.is_none());
		assert!(std::panic::catch_unwind(|| ensemble.all_diagrams(true)).is_err());
		let mut stored = FileEnsemble {
			d_cod: to_file(ensemble.d_cod),
			d_dom: to_file(ensemble.d_dom),
			d_im: to_file(ensemble.d_im),
			d_cok: to_file(ensemble.d_cok),
			d_ker: to_file(ensemble.d_ker),
			d_rel: to_file(ensemble.d_rel),
			metadata: ensemble.metadata,
		};
		let diagrams = stored.all_diagrams(false);
		assert_eq!(diagrams.image[&0], Infinity);
		assert!(diagrams.domain_representatives.is_none());
		assert!(diagrams.codomain_representatives.is_none());
		assert!(diagrams.image_representatives.is_none());
		assert!(diagrams.kernel_representatives.is_none());
		assert!(diagrams.cokernel_representatives.is_none());
		for file in [
			&mut stored.d_cod,
			&mut stored.d_dom,
			&mut stored.d_im,
			&mut stored.d_cok,
			&mut stored.d_ker,
			&mut stored.d_rel,
		] {
			file.rewind().unwrap();
		}
		assert!(std::panic::catch_unwind(|| stored.all_diagrams(true)).is_err());
	}
}

/// The flag changes only representative availability, including empty inputs
/// and a permuted domain with finite and essential cycles, in both modes.
#[test]
fn optional_representatives_preserve_all_diagrams() {
	let cases = [
		(vec![], vec![], vec![]),
		(vec![vec![]], vec![0], vec![]),
		(vec![vec![]], vec![0], vec![0]),
		(
			vec![vec![], vec![], vec![], vec![1, 2], vec![1, 2], vec![3, 4]],
			vec![0, 0, 0, 1, 1, 2],
			vec![4, 2, 1, 3],
		),
	];
	for (matrix, degrees, domain) in cases {
		for slow in [false, true] {
			let plain =
				crate::sixpack_from_inclusion(&matrix, &degrees, &domain, false, 1, slow).unwrap();
			let with_reps =
				crate::sixpack_from_inclusion(&matrix, &degrees, &domain, true, 1, slow).unwrap();
			assert!(plain.domain_representatives.is_none());
			assert!(plain.codomain_representatives.is_none());
			assert!(plain.image_representatives.is_none());
			assert!(plain.kernel_representatives.is_none());
			assert!(plain.cokernel_representatives.is_none());
			for (plain, requested) in [
				(&plain.domain, &with_reps.domain),
				(&plain.codomain, &with_reps.codomain),
				(&plain.image, &with_reps.image),
				(&plain.kernel, &with_reps.kernel),
				(&plain.cokernel, &with_reps.cokernel),
				(&plain.relative, &with_reps.relative),
			] {
				assert_eq!(plain, requested);
			}
			for (diagram, representatives) in [
				(&with_reps.domain, &with_reps.domain_representatives),
				(&with_reps.codomain, &with_reps.codomain_representatives),
				(&with_reps.image, &with_reps.image_representatives),
				(&with_reps.kernel, &with_reps.kernel_representatives),
				(&with_reps.cokernel, &with_reps.cokernel_representatives),
			] {
				assert_eq!(representatives.as_ref().unwrap().len(), diagram.len());
			}
		}
	}
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
	assert_diagrams(
		&diagrams(&[], &[], &[], false),
		[&[] as &[(usize, ExtendedUsize)]; 6],
	);
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
				.expect_err("Invalid input must fail before serialization");
		assert_eq!(fast, expected);
		assert_eq!(slow, expected);
		assert!(!expected.to_string().is_empty());
		for slow in [false, true] {
			assert_eq!(
				sixpack_from_inclusion(&matrix, &degrees, &domain, false, 1, slow).unwrap_err(),
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
		let result =
			crate::sixpack_from_inclusion(&matrix, &[0, 0, 1], &domain, false, 1, slow).unwrap();
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
		let error = crate::sixpack(&malformed, &malformed, &[vec![0], vec![1]], false, 1, slow)
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
	for (slow, get_representatives) in [(false, false), (false, true), (true, false), (true, true)]
	{
		let (result, metadata) =
			crate::sixpack(&point, &point, &[vec![0]], get_representatives, 1, slow).unwrap();
		assert_eq!(result.domain_representatives.is_some(), get_representatives);
		assert_eq!(
			result.codomain_representatives.is_some(),
			get_representatives
		);
		assert_eq!(result.image_representatives.is_some(), get_representatives);
		assert_eq!(result.kernel_representatives.is_some(), get_representatives);
		assert_eq!(
			result.cokernel_representatives.is_some(),
			get_representatives
		);
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
