//! Cokernel representatives are codomain cycles modulo boundaries and domain
//! cycles.

use std::collections::{HashMap, HashSet};

use lophat::{
	algorithms::{Decomposition, DecompositionAlgo, SerialAlgorithm},
	columns::{Column, VecColumn},
	options::LoPhatOptions,
};

use crate::{
	all_decompositions,
	cylinder::build_cylinder,
	diagrams::{
		DiagramEnsemble,
		ExtendedUsize::{Finite, Infinity},
	},
	sixpack_from_inclusion,
	tests::binary_rank,
};

// Build the quotient relation space independently from the input boundaries
// and all domain cycles available by this index, including dead domain
// features.
fn quotient_relations(
	result: &DiagramEnsemble,
	matrix: &[Vec<usize>],
	index: usize,
) -> Vec<HashSet<usize>> {
	matrix
		.iter()
		.take(index + 1)
		.map(|column| column.iter().copied().collect())
		.chain(
			result
				.domain_representatives
				.as_ref()
				.unwrap()
				.iter()
				.filter(|&(&birth, _)| birth <= index)
				.map(|(_, support)| support.iter().copied().collect()),
		)
		.collect()
}

fn assert_cokernel_cycles(result: &DiagramEnsemble, matrix: &[Vec<usize>], degrees: &[usize]) {
	let representatives = result.cokernel_representatives.as_ref().unwrap();
	assert_eq!(representatives.len(), result.cokernel.len());
	for (&birth, support) in representatives {
		assert!(result.cokernel.contains_key(&birth));
		assert_eq!(support.iter().max(), Some(&birth));
		assert_eq!(
			support.iter().copied().collect::<HashSet<_>>().len(),
			support.len()
		);
		assert!(support.iter().all(|&idx| degrees[idx] == degrees[birth]));
		let mut boundary = HashSet::new();
		for &idx in support {
			for &row in &matrix[idx] {
				if !boundary.insert(row) {
					boundary.remove(&row);
				}
			}
		}
		assert!(boundary.is_empty(), "Cokernel cycle at birth {birth}");
		if let Finite(death) = result.cokernel[&birth] {
			let mut relations = quotient_relations(result, matrix, death);
			let rank = binary_rank(relations.clone());
			relations.push(support.iter().copied().collect());
			assert_eq!(
				binary_rank(relations),
				rank,
				"Cokernel cycle vanishes at {death}"
			);
		}
	}
	for index in 0..=matrix.len() {
		let relations = quotient_relations(result, matrix, index);
		let rank = binary_rank(relations.clone());
		let active: Vec<HashSet<_>> = representatives
			.iter()
			.filter(|&(&birth, _)| birth <= index && Finite(index) < result.cokernel[&birth])
			.map(|(_, support)| support.iter().copied().collect())
			.collect();
		assert_eq!(
			binary_rank(relations.into_iter().chain(active.iter().cloned())),
			rank + active.len(),
			"Independent quotient classes at {index}"
		);
		let cycle_dimension = result
			.codomain
			.keys()
			.filter(|&&birth| birth <= index)
			.count();
		assert_eq!(
			rank + active.len(),
			cycle_dimension,
			"Complete quotient basis at {index}"
		);
	}
	// All image and cokernel lifts together span the full codomain cycle space.
	let image = result.image_representatives.as_ref().unwrap();
	assert!(
		image
			.keys()
			.all(|birth| !representatives.contains_key(birth))
	);
	assert_eq!(
		binary_rank(
			image
				.values()
				.chain(representatives.values())
				.map(|support| support.iter().copied().collect())
		),
		result.codomain.len()
	);
}

/// Finite death bases, essential lifts, empty/identity inclusions, and unsorted
/// domain coordinates all give a basis of the quotient at each index, in both
/// modes.
#[test]
fn cokernel_cycles_form_quotient_bases_in_both_modes() {
	let cases = [
		(vec![], vec![], vec![], HashMap::new()),
		(vec![vec![]], vec![0], vec![], HashMap::from([(0, vec![0])])),
		(
			vec![vec![], vec![], vec![0, 1]],
			vec![0, 0, 1],
			vec![2, 1, 0],
			HashMap::new(),
		),
		(
			vec![vec![], vec![], vec![0, 1]],
			vec![0, 0, 1],
			vec![1],
			HashMap::from([(0, vec![0])]),
		),
		(
			vec![vec![], vec![], vec![], vec![0, 2]],
			vec![0, 0, 0, 1],
			vec![1],
			HashMap::from([(0, vec![0]), (2, vec![0, 2])]),
		),
		(
			vec![vec![], vec![], vec![0, 1], vec![0, 1]],
			vec![0, 0, 1, 1],
			vec![3, 1, 0],
			HashMap::from([(3, vec![2, 3])]),
		),
		(
			vec![vec![], vec![], vec![0, 1], vec![0, 1], vec![2, 3]],
			vec![0, 0, 1, 1, 2],
			vec![0, 1, 3],
			HashMap::from([(3, vec![2, 3])]),
		),
		(
			vec![vec![], vec![], vec![], vec![1, 2], vec![1, 2], vec![3, 4]],
			vec![0, 0, 0, 1, 1, 2],
			vec![4, 2, 1, 3],
			HashMap::from([(0, vec![0])]),
		),
	];
	for (matrix, degrees, domain, expected) in cases {
		for slow in [false, true] {
			let result = sixpack_from_inclusion(&matrix, &degrees, &domain, true, 1, slow).unwrap();
			assert_eq!(result.cokernel_representatives.as_ref().unwrap(), &expected);
			assert_cokernel_cycles(&result, &matrix, &degrees);
		}
	}
}

/// A younger domain vertex makes the older vertex zero in the cokernel when
/// they merge, although the older vertex remains a nonzero codomain H0 class.
#[test]
fn finite_cokernel_cycle_need_not_be_a_codomain_boundary_at_death() {
	for slow in [false, true] {
		let result = sixpack_from_inclusion(
			&[vec![], vec![], vec![0, 1]],
			&[0, 0, 1],
			&[1],
			true,
			1,
			slow,
		)
		.unwrap();
		assert_eq!(result.cokernel[&0], Finite(2));
		assert_eq!(result.codomain[&0], Infinity);
		let cycle = result.cokernel_representatives.as_ref().unwrap()[&0]
			.iter()
			.copied()
			.collect();
		assert_eq!(binary_rank([HashSet::from([0, 1]), cycle]), 2);
	}
}

/// Cokernel V can mix vertices with an edge; essential extraction must instead
/// use the codomain cycle. The calculation needs no retained cokernel V.
#[test]
fn essential_cokernel_uses_codomain_v_without_retaining_cokernel_v() {
	let matrix = [vec![], vec![], vec![0, 1], vec![0, 1]];
	let ensemble =
		all_decompositions::<SerialAlgorithm<_>, _>(&matrix, &[0, 0, 1, 1], &[0, 1, 3], 1).unwrap();
	assert!(ensemble.d_cok.has_v().is_err());
	let result = ensemble.all_diagrams(true);
	assert_eq!(result.cokernel[&3], Infinity);
	assert_eq!(result.cokernel_representatives.unwrap()[&3], vec![2, 3]);
	let relations = [(0, vec![0]), (0, vec![1]), (1, vec![0, 1]), (1, vec![0, 1])];
	let hypothetical = SerialAlgorithm::<VecColumn>::init(Some(LoPhatOptions {
		maintain_v: true,
		clearing: false,
		..Default::default()
	}))
	.add_cols(relations.into_iter().map(VecColumn::from))
	.decompose();
	assert_eq!(
		hypothetical
			.get_v_col(3)
			.unwrap()
			.entries()
			.collect::<Vec<_>>(),
		vec![0, 1, 3]
	);
}

/// Raw cylinder inclusions retain finite equal-time cokernel features,
/// allowing quotient cycle bases to be checked at intermediate indices.
#[test]
fn cokernel_representatives_in_mapping_cylinders_in_both_modes() {
	let point = [(0.0, 0, vec![])];
	for mapping in [vec![vec![0]], vec![vec![]]] {
		let (matrix, metadata) = build_cylinder(&point, &point, &mapping);
		for slow in [false, true] {
			let result = sixpack_from_inclusion(
				&matrix,
				&metadata.degrees,
				&metadata.domain_indices,
				true,
				1,
				slow,
			)
			.unwrap();
			let birth = metadata.codomain_indices[0];
			let support = if mapping[0].is_empty() {
				vec![birth]
			} else {
				vec![metadata.domain_indices[0], birth]
			};
			assert_eq!(
				result.cokernel_representatives.as_ref().unwrap(),
				&HashMap::from([(birth, support)])
			);
			assert_cokernel_cycles(&result, &matrix, &metadata.degrees);
		}
	}
}
