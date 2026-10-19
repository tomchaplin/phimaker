//! Relative representatives form homology bases in B/A, with ambient indices.

use std::{
	collections::{HashMap, HashSet},
	io::Seek,
};

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
	ensemble::{FileEnsemble, to_file},
	sixpack_from_inclusion,
	tests::binary_rank,
};

// Expected quotient lifts remain independent of how their boundary is applied.
fn relative_lifts(result: &DiagramEnsemble) -> HashMap<usize, Vec<usize>> {
	result
		.relative_representatives
		.as_ref()
		.unwrap()
		.iter()
		.map(|(&birth, (_, support))| (birth, support.clone()))
		.collect()
}

// Project input boundaries by deleting domain rows, retaining ambient labels.
// This oracle does not use the quotient builder or any computed
// representatives.
fn quotient_boundaries(
	matrix: &[Vec<usize>],
	degrees: &[usize],
	domain: &HashSet<usize>,
	index: usize,
	degree: usize,
) -> Vec<HashSet<usize>> {
	matrix
		.iter()
		.enumerate()
		.take(index + 1)
		.filter(|(idx, _)| !domain.contains(idx) && degrees[*idx] == degree)
		.map(|(_, column)| {
			column
				.iter()
				.copied()
				.filter(|row| !domain.contains(row))
				.collect()
		})
		.collect()
}

fn assert_relative_cycles(
	result: &DiagramEnsemble,
	matrix: &[Vec<usize>],
	degrees: &[usize],
	domain: &[usize],
) {
	let domain: HashSet<_> = domain.iter().copied().collect();
	let representatives = result.relative_representatives.as_ref().unwrap();
	assert_eq!(representatives.len(), result.relative.len());
	for (&birth, (domain_cycle, support)) in representatives {
		assert!(result.relative.contains_key(&birth));
		assert_eq!(support.iter().max(), Some(&birth));
		assert_eq!(
			support.iter().copied().collect::<HashSet<_>>().len(),
			support.len()
		);
		assert!(support.iter().all(|idx| !domain.contains(idx)));
		assert!(support.iter().all(|&idx| degrees[idx] == degrees[birth]));
		let mut boundary = HashSet::new();
		for &idx in support {
			for &row in &matrix[idx] {
				if !boundary.insert(row) {
					boundary.remove(&row);
				}
			}
		}
		assert!(boundary.is_subset(&domain), "Relative cycle at {birth}");
		assert_eq!(
			domain_cycle.iter().copied().collect::<HashSet<_>>(),
			boundary
		);
		assert_eq!(domain_cycle.len(), boundary.len());
		assert!(
			domain_cycle
				.iter()
				.all(|&idx| idx < birth && degrees[idx] + 1 == degrees[birth])
		);
		let mut second_boundary = HashSet::new();
		for &idx in domain_cycle {
			for &row in &matrix[idx] {
				if !second_boundary.insert(row) {
					second_boundary.remove(&row);
				}
			}
		}
		assert!(second_boundary.is_empty());
		if let Finite(death) = result.relative[&birth] {
			let mut relations =
				quotient_boundaries(matrix, degrees, &domain, death, degrees[birth] + 1);
			let rank = binary_rank(relations.clone());
			relations.push(support.iter().copied().collect());
			assert_eq!(binary_rank(relations), rank, "Relative death at {death}");
		}
	}
	let max_degree = degrees.iter().copied().max().unwrap_or(0);
	for index in 0..=matrix.len() {
		for degree in 0..=max_degree {
			let n_chains = degrees
				.iter()
				.enumerate()
				.take(index + 1)
				.filter(|&(idx, &d)| !domain.contains(&idx) && d == degree)
				.count();
			let outgoing = quotient_boundaries(matrix, degrees, &domain, index, degree);
			let relations = quotient_boundaries(matrix, degrees, &domain, index, degree + 1);
			let boundary_rank = binary_rank(relations.clone());
			let cycle_dimension = n_chains - binary_rank(outgoing);
			let active: Vec<HashSet<_>> = representatives
				.iter()
				.filter(|&(&birth, _)| {
					birth <= index
						&& degrees[birth] == degree
						&& Finite(index) < result.relative[&birth]
				})
				.map(|(_, (_, support))| support.iter().copied().collect())
				.collect();
			assert_eq!(
				binary_rank(relations.into_iter().chain(active.iter().cloned())),
				boundary_rank + active.len(),
				"Independent relative classes in degree {degree} at {index}"
			);
			assert_eq!(
				boundary_rank + active.len(),
				cycle_dimension,
				"Complete relative basis in degree {degree} at {index}"
			);
		}
	}
}

/// Finite and essential classes form quotient homology bases at every index,
/// including empty quotients and unsorted, gapped domain coordinates.
#[test]
fn relative_cycles_form_quotient_bases_in_both_modes() {
	let triangle = vec![
		vec![],
		vec![],
		vec![],
		vec![0, 1],
		vec![0, 2],
		vec![1, 2],
		vec![3, 4, 5],
	];
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
			vec![vec![], vec![], vec![0, 1]],
			vec![0, 0, 1],
			vec![1, 0],
			HashMap::from([(2, vec![2])]),
		),
		(
			vec![vec![], vec![], vec![0, 1], vec![0, 1]],
			vec![0, 0, 1, 1],
			vec![],
			HashMap::from([(0, vec![0]), (1, vec![0, 1]), (3, vec![2, 3])]),
		),
		(
			triangle.clone(),
			vec![0, 0, 0, 1, 1, 1, 2],
			vec![3, 1, 0],
			HashMap::from([(2, vec![2]), (5, vec![4, 5])]),
		),
		(
			triangle,
			vec![0, 0, 0, 1, 1, 1, 2],
			vec![5, 4, 3, 2, 1, 0],
			HashMap::from([(6, vec![6])]),
		),
	];
	for (matrix, degrees, domain, expected) in cases {
		for slow in [false, true] {
			let result = sixpack_from_inclusion(&matrix, &degrees, &domain, true, 1, slow).unwrap();
			assert_eq!(relative_lifts(&result), expected);
			assert_relative_cycles(&result, &matrix, &degrees, &domain);
		}
	}
}

/// An edge relative to its endpoints is an essential H1 cycle, although its
/// boundary in the codomain is nonzero.
#[test]
fn relative_cycle_need_not_be_a_codomain_cycle() {
	for slow in [false, true] {
		let result = sixpack_from_inclusion(
			&[vec![], vec![], vec![0, 1]],
			&[0, 0, 1],
			&[0, 1],
			true,
			1,
			slow,
		)
		.unwrap();
		assert_eq!(result.relative[&2], Infinity);
		assert_eq!(
			result.relative_representatives.unwrap()[&2],
			(vec![0, 1], vec![2])
		);
	}
}

/// Raw relative R and V use forward quotient coordinates. Verify R = D V
/// directly for a gapped domain without using the quotient builder.
#[test]
fn relative_decomposition_retains_v_in_forward_quotient_coordinates() {
	let matrix = [
		vec![],
		vec![],
		vec![],
		vec![0, 1],
		vec![0, 2],
		vec![1, 2],
		vec![3, 4, 5],
	];
	let ensemble =
		all_decompositions::<SerialAlgorithm<_>, _>(&matrix, &[0, 0, 0, 1, 1, 1, 2], &[3, 1, 0], 1)
			.unwrap();
	let relative = &ensemble.d_rel;
	assert!(relative.has_v().is_ok());
	let quotient = [vec![], vec![0], vec![0], vec![1, 2]];
	for index in 0..quotient.len() {
		let mut boundary = HashSet::new();
		for generator in relative.get_v_col(index).unwrap().entries() {
			for &row in &quotient[generator] {
				if !boundary.insert(row) {
					boundary.remove(&row);
				}
			}
		}
		assert_eq!(boundary, relative.get_r_col(index).entries().collect());
	}
	assert_eq!(relative.diagram()[&0], Finite(1));
	assert_eq!(relative.diagram()[&2], Finite(3));
}

/// Diagram-only extraction needs no relative V, in memory or from files;
/// requesting representatives rejects a nonempty relative decomposition
/// without V.
#[test]
fn relative_extraction_without_v_in_both_modes() {
	let mut ensemble =
		all_decompositions::<SerialAlgorithm<_>, _>(&[vec![]], &[0], &[], 1).unwrap();
	ensemble.d_rel = SerialAlgorithm::<VecColumn>::init(Some(LoPhatOptions {
		maintain_v: false,
		..Default::default()
	}))
	.add_cols([VecColumn::from((0, vec![]))].into_iter())
	.decompose();
	let diagrams = ensemble.all_diagrams(false);
	assert_eq!(diagrams.relative[&0], Infinity);
	assert!(diagrams.relative_representatives.is_none());
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
	assert_eq!(diagrams.relative[&0], Infinity);
	assert!(diagrams.relative_representatives.is_none());
	for file in [
		&mut stored.d_cod,
		&mut stored.d_dom,
		&mut stored.d_im,
		&mut stored.d_ker,
		&mut stored.d_cok,
		&mut stored.d_rel,
	] {
		file.rewind().unwrap();
	}
	assert!(std::panic::catch_unwind(|| stored.all_diagrams(true)).is_err());
}

/// Raw cylinder inclusions retain all quotient cycles, including equal-time
/// intervals and the suspended domain cycle of a zero map.
#[test]
fn relative_representatives_in_mapping_cylinders_in_both_modes() {
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
			let mut expected = HashMap::from([(
				metadata.codomain_indices[0],
				vec![metadata.codomain_indices[0]],
			)]);
			if mapping[0].is_empty() {
				expected.insert(metadata.domain_shift[0], vec![metadata.domain_shift[0]]);
			}
			assert_eq!(relative_lifts(&result), expected);
			assert_relative_cycles(
				&result,
				&matrix,
				&metadata.degrees,
				&metadata.domain_indices,
			);
		}
	}
}
