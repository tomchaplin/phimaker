//! Validate transported bases against the original inputs, without cylinder
//! chains.

use std::collections::{HashMap, HashSet};

use crate::{
	cylinder::CylinderMetadata,
	diagrams::{DiagramEnsemble, ExtendedUsize::Finite, PersistenceDiagram},
	sixpack,
	tests::binary_rank,
};

type Matrix = Vec<(f64, usize, Vec<usize>)>;

fn boundary(matrix: &[(f64, usize, Vec<usize>)], support: &[usize]) -> HashSet<usize> {
	let mut out = HashSet::new();
	for &idx in support {
		for &row in &matrix[idx].2 {
			if !out.insert(row) {
				out.remove(&row);
			}
		}
	}
	out
}

fn map_chain(mapping: &[Vec<usize>], support: &HashSet<usize>) -> HashSet<usize> {
	let mut out = HashSet::new();
	for &idx in support {
		for &row in &mapping[idx] {
			if !out.insert(row) {
				out.remove(&row);
			}
		}
	}
	out
}

fn xor(left: &mut HashSet<usize>, right: &HashSet<usize>) {
	for &idx in right {
		if !left.insert(idx) {
			left.remove(&idx);
		}
	}
}

// Gaussian elimination with chain coefficients supplies an independent basis
// of the original cycle space at a time, including cycles that already bound.
fn cycles(matrix: &Matrix, time: f64) -> Vec<HashSet<usize>> {
	let mut pivots: HashMap<usize, (HashSet<usize>, HashSet<usize>)> = HashMap::new();
	let mut out = Vec::new();
	for (idx, (entrance, _, column)) in matrix.iter().enumerate() {
		if *entrance > time {
			continue;
		}
		let mut residual: HashSet<_> = column.iter().copied().collect();
		let mut chain = HashSet::from([idx]);
		while let Some(&pivot) = residual.iter().max() {
			if let Some((earlier_boundary, earlier_chain)) = pivots.get(&pivot) {
				xor(&mut residual, earlier_boundary);
				xor(&mut chain, earlier_chain);
			} else {
				pivots.insert(pivot, (residual.clone(), chain.clone()));
				break;
			}
		}
		if residual.is_empty() {
			out.push(chain);
		}
	}
	out
}

fn boundaries(matrix: &Matrix, time: f64) -> Vec<HashSet<usize>> {
	matrix
		.iter()
		.filter(|(entrance, ..)| *entrance <= time)
		.map(|(_, _, column)| column.iter().copied().collect())
		.collect()
}

fn assert_inputs(
	result: &DiagramEnsemble,
	metadata: &CylinderMetadata,
	domain: &Matrix,
	codomain: &Matrix,
	mapping: &[Vec<usize>],
) {
	for (diagram, reps, matrix, shift) in [
		(&result.domain, &result.domain_representatives, domain, 0),
		(&result.kernel, &result.kernel_representatives, domain, 1),
		(&result.image, &result.image_representatives, codomain, 0),
		(
			&result.codomain,
			&result.codomain_representatives,
			codomain,
			0,
		),
		(
			&result.cokernel,
			&result.cokernel_representatives,
			codomain,
			0,
		),
	] {
		let reps = reps.as_ref().unwrap();
		assert_eq!(diagram.len(), reps.len());
		for (&birth, support) in reps {
			assert!(diagram.contains_key(&birth));
			assert!(!support.is_empty());
			assert!(support.windows(2).all(|pair| pair[0] < pair[1]));
			for &idx in support {
				assert!(idx < matrix.len());
				assert_eq!(matrix[idx].1 + shift, metadata.degrees[birth]);
				assert!(matrix[idx].0 <= metadata.times[birth]);
			}
			assert!(boundary(matrix, support).is_empty());
		}
	}
	for (&birth, (a, b)) in result.relative_representatives.as_ref().unwrap() {
		assert!(!a.is_empty() || !b.is_empty());
		assert!(a.windows(2).all(|pair| pair[0] < pair[1]));
		assert!(b.windows(2).all(|pair| pair[0] < pair[1]));
		for &idx in a {
			assert!(idx < domain.len());
			assert_eq!(domain[idx].1 + 1, metadata.degrees[birth]);
			assert!(domain[idx].0 <= metadata.times[birth]);
		}
		for &idx in b {
			assert!(idx < codomain.len());
			assert_eq!(codomain[idx].1, metadata.degrees[birth]);
			assert!(codomain[idx].0 <= metadata.times[birth]);
		}
		assert!(boundary(domain, a).is_empty());
		assert_eq!(
			boundary(codomain, b),
			map_chain(mapping, &a.iter().copied().collect())
		);
	}
	let mut cone = codomain.clone();
	for (idx, (time, degree, boundary)) in domain.iter().enumerate() {
		let mut column = mapping[idx].clone();
		column.extend(boundary.iter().map(|row| codomain.len() + row));
		cone.push((*time, degree + 1, column));
	}
	let mut times: Vec<_> = domain
		.iter()
		.chain(codomain)
		.map(|(time, ..)| *time)
		.collect();
	times.sort_by(f64::total_cmp);
	times.dedup();
	for time in times {
		let alive = |diagram: &PersistenceDiagram, birth: usize| {
			metadata.times[birth] <= time
				&& match diagram[&birth] {
					Finite(death) => time < metadata.times[death],
					_ => true,
				}
		};
		let a_cycles = cycles(domain, time);
		let a_boundaries = boundaries(domain, time);
		let b_boundaries = boundaries(codomain, time);
		let image_cycles: Vec<_> = a_cycles
			.iter()
			.map(|cycle| map_chain(mapping, cycle))
			.collect();
		let dim_a = a_cycles.len() - binary_rank(a_boundaries.clone());
		let dim_b = cycles(codomain, time).len() - binary_rank(b_boundaries.clone());
		let quotient_relations: Vec<_> =
			b_boundaries.iter().chain(&image_cycles).cloned().collect();
		let dim_image = binary_rank(quotient_relations.clone()) - binary_rank(b_boundaries.clone());
		for (diagram, reps, relations, expected) in [
			(
				&result.domain,
				&result.domain_representatives,
				&a_boundaries,
				dim_a,
			),
			(
				&result.kernel,
				&result.kernel_representatives,
				&a_boundaries,
				dim_a - dim_image,
			),
			(
				&result.image,
				&result.image_representatives,
				&b_boundaries,
				dim_image,
			),
			(
				&result.codomain,
				&result.codomain_representatives,
				&b_boundaries,
				dim_b,
			),
			(
				&result.cokernel,
				&result.cokernel_representatives,
				&quotient_relations,
				dim_b - dim_image,
			),
		] {
			let active: Vec<HashSet<_>> = reps
				.as_ref()
				.unwrap()
				.iter()
				.filter(|&(&birth, _)| alive(diagram, birth))
				.map(|(_, support)| support.iter().copied().collect())
				.collect();
			assert_eq!(active.len(), expected);
			assert_eq!(
				binary_rank(relations.iter().chain(&active).cloned()),
				binary_rank(relations.clone()) + expected
			);
		}
		for (map, source, target, source_reps, target_reps, relations, apply_f, expected_rank) in [
			(
				&result.kernel_to_domain,
				&result.kernel,
				&result.domain,
				&result.kernel_representatives,
				&result.domain_representatives,
				&a_boundaries,
				false,
				dim_a - dim_image,
			),
			(
				&result.domain_to_image,
				&result.domain,
				&result.image,
				&result.domain_representatives,
				&result.image_representatives,
				&b_boundaries,
				true,
				dim_image,
			),
			(
				&result.image_to_codomain,
				&result.image,
				&result.codomain,
				&result.image_representatives,
				&result.codomain_representatives,
				&b_boundaries,
				false,
				dim_image,
			),
			(
				&result.domain_to_codomain,
				&result.domain,
				&result.codomain,
				&result.domain_representatives,
				&result.codomain_representatives,
				&b_boundaries,
				true,
				dim_image,
			),
			(
				&result.codomain_to_cokernel,
				&result.codomain,
				&result.cokernel,
				&result.codomain_representatives,
				&result.cokernel_representatives,
				&quotient_relations,
				false,
				dim_b - dim_image,
			),
		] {
			let mut columns = Vec::new();
			for (&birth, support) in source_reps.as_ref().unwrap() {
				if !alive(source, birth) {
					continue;
				}
				let coefficients: HashSet<_> = map.as_ref().unwrap()[&birth]
					.iter()
					.copied()
					.filter(|&idx| alive(target, idx))
					.collect();
				let mut residual: HashSet<_> = support.iter().copied().collect();
				if apply_f {
					residual = map_chain(mapping, &residual);
				}
				for target_birth in &coefficients {
					xor(
						&mut residual,
						&target_reps.as_ref().unwrap()[target_birth]
							.iter()
							.copied()
							.collect(),
					);
				}
				assert_eq!(
					binary_rank(relations.iter().cloned().chain([residual])),
					binary_rank(relations.clone())
				);
				columns.push(coefficients);
			}
			assert_eq!(binary_rank(columns), expected_rank);
		}
		let cone_relations = boundaries(&cone, time);
		let relative: Vec<HashSet<_>> = result
			.relative_representatives
			.as_ref()
			.unwrap()
			.iter()
			.filter(|&(&birth, _)| alive(&result.relative, birth))
			.map(|(_, (a, b))| {
				b.iter()
					.copied()
					.chain(a.iter().map(|idx| codomain.len() + idx))
					.collect()
			})
			.collect();
		let cone_rank = binary_rank(cone_relations.clone());
		assert_eq!(relative.len(), cycles(&cone, time).len() - cone_rank);
		assert_eq!(
			binary_rank(cone_relations.into_iter().chain(relative.clone())),
			cone_rank + relative.len()
		);
	}
}

/// Basis ranks, map actions, and cone equations are checked in original input
/// coordinates at all supplied times, including sums, finite deaths, and loops.
#[test]
fn converted_bases_and_maps_match_original_inputs() {
	let point = vec![(0.0, 0, vec![])];
	let vertices = vec![(0.0, 0, vec![]), (0.0, 0, vec![])];
	let interval = vec![(0.0, 0, vec![]), (0.0, 0, vec![]), (1.0, 1, vec![0, 1])];
	let mut cases: Vec<(Matrix, Matrix, Vec<Vec<usize>>)> = vec![
		(vec![], vec![], vec![]),
		(vec![], point.clone(), vec![]),
		(point.clone(), vec![], vec![vec![]]),
		(vertices.clone(), point.clone(), vec![vec![0], vec![0]]),
		(
			vertices.clone(),
			vec![(0.0, 0, vec![]); 3],
			vec![vec![0, 1], vec![1, 2]],
		),
		(vertices.clone(), interval.clone(), vec![vec![0], vec![1]]),
		(
			vec![(0.0, 0, vec![]), (0.0, 0, vec![]), (2.0, 1, vec![0, 1])],
			interval.clone(),
			vec![vec![0], vec![1], vec![2]],
		),
		(
			point,
			vec![
				(0.0, 0, vec![]),
				(0.0, 0, vec![]),
				(1.0, 1, vec![0, 1]),
				(2.0, 1, vec![0, 1]),
			],
			vec![vec![1]],
		),
	];
	let triangle = vec![
		(0.0, 0, vec![]),
		(0.0, 0, vec![]),
		(0.0, 0, vec![]),
		(1.0, 1, vec![0, 1]),
		(1.0, 1, vec![0, 2]),
		(1.0, 1, vec![1, 2]),
	];
	let mut filled = triangle.clone();
	filled.push((2.0, 2, vec![3, 4, 5]));
	cases.push((triangle, filled, (0..6).map(|idx| vec![idx]).collect()));
	for (domain, codomain, mapping) in cases {
		for slow in [false, true] {
			let (result, metadata) = sixpack(&domain, &codomain, &mapping, true, 1, slow).unwrap();
			assert_inputs(&result, &metadata, &domain, &codomain, &mapping);
		}
	}
}

/// The first relative component survives the zero map into an empty codomain.
/// A nonzero codomain component can instead be a chain with domain boundary.
#[test]
fn relative_pairs_have_original_domain_and_codomain_meanings() {
	for slow in [false, true] {
		let (result, metadata) =
			sixpack(&[(0.0, 0, vec![])], &[], &[vec![]], true, 1, slow).unwrap();
		assert_eq!(
			result.relative_representatives.unwrap(),
			HashMap::from([(metadata.domain_shift[0], (vec![0], vec![]))])
		);
		let (result, _) = sixpack(
			&[(0.0, 0, vec![]), (0.0, 0, vec![])],
			&[(0.0, 0, vec![]), (0.0, 0, vec![]), (1.0, 1, vec![0, 1])],
			&[vec![0], vec![1]],
			true,
			1,
			slow,
		)
		.unwrap();
		assert_eq!(
			result
				.relative_representatives
				.unwrap()
				.into_values()
				.collect::<Vec<_>>(),
			vec![(vec![0, 1], vec![2])]
		);
	}
}
