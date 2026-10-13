//! Stored maps are verified against chains and pointwise exactness over F2.

use std::collections::HashSet;

use crate::{
	cylinder::build_cylinder,
	diagrams::{DiagramEnsemble, ExtendedUsize::Finite, PersistenceDiagram, PersistenceMap},
	sixpack,
	sixpack_from_inclusion,
	tests::binary_rank,
};

fn alive(diagram: &PersistenceDiagram, birth: usize, index: usize) -> bool {
	birth <= index && Finite(index) < diagram[&birth]
}

fn columns(
	map: &PersistenceMap,
	source: &PersistenceDiagram,
	target: &PersistenceDiagram,
	index: usize,
) -> Vec<HashSet<usize>> {
	source
		.keys()
		.filter(|&&birth| alive(source, birth, index))
		.map(|birth| {
			map[birth]
				.iter()
				.copied()
				.filter(|&birth| alive(target, birth, index))
				.collect()
		})
		.collect()
}

fn composite(
	first: &PersistenceMap,
	middle: &PersistenceDiagram,
	second: &PersistenceMap,
	target: &PersistenceDiagram,
	birth: usize,
	index: usize,
) -> HashSet<usize> {
	let mut out = HashSet::new();
	for &mid in &first[&birth] {
		if !alive(middle, mid, index) {
			continue;
		}
		for &end in &second[&mid] {
			if alive(target, end, index) && !out.insert(end) {
				out.remove(&end);
			}
		}
	}
	out
}

fn assert_stored_maps(
	result: &DiagramEnsemble,
	matrix: &[Vec<usize>],
	degrees: &[usize],
	domain: &[usize],
) {
	for (
		map,
		source,
		target,
		source_reps,
		target_reps,
		domain_relations,
		quotient,
		kernel_source,
	) in [
		(
			&result.kernel_to_domain,
			&result.kernel,
			&result.domain,
			&result.kernel_representatives,
			&result.domain_representatives,
			true,
			false,
			true,
		),
		(
			&result.domain_to_image,
			&result.domain,
			&result.image,
			&result.domain_representatives,
			&result.image_representatives,
			false,
			false,
			false,
		),
		(
			&result.image_to_codomain,
			&result.image,
			&result.codomain,
			&result.image_representatives,
			&result.codomain_representatives,
			false,
			false,
			false,
		),
		(
			&result.domain_to_codomain,
			&result.domain,
			&result.codomain,
			&result.domain_representatives,
			&result.codomain_representatives,
			false,
			false,
			false,
		),
		(
			&result.codomain_to_cokernel,
			&result.codomain,
			&result.cokernel,
			&result.codomain_representatives,
			&result.cokernel_representatives,
			false,
			true,
			false,
		),
	] {
		let map = map.as_ref().unwrap();
		assert_eq!(map.len(), source.len());
		for (&birth, coefficients) in map {
			assert!(source.contains_key(&birth));
			assert!(coefficients.windows(2).all(|pair| pair[0] < pair[1]));
			let mut residual: HashSet<_> = source_reps.as_ref().unwrap()[&birth]
				.iter()
				.copied()
				.collect();
			for &target_birth in coefficients {
				assert!(target_birth <= birth && Finite(birth) < target[&target_birth]);
				assert!(target[&target_birth] <= source[&birth]);
				assert_eq!(
					degrees[target_birth],
					degrees[birth] - usize::from(kernel_source)
				);
				for &idx in &target_reps.as_ref().unwrap()[&target_birth] {
					if !residual.insert(idx) {
						residual.remove(&idx);
					}
				}
			}
			let mut relations: Vec<HashSet<_>> = matrix
				.iter()
				.enumerate()
				.filter(|&(idx, _)| idx <= birth && (!domain_relations || domain.contains(&idx)))
				.map(|(_, column)| column.iter().copied().collect())
				.collect();
			if quotient {
				relations.extend(
					result
						.domain_representatives
						.as_ref()
						.unwrap()
						.iter()
						.filter(|&(&idx, _)| idx <= birth)
						.map(|(_, support)| support.iter().copied().collect()),
				);
			}
			let rank = binary_rank(relations.clone());
			relations.push(residual);
			assert_eq!(
				binary_rank(relations),
				rank,
				"Chain map at source birth {birth}"
			);
		}
	}
	let kd = result.kernel_to_domain.as_ref().unwrap();
	let di = result.domain_to_image.as_ref().unwrap();
	let ic = result.image_to_codomain.as_ref().unwrap();
	let dc = result.domain_to_codomain.as_ref().unwrap();
	let cq = result.codomain_to_cokernel.as_ref().unwrap();
	for index in 0..=matrix.len() {
		let dim = |diagram: &PersistenceDiagram| {
			diagram
				.keys()
				.filter(|&&b| alive(diagram, b, index))
				.count()
		};
		assert_eq!(
			binary_rank(columns(kd, &result.kernel, &result.domain, index)),
			dim(&result.kernel)
		);
		assert_eq!(
			binary_rank(columns(di, &result.domain, &result.image, index)),
			dim(&result.image)
		);
		assert_eq!(
			binary_rank(columns(ic, &result.image, &result.codomain, index)),
			dim(&result.image)
		);
		assert_eq!(
			binary_rank(columns(dc, &result.domain, &result.codomain, index)),
			dim(&result.image)
		);
		assert_eq!(
			binary_rank(columns(cq, &result.codomain, &result.cokernel, index)),
			dim(&result.cokernel)
		);
		assert_eq!(
			dim(&result.domain),
			dim(&result.kernel) + dim(&result.image)
		);
		assert_eq!(
			dim(&result.codomain),
			dim(&result.image) + dim(&result.cokernel)
		);
		for &birth in result
			.domain
			.keys()
			.filter(|&&b| alive(&result.domain, b, index))
		{
			let direct: HashSet<_> = dc[&birth]
				.iter()
				.copied()
				.filter(|&b| alive(&result.codomain, b, index))
				.collect();
			assert_eq!(
				composite(di, &result.image, ic, &result.codomain, birth, index),
				direct
			);
			assert!(composite(dc, &result.codomain, cq, &result.cokernel, birth, index).is_empty());
		}
		for &birth in result
			.kernel
			.keys()
			.filter(|&&b| alive(&result.kernel, b, index))
		{
			assert!(composite(kd, &result.domain, di, &result.image, birth, index).is_empty());
			assert!(composite(kd, &result.domain, dc, &result.codomain, birth, index).is_empty());
		}
		for &birth in result
			.image
			.keys()
			.filter(|&&b| alive(&result.image, b, index))
		{
			assert!(composite(ic, &result.codomain, cq, &result.cokernel, birth, index).is_empty());
		}
	}
}

/// Every stored map has the right chain action and is pointwise exact,
/// including empty/identity cases, sums, dead targets, finite/essential
/// kernels, and quotient lifts.
#[test]
fn stored_maps_are_correct_in_both_modes() {
	let cases = [
		(vec![], vec![], vec![]),
		(vec![vec![]], vec![0], vec![]),
		(
			vec![vec![], vec![], vec![0, 1]],
			vec![0, 0, 1],
			vec![2, 0, 1],
		),
		(vec![vec![], vec![], vec![0, 1]], vec![0, 0, 1], vec![1]),
		(
			vec![vec![], vec![], vec![], vec![0, 1], vec![0, 2], vec![1, 2]],
			vec![0, 0, 0, 1, 1, 1],
			vec![5, 2, 1],
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
		),
		(
			vec![vec![], vec![], vec![0, 1], vec![], vec![0, 3], vec![1, 3]],
			vec![0, 0, 1, 0, 1, 1],
			vec![5, 3, 1],
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
		),
		(
			vec![vec![], vec![], vec![0, 1], vec![0, 1]],
			vec![0, 0, 1, 1],
			vec![0, 1, 3],
		),
		(
			vec![vec![], vec![], vec![0, 1], vec![0, 1], vec![2, 3]],
			vec![0, 0, 1, 1, 2],
			vec![0, 1, 3],
		),
	];
	for (matrix, degrees, domain) in cases {
		for slow in [false, true] {
			let skipped =
				sixpack_from_inclusion(&matrix, &degrees, &domain, false, 1, slow).unwrap();
			for map in [
				&skipped.kernel_to_domain,
				&skipped.domain_to_image,
				&skipped.image_to_codomain,
				&skipped.domain_to_codomain,
				&skipped.codomain_to_cokernel,
			] {
				assert!(map.is_none());
			}
			let result = sixpack_from_inclusion(&matrix, &degrees, &domain, true, 1, slow).unwrap();
			assert_stored_maps(&result, &matrix, &degrees, &domain);
		}
	}
}

/// A quotient requires both image directions and cokernel lifts for expansion;
/// an identity inclusion instead has one explicit zero column per codomain
/// feature.
#[test]
fn cokernel_projection_has_expected_coefficients() {
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
		assert_eq!(
			result.codomain_to_cokernel.unwrap(),
			[(0, vec![0]), (1, vec![0])].into()
		);
		let identity = sixpack_from_inclusion(
			&[vec![], vec![], vec![0, 1]],
			&[0, 0, 1],
			&[0, 1, 2],
			true,
			1,
			slow,
		)
		.unwrap();
		assert_eq!(
			identity.codomain_to_cokernel.unwrap(),
			[(0, vec![]), (1, vec![])].into()
		);
	}
}

/// This actual inclusion has a spurious coefficient if complete tables are
/// multiplied without discarding a codomain interval dead at the source birth.
#[test]
fn stored_factorization_requires_interval_lifetimes() {
	let matrix = [vec![], vec![], vec![0], vec![], vec![0, 3], vec![1, 3]];
	for slow in [false, true] {
		let result =
			sixpack_from_inclusion(&matrix, &[0, 0, 1, 0, 1, 1], &[1, 3, 5], true, 1, slow)
				.unwrap();
		let di = result.domain_to_image.as_ref().unwrap();
		let ic = result.image_to_codomain.as_ref().unwrap();
		assert_eq!(di[&3], vec![1, 3]);
		assert_eq!(ic[&1], vec![0, 1]);
		assert_eq!(ic[&3], vec![3]);
		assert_eq!(result.domain_to_codomain.as_ref().unwrap()[&3], vec![1, 3]);
		assert_eq!(result.codomain[&0], Finite(2));
		let mut naive = HashSet::new();
		for &mid in &di[&3] {
			for &end in &ic[&mid] {
				if !naive.insert(end) {
					naive.remove(&end);
				}
			}
		}
		assert_eq!(naive, HashSet::from([0, 1, 3]));
		assert_stored_maps(&result, &matrix, &[0, 0, 1, 0, 1, 1], &[1, 3, 5]);
	}
}

/// Evaluate both nontrivial and zero maps at every cylinder index, retaining
/// zero-duration intervals and the original coordinate convention.
#[test]
fn stored_maps_in_mapping_cylinders_in_both_modes() {
	let point = [(0.0, 0, vec![])];
	for mapping in [vec![vec![0]], vec![vec![]]] {
		let (matrix, metadata) = build_cylinder(&point, &point, &mapping);
		for slow in [false, true] {
			let (result, _) = sixpack(&point, &point, &mapping, true, 1, slow).unwrap();
			assert_stored_maps(
				&result,
				&matrix,
				&metadata.degrees,
				&metadata.domain_indices,
			);
			let index = metadata.times.len() - 1;
			let columns = columns(
				result.domain_to_codomain.as_ref().unwrap(),
				&result.domain,
				&result.codomain,
				index,
			);
			assert_eq!(binary_rank(columns), usize::from(!mapping[0].is_empty()));
		}
	}
}
