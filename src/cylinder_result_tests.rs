//! General-map results are evaluated after complete filtration-time blocks.

use std::collections::HashMap;

use crate::{
	diagrams::{
		DiagramEnsemble,
		ExtendedUsize::{Finite, Infinity},
	},
	sixpack,
	sixpack_from_inclusion,
};

fn assert_result_keys(result: &DiagramEnsemble, times: &[f64], requested: bool) {
	for (diagram, representatives) in [
		(&result.domain, &result.domain_representatives),
		(&result.codomain, &result.codomain_representatives),
		(&result.image, &result.image_representatives),
		(&result.kernel, &result.kernel_representatives),
		(&result.cokernel, &result.cokernel_representatives),
		(&result.relative, &result.relative_representatives),
	] {
		for (&birth, &death) in diagram.iter() {
			if let Finite(death) = death {
				assert!(times[birth] < times[death]);
			}
		}
		assert_eq!(representatives.is_some(), requested);
		if let Some(representatives) = representatives {
			assert_eq!(representatives.len(), diagram.len());
			assert!(
				representatives
					.keys()
					.all(|birth| diagram.contains_key(birth))
			);
		}
	}
	for (mapping, source, target) in [
		(&result.kernel_to_domain, &result.kernel, &result.domain),
		(&result.domain_to_image, &result.domain, &result.image),
		(&result.image_to_codomain, &result.image, &result.codomain),
		(&result.domain_to_codomain, &result.domain, &result.codomain),
		(
			&result.codomain_to_cokernel,
			&result.codomain,
			&result.cokernel,
		),
	] {
		assert_eq!(mapping.is_some(), requested);
		if let Some(mapping) = mapping {
			assert_eq!(mapping.len(), source.len());
			for (birth, coefficients) in mapping {
				assert!(source.contains_key(birth));
				assert!(
					coefficients
						.iter()
						.all(|target_birth| target.contains_key(target_birth))
				);
			}
		}
	}
}

/// Equal-time artifacts disappear from all diagrams and stored maps. The zero
/// map still needs their cycles during solving, before the cleanup runs.
#[test]
fn equal_time_identity_and_zero_maps() {
	let point = [(0.0, 0, vec![])];
	for slow in [false, true] {
		for requested in [false, true] {
			for nonzero in [false, true] {
				let mapping = [if nonzero { vec![0] } else { vec![] }];
				let (result, metadata) =
					sixpack(&point, &point, &mapping, requested, 1, slow).unwrap();
				assert_eq!(metadata.times, vec![0.0; 3]);
				assert_eq!(result.domain, [(0, Infinity)].into_iter().collect());
				assert_eq!(
					result.codomain,
					[(usize::from(!nonzero), Infinity)].into_iter().collect()
				);
				assert_eq!(result.image.len(), usize::from(nonzero));
				assert_eq!(result.kernel.len(), usize::from(!nonzero));
				assert_eq!(result.cokernel.len(), usize::from(!nonzero));
				assert_eq!(result.relative.len(), 2 * usize::from(!nonzero));
				assert_result_keys(&result, &metadata.times, requested);
				if requested {
					let expected = if nonzero { vec![0] } else { vec![] };
					assert_eq!(
						result.domain_to_codomain.unwrap(),
						HashMap::from([(0, expected.clone())])
					);
					assert_eq!(
						result.domain_to_image.unwrap(),
						HashMap::from([(0, expected)])
					);
				}
			}
		}
	}
}

/// Filtering uses exact time equality, preserving even a subnormal positive
/// lifetime while removing the cylinder's unrelated equal-time artifacts.
#[test]
fn positive_duration_intervals_are_not_removed() {
	for death in [1.0, f64::from_bits(1)] {
		let codomain = [(0.0, 0, vec![]), (0.0, 0, vec![]), (death, 1, vec![0, 1])];
		for slow in [false, true] {
			for requested in [false, true] {
				let (result, metadata) = sixpack(
					&[(0.0, 0, vec![])],
					&codomain,
					&[vec![1]],
					requested,
					1,
					slow,
				)
				.unwrap();
				assert_result_keys(&result, &metadata.times, requested);
				assert_eq!(result.codomain.len(), 2);
				for diagram in [&result.codomain, &result.cokernel, &result.relative] {
					let finite: Vec<_> = diagram
						.iter()
						.filter_map(|(&birth, &end)| {
							match end {
								Finite(end) => Some((metadata.times[birth], metadata.times[end])),
								Infinity => None,
							}
						})
						.collect();
					assert_eq!(finite, vec![(0.0, death)]);
				}
			}
		}
	}
}

/// Inclusion callers supply no times, so index-positive intervals remain
/// available for complete cycle-basis calculations and raw cylinder analysis.
#[test]
fn index_only_inclusions_keep_complete_intervals() {
	for slow in [false, true] {
		let result =
			sixpack_from_inclusion(&[vec![], vec![], vec![0]], &[0, 0, 1], &[0], true, 1, slow)
				.unwrap();
		assert_eq!(result.image, [(0, Finite(2))].into_iter().collect());
		assert_eq!(
			result.domain_to_image.unwrap(),
			HashMap::from([(0, vec![0])])
		);
	}
}
