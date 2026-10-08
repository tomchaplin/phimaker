//! Public standalone API checks using minimal triangular cycle bases over F2.

use std::collections::HashMap;

use phimaker::{
	diagrams::{
		ExtendedUsize::{Finite, Infinity},
		PersistenceDiagram,
	},
	errors::PersistenceMapError,
	get_persistence_map,
};

/// A generator can map to a sum of two target features instead of an inclusion.
#[test]
fn generator_images_produce_binary_coefficients() {
	let source: PersistenceDiagram = [(1, Infinity)].into_iter().collect();
	let target = [(0, Infinity), (1, Infinity)].into_iter().collect();
	let result = get_persistence_map(
		&source,
		&HashMap::from([(1, vec![1])]),
		&target,
		&HashMap::from([(0, vec![0]), (1, vec![1])]),
		&HashMap::from([(1, vec![0, 1])]),
	)
	.unwrap();
	assert_eq!(result, HashMap::from([(1, vec![0, 1])]));
}

/// Equal images of distinct source generators cancel, yielding the zero cycle.
#[test]
fn overlapping_generator_images_cancel() {
	let source = [(1, Infinity)].into_iter().collect();
	let target = [(0, Infinity)].into_iter().collect();
	let result = get_persistence_map(
		&source,
		&HashMap::from([(1, vec![0, 1])]),
		&target,
		&HashMap::from([(0, vec![0])]),
		&HashMap::from([(0, vec![0]), (1, vec![0])]),
	)
	.unwrap();
	assert_eq!(result, HashMap::from([(1, vec![])]));
}

/// Missing images mean zero, so an empty target still admits a zero map.
#[test]
fn missing_images_are_zero_and_unused_images_are_ignored() {
	let source = [(1, Infinity)].into_iter().collect();
	let result = get_persistence_map(
		&source,
		&HashMap::from([(1, vec![1])]),
		&PersistenceDiagram::default(),
		&HashMap::new(),
		&HashMap::from([(99, vec![100])]),
	)
	.unwrap();
	assert_eq!(result, HashMap::from([(1, vec![])]));
}

/// Repeated listings within columns are ignored, but overlaps between columns
/// cancel.
#[test]
fn supports_are_normalized_without_mutating_inputs() {
	let source = [(1, Infinity)].into_iter().collect();
	let target = [(0, Infinity), (1, Infinity)].into_iter().collect();
	let source_reps = HashMap::from([(1, vec![1, 0, 1])]);
	let target_reps = HashMap::from([(0, vec![0, 0]), (1, vec![1, 1])]);
	let mapping = HashMap::from([(0, vec![0, 0]), (1, vec![1, 0, 1])]);
	let originals = (source_reps.clone(), target_reps.clone(), mapping.clone());
	assert_eq!(
		get_persistence_map(&source, &source_reps, &target, &target_reps, &mapping).unwrap(),
		HashMap::from([(1, vec![1])]),
	);
	assert_eq!((source_reps, target_reps, mapping), originals);
}

/// A future birth cannot receive a coefficient even when both intervals are
/// essential.
#[test]
fn future_target_birth_is_rejected() {
	let source = [(1, Infinity)].into_iter().collect();
	let target = [(5, Infinity)].into_iter().collect();
	assert_eq!(
		get_persistence_map(
			&source,
			&HashMap::from([(1, vec![1])]),
			&target,
			&HashMap::from([(5, vec![5])]),
			&HashMap::from([(1, vec![5])]),
		),
		Err(PersistenceMapError::IncompatibleIntervals {
			target: "target",
			birth: 1,
			target_birth: 5
		}),
	);
}

/// A nonzero target coefficient must die no later than its finite source.
#[test]
fn target_outliving_source_is_rejected() {
	let source = [(1, Finite(3))].into_iter().collect();
	let target = [(0, Finite(4))].into_iter().collect();
	assert!(matches!(
		get_persistence_map(
			&source,
			&HashMap::from([(1, vec![1])]),
			&target,
			&HashMap::from([(0, vec![0])]),
			&HashMap::from([(1, vec![0])]),
		),
		Err(PersistenceMapError::IncompatibleIntervals { .. })
	));
}

/// A mapped cycle outside the supplied target span reports an error.
#[test]
fn mapped_support_must_be_in_target_span() {
	let source = [(1, Infinity)].into_iter().collect();
	let target = [(0, Infinity)].into_iter().collect();
	assert!(matches!(
		get_persistence_map(
			&source,
			&HashMap::from([(1, vec![1])]),
			&target,
			&HashMap::from([(0, vec![0])]),
			&HashMap::from([(1, vec![1])]),
		),
		Err(PersistenceMapError::Unsolvable { birth: 1, .. })
	));
}

/// Mapping to an already dead target cycle gives zero after exact expansion.
#[test]
fn already_dead_target_is_removed() {
	let source = [(3, Infinity)].into_iter().collect();
	let target = [(0, Finite(2))].into_iter().collect();
	assert_eq!(
		get_persistence_map(
			&source,
			&HashMap::from([(3, vec![3])]),
			&target,
			&HashMap::from([(0, vec![0])]),
			&HashMap::from([(3, vec![0])]),
		)
		.unwrap(),
		HashMap::from([(3, vec![])])
	);
}
