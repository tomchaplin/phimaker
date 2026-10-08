"""Persistence morphisms are sparse binary columns in the chosen interval bases."""

import copy

import pytest
from phimaker import DiagramEnsemble, get_persistence_map, sixpack, sixpack_from_inclusion


def ensemble_maps(result: DiagramEnsemble) -> tuple[dict[int, list[int]], dict[int, list[int]]]:
	"""Exercise both inclusion targets through the standalone function."""
	identity = {
		index: [index] for support in result.domain_representatives.values() for index in support
	}
	return (
		get_persistence_map(
			result.domain,
			result.domain_representatives,
			result.codomain,
			result.codomain_representatives,
			identity,
		),
		get_persistence_map(
			result.domain,
			result.domain_representatives,
			result.image,
			result.image_representatives,
			identity,
		),
	)


@pytest.mark.parametrize("slow", [False, True])
def test_sums_of_codomain_features(*, slow: bool) -> None:
	"""A domain interval has two nonzero codomain coefficients and one image coefficient."""
	result = sixpack_from_inclusion(
		[[], [], [], [0, 1], [0, 2], [1, 2]], [0, 0, 0, 1, 1, 1], [5, 2, 1], 1, slow
	)
	assert ensemble_maps(result) == ({1: [0, 1], 2: [1, 2]}, {1: [1], 2: [2]})


@pytest.mark.parametrize("slow", [False, True])
def test_sums_of_image_features(*, slow: bool) -> None:
	"""Different death reductions make a domain feature a sum of image features."""
	result = sixpack_from_inclusion(
		[[], [], [], [], [1, 2], [2, 3], [1, 3], [2, 3]],
		[0, 0, 0, 0, 1, 1, 1, 1],
		[7, 3, 1, 6, 2],
		1,
		slow,
	)
	expected = {1: [1], 2: [2], 3: [2, 3]}
	assert ensemble_maps(result) == (expected, expected)


@pytest.mark.parametrize("slow", [False, True])
def test_dead_targets_are_removed_after_solving(*, slow: bool) -> None:
	"""Exact chain expansion needs a dead cycle, which contributes zero in homology."""
	result = sixpack_from_inclusion(
		[[], [], [0, 1], [], [0, 3], [1, 3]], [0, 0, 1, 0, 1, 1], [5, 3, 1], 1, slow
	)
	assert ensemble_maps(result) == ({1: [0, 1], 3: [3]}, {1: [1], 3: [3]})
	# After target 3 dies, source 3 remains alive but has zero image.
	assert result.domain[3] == 5
	assert result.codomain[3] == result.image[3] == 4


@pytest.mark.parametrize("slow", [False, True])
def test_identity_and_empty_maps(*, slow: bool) -> None:
	"""Identity has diagonal columns; empty domains need no representatives."""
	result = sixpack_from_inclusion([[], [], [0, 1]], [0, 0, 1], [2, 0, 1], 1, slow)
	expected = {0: [0], 1: [1]}
	assert ensemble_maps(result) == (expected, expected)
	for matrix, degrees in (([], []), ([[]], [0])):
		empty = sixpack_from_inclusion(matrix, degrees, [], 1, slow)
		assert ensemble_maps(empty) == ({}, {})


@pytest.mark.parametrize("slow", [False, True])
def test_zero_general_map_in_original_filtration(*, slow: bool) -> None:
	"""Equal-time cylinder artifacts vanish after all cells at that time enter."""
	result, metadata = sixpack([(0.0, 0, [])], [(0.0, 0, [])], [[]], 1, slow)
	to_codomain, to_image = ensemble_maps(result)
	birth = metadata.domain_indices[0]
	for mapping, target in ((to_codomain, result.codomain), (to_image, result.image)):
		assert set(mapping) == {birth}
		assert all(
			target[index] is not None and metadata.times[target[index]] <= metadata.times[birth]
			for index in mapping[birth]
		)


def test_persistence_map_returns_independent_results() -> None:
	"""Neither dictionary edits nor coefficient-list edits mutate or cache the result."""
	result = sixpack_from_inclusion([[], [], [0, 1]], [0, 0, 1], [1], 1)
	to_codomain, to_image = ensemble_maps(result)
	to_codomain[1].clear()
	to_image.clear()
	assert ensemble_maps(result) == ({1: [0, 1]}, {1: [1]})


@pytest.mark.parametrize(
	("source", "source_reps", "target", "target_reps", "mapping", "expected"),
	[
		({1: None}, {1: [1]}, {0: None, 1: None}, {0: [0], 1: [1]}, {1: [0, 1]}, {1: [0, 1]}),
		({1: None}, {1: [0, 1]}, {0: None}, {0: [0]}, {0: [0], 1: [0]}, {1: []}),
		({1: None}, {1: [1]}, {}, {}, {}, {1: []}),
		({1: None}, {1: [1]}, {}, {}, {99: [100]}, {1: []}),
		(
			{1: None},
			{1: [1, 0, 1]},
			{0: None, 1: None},
			{0: [0, 0], 1: [1, 1]},
			{0: [0, 0], 1: [1, 0, 1]},
			{1: [1]},
		),
		({3: None}, {3: [3]}, {0: 2}, {0: [0]}, {3: [0]}, {3: []}),
	],
)
def test_standalone_generator_maps(
	source: dict[int, int | None],
	source_reps: dict[int, list[int]],
	target: dict[int, int | None],
	target_reps: dict[int, list[int]],
	mapping: dict[int, list[int]],
	*,
	expected: dict[int, list[int]],
) -> None:
	"""Supplied images support sums, cancellation, implicit zero, and column normalization."""
	inputs = (source, source_reps, target, target_reps, mapping)
	originals = copy.deepcopy(inputs)
	assert get_persistence_map(*inputs) == expected
	assert inputs == originals


@pytest.mark.parametrize(
	("source", "source_reps", "target", "target_reps", "mapping"),
	[
		({1: None}, {}, {}, {}, {}),
		({1: None}, {1: [0]}, {}, {}, {}),
		({1: 1}, {1: [1]}, {}, {}, {}),
		({1: None}, {1: [1]}, {0: None}, {}, {}),
		({1: None}, {1: [1]}, {0: None}, {0: [0]}, {1: [1]}),
		({1: None}, {1: [1]}, {5: None}, {5: [5]}, {1: [5]}),
		({1: 3}, {1: [1]}, {0: 4}, {0: [0]}, {1: [0]}),
	],
)
def test_standalone_semantic_errors_are_runtime_errors(
	source: dict[int, int | None],
	source_reps: dict[int, list[int]],
	target: dict[int, int | None],
	target_reps: dict[int, list[int]],
	mapping: dict[int, list[int]],
) -> None:
	"""Reject inconsistent diagrams, non-spanning targets, and incompatible lifetimes."""
	with pytest.raises(RuntimeError):
		get_persistence_map(source, source_reps, target, target_reps, mapping)


@pytest.mark.parametrize("diagram", [{"0": None}, {0: "2"}, {0: 2.0}, {-1: None}, {0: -1}])
def test_diagram_input_conversion_errors(diagram: dict) -> None:
	"""The derived LoPhat input conversion retains a contextual TypeError and cause."""
	with pytest.raises(TypeError) as error:
		get_persistence_map(diagram, {}, {}, {}, {})
	assert error.value.__cause__ is not None


@pytest.mark.parametrize("argument", ["representatives", "mapping"])
def test_negative_support_indices_are_overflow_errors(argument: str) -> None:
	"""Support dictionaries use ordinary index extraction rather than diagram conversion."""
	representatives = {0: [-1]} if argument == "representatives" else {0: [0]}
	mapping = {0: [-1]} if argument == "mapping" else {}
	with pytest.raises(OverflowError):
		get_persistence_map({0: None}, representatives, {}, {}, mapping)
