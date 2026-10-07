"""Persistence morphisms are sparse binary columns in the chosen interval bases."""

import pytest
from phimaker import sixpack, sixpack_from_inclusion


@pytest.mark.parametrize("slow", [False, True])
def test_sums_of_codomain_features(*, slow: bool) -> None:
	"""A domain interval has two nonzero codomain coefficients and one image coefficient."""
	result = sixpack_from_inclusion(
		[[], [], [], [0, 1], [0, 2], [1, 2]], [0, 0, 0, 1, 1, 1], [5, 2, 1], 1, slow
	)
	assert result.get_persistence_map() == ({1: [0, 1], 2: [1, 2]}, {1: [1], 2: [2]})


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
	assert result.get_persistence_map() == (expected, expected)


@pytest.mark.parametrize("slow", [False, True])
def test_dead_targets_are_removed_after_solving(*, slow: bool) -> None:
	"""Exact chain expansion needs a dead cycle, which contributes zero in homology."""
	result = sixpack_from_inclusion(
		[[], [], [0, 1], [], [0, 3], [1, 3]], [0, 0, 1, 0, 1, 1], [5, 3, 1], 1, slow
	)
	assert result.get_persistence_map() == ({1: [0, 1], 3: [3]}, {1: [1], 3: [3]})
	# After target 3 dies, source 3 remains alive but has zero image.
	assert result.domain[3] == 5
	assert result.codomain[3] == result.image[3] == 4


@pytest.mark.parametrize("slow", [False, True])
def test_identity_and_empty_maps(*, slow: bool) -> None:
	"""Identity has diagonal columns; empty domains need no representatives."""
	result = sixpack_from_inclusion([[], [], [0, 1]], [0, 0, 1], [2, 0, 1], 1, slow)
	expected = {0: [0], 1: [1]}
	assert result.get_persistence_map() == (expected, expected)
	for matrix, degrees in (([], []), ([[]], [0])):
		empty = sixpack_from_inclusion(matrix, degrees, [], 1, slow)
		assert empty.get_persistence_map() == ({}, {})


@pytest.mark.parametrize("slow", [False, True])
def test_zero_general_map_in_original_filtration(*, slow: bool) -> None:
	"""Equal-time cylinder artifacts vanish after all cells at that time enter."""
	result, metadata = sixpack([(0.0, 0, [])], [(0.0, 0, [])], [[]], 1, slow)
	to_codomain, to_image = result.get_persistence_map()
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
	to_codomain, to_image = result.get_persistence_map()
	to_codomain[1].clear()
	to_image.clear()
	assert result.get_persistence_map() == ({1: [0, 1]}, {1: [1]})
