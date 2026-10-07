"""Representative supports use diagram coordinates and coefficients in F2."""

from typing import Any

import pytest
from phimaker import sixpack_from_inclusion


def assert_representatives(result: Any, matrix: list[list[int]]) -> None:
	"""Each birth has one cycle supported at its birth, in the original basis."""
	for name in ("domain", "codomain", "image"):
		diagram = getattr(result, name)
		representatives = getattr(result, f"{name}_representatives")
		assert isinstance(representatives, dict)
		assert representatives.keys() == diagram.keys()
		for birth, support in representatives.items():
			assert len(support) == len(set(support))
			assert max(support) == birth
			boundary: set[int] = set()
			for generator in support:
				boundary.symmetric_difference_update(matrix[generator])
			assert not boundary


@pytest.mark.parametrize("slow", [False, True])
def test_representatives_with_permuted_domain(*, slow: bool) -> None:
	"""Unpermute keys and supports; the domain's essential loop dies in the image."""
	matrix = [[], [], [], [1, 2], [1, 2], [3, 4]]
	result = sixpack_from_inclusion(matrix, [0, 0, 0, 1, 1, 2], [4, 2, 1, 3], 1, slow)
	expected = {1: [1], 2: [1, 2], 4: [3, 4]}
	assert result.domain_representatives == expected
	assert result.image_representatives == expected
	assert result.codomain_representatives == {0: [0], **expected}
	assert result.domain[4] is None
	assert result.image[4] == 5
	assert_representatives(result, matrix)


@pytest.mark.parametrize("slow", [False, True])
def test_essential_image_representative_is_in_domain(*, slow: bool) -> None:
	"""A younger vertex's image cycle must not include the external older vertex."""
	matrix = [[], [], [0, 1]]
	result = sixpack_from_inclusion(matrix, [0, 0, 1], [1], 1, slow)
	assert result.domain_representatives == {1: [1]}
	assert result.image_representatives == {1: [1]}
	assert result.codomain_representatives == {0: [0], 1: [0, 1]}
	assert_representatives(result, matrix)


@pytest.mark.parametrize("slow", [False, True])
def test_empty_representative_maps(*, slow: bool) -> None:
	"""Every empty diagram has an empty representative dictionary."""
	empty = sixpack_from_inclusion([], [], [], 1, slow)
	assert empty.domain_representatives == {}
	assert empty.codomain_representatives == {}
	assert empty.image_representatives == {}
	result = sixpack_from_inclusion([[]], [0], [], 1, slow)
	assert result.domain_representatives == {}
	assert result.codomain_representatives == {0: [0]}
	assert result.image_representatives == {}


def test_representative_getters_return_independent_lists() -> None:
	"""Neither dictionary edits nor nested list edits mutate the stored result."""
	result = sixpack_from_inclusion([[]], [0], [0], 1)
	for name in ("domain_representatives", "codomain_representatives", "image_representatives"):
		representatives = getattr(result, name)
		representatives[0].append(1)
		representatives.clear()
		assert getattr(result, name) == {0: [0]}
