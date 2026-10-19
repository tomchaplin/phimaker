"""Representative supports use diagram coordinates and coefficients in F2."""

from typing import Any

import pytest
from phimaker import DiagramEnsemble, sixpack, sixpack_from_inclusion


@pytest.mark.parametrize("slow", [False, True])
@pytest.mark.parametrize("case", ["empty", "empty_domain", "inclusion", "general_empty", "general"])
def test_optional_representatives(*, slow: bool, case: str) -> None:
	"""Default and explicit opt-out skip cycles; opting in preserves all six diagrams."""

	def compute(**options: bool) -> DiagramEnsemble:
		if case == "general_empty":
			return sixpack([], [], [], num_threads=1, slow=slow, **options)[0]
		if case == "general":
			point = [(0.0, 0, [])]
			return sixpack(point, point, [[0]], num_threads=1, slow=slow, **options)[0]
		if case == "empty":
			return sixpack_from_inclusion([], [], [], num_threads=1, slow=slow, **options)
		if case == "empty_domain":
			return sixpack_from_inclusion([[]], [0], [], num_threads=1, slow=slow, **options)
		return sixpack_from_inclusion(
			[[], [], [], [1, 2], [1, 2], [3, 4]],
			[0, 0, 0, 1, 1, 2],
			[4, 2, 1, 3],
			num_threads=1,
			slow=slow,
			**options,
		)

	default = compute()
	skipped = compute(get_representatives=False)
	requested = compute(get_representatives=True)
	for name in ("domain", "codomain", "image", "kernel", "cokernel", "relative"):
		assert getattr(default, name) == getattr(skipped, name) == getattr(requested, name)
	for name in ("domain", "codomain", "image", "kernel", "cokernel", "relative"):
		assert getattr(default, f"{name}_representatives") is None
		assert getattr(skipped, f"{name}_representatives") is None
		representatives = getattr(requested, f"{name}_representatives")
		assert isinstance(representatives, dict)
		assert representatives.keys() == getattr(requested, name).keys()
		for birth, representative in representatives.items():
			support = representative[1] if name == "relative" else representative
			if name == "kernel":
				assert max(support) < birth
			else:
				assert max(support) == birth

	assert_optional_maps(default, skipped, requested)


def assert_optional_maps(
	default: DiagramEnsemble, skipped: DiagramEnsemble, requested: DiagramEnsemble
) -> None:
	"""Skipped map fields are None and requested maps include every source feature."""
	for name, source in (
		("kernel_to_domain", "kernel"),
		("domain_to_image", "domain"),
		("image_to_codomain", "image"),
		("domain_to_codomain", "domain"),
		("codomain_to_cokernel", "codomain"),
	):
		assert getattr(default, name) is None
		assert getattr(skipped, name) is None
		assert getattr(requested, name).keys() == getattr(requested, source).keys()


def test_representative_flag_accepts_positional_argument() -> None:
	"""The representative flag precedes worker count and slow mode in positional calls."""
	result = sixpack_from_inclusion([[]], [0], [0], True, 1, False)  # noqa: FBT003
	assert result.domain_representatives == {0: [0]}


def assert_representatives(result: Any, matrix: list[list[int]], domain: list[int]) -> None:
	"""Each birth has a cycle in its ambient or quotient complex, in input coordinates."""
	for name in ("domain", "codomain", "image", "kernel", "cokernel", "relative"):
		diagram = getattr(result, name)
		representatives = getattr(result, f"{name}_representatives")
		assert isinstance(representatives, dict)
		assert representatives.keys() == diagram.keys()
		for birth, representative in representatives.items():
			support = representative[1] if name == "relative" else representative
			assert len(support) == len(set(support))
			if name == "kernel":
				assert max(support) < birth
			else:
				assert max(support) == birth
			boundary: set[int] = set()
			for generator in support:
				boundary.symmetric_difference_update(matrix[generator])
			if name == "relative":
				assert not set(support) & set(domain)
				assert boundary <= set(domain)
				assert set(representative[0]) == boundary
			else:
				assert not boundary


@pytest.mark.parametrize("slow", [False, True])
def test_representatives_with_permuted_domain(*, slow: bool) -> None:
	"""Unpermute keys and supports; the domain's essential loop dies in the image."""
	matrix = [[], [], [], [1, 2], [1, 2], [3, 4]]
	result = sixpack_from_inclusion(
		matrix, [0, 0, 0, 1, 1, 2], [4, 2, 1, 3], num_threads=1, slow=slow, get_representatives=True
	)
	expected = {1: [1], 2: [1, 2], 4: [3, 4]}
	assert result.domain_representatives == expected
	assert result.image_representatives == expected
	assert result.codomain_representatives == {0: [0], **expected}
	assert result.domain[4] is None
	assert result.image[4] == 5
	assert_representatives(result, matrix, [4, 2, 1, 3])


@pytest.mark.parametrize("slow", [False, True])
def test_essential_image_representative_is_in_domain(*, slow: bool) -> None:
	"""A younger vertex's image cycle must not include the external older vertex."""
	matrix = [[], [], [0, 1]]
	result = sixpack_from_inclusion(
		matrix, [0, 0, 1], [1], num_threads=1, slow=slow, get_representatives=True
	)
	assert result.domain_representatives == {1: [1]}
	assert result.image_representatives == {1: [1]}
	assert result.codomain_representatives == {0: [0], 1: [0, 1]}
	assert_representatives(result, matrix, [1])


@pytest.mark.parametrize("slow", [False, True])
def test_empty_representative_maps(*, slow: bool) -> None:
	"""Every empty diagram has an empty representative dictionary."""
	empty = sixpack_from_inclusion([], [], [], num_threads=1, slow=slow, get_representatives=True)
	assert empty.domain_representatives == {}
	assert empty.codomain_representatives == {}
	assert empty.image_representatives == {}
	assert empty.kernel_representatives == {}
	assert empty.cokernel_representatives == {}
	assert empty.relative_representatives == {}
	result = sixpack_from_inclusion(
		[[]], [0], [], num_threads=1, slow=slow, get_representatives=True
	)
	assert result.domain_representatives == {}
	assert result.codomain_representatives == {0: [0]}
	assert result.image_representatives == {}
	assert result.kernel_representatives == {}
	assert result.cokernel_representatives == {0: [0]}
	assert result.relative_representatives == {0: ([], [0])}


def test_representative_getters_return_independent_lists() -> None:
	"""Neither dictionary edits nor nested list edits mutate the stored result."""
	result = sixpack_from_inclusion(
		[[], [], [], [0, 1], [0, 2], [1, 2]],
		[0, 0, 0, 1, 1, 1],
		[1, 2, 5],
		num_threads=1,
		get_representatives=True,
	)
	for name in (
		"domain_representatives",
		"codomain_representatives",
		"image_representatives",
		"kernel_representatives",
		"cokernel_representatives",
		"relative_representatives",
	):
		expected = getattr(result, name)
		representatives = getattr(result, name)
		value = representatives[next(iter(representatives))]
		for support in value if name == "relative_representatives" else (value,):
			support.append(99)
		representatives.clear()
		assert getattr(result, name) == expected


def test_map_getters_return_independent_lists() -> None:
	"""All five stored morphisms expose independent dictionaries and coefficient lists."""
	result = sixpack_from_inclusion(
		[[], [], [], [0, 1], [0, 2], [1, 2]],
		[0, 0, 0, 1, 1, 1],
		[1, 2, 5],
		num_threads=1,
		get_representatives=True,
	)
	for name in (
		"kernel_to_domain",
		"domain_to_image",
		"image_to_codomain",
		"domain_to_codomain",
		"codomain_to_cokernel",
	):
		expected = getattr(result, name)
		mapping = getattr(result, name)
		mapping[next(iter(mapping))].append(99)
		mapping.clear()
		assert getattr(result, name) == expected
