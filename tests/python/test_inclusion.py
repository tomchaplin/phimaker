"""Exact six-diagram examples over F2; expected bars are derived by hand."""

from typing import Any

import pytest
from phimaker import sixpack_from_inclusion

NAMES = ("domain", "codomain", "image", "kernel", "cokernel", "relative")


def assert_diagrams(result: Any, expected: dict[str, dict[int, int | None]]) -> None:
	"""Compare every diagram, including empty ones and essential classes."""
	assert set(expected) == set(NAMES)
	for name in NAMES:
		actual = getattr(result, name)
		assert isinstance(actual, dict)
		assert actual == expected[name], name


@pytest.mark.parametrize("slow", [False, True])
@pytest.mark.parametrize("vertex", [0, 1])
def test_vertex_into_interval(vertex: int, *, slow: bool) -> None:
	"""Domain-first row order must not change ordinary codomain elder pairing."""
	result = sixpack_from_inclusion([[], [], [0, 1]], [0, 0, 1], [vertex], 1, slow)
	assert_diagrams(
		result,
		{
			"domain": {vertex: None},
			"codomain": {0: None, 1: 2},
			"image": {vertex: None},
			"kernel": {},
			"cokernel": {1 - vertex: 2},
			"relative": {1 - vertex: 2},
		},
	)


@pytest.mark.parametrize("slow", [False, True])
@pytest.mark.parametrize("domain", [[0, 1], [1, 0]])
def test_unsorted_domain_with_finite_class(domain: list[int], *, slow: bool) -> None:
	"""Sorting domain indices must keep local columns and row coordinates aligned."""
	result = sixpack_from_inclusion([[], [0], [], [2]], [0, 1, 0, 1], domain, 1, slow)
	assert_diagrams(
		result,
		{
			"domain": {0: 1},
			"codomain": {0: 1, 2: 3},
			"image": {0: 1},
			"kernel": {},
			"cokernel": {2: 3},
			"relative": {2: 3},
		},
	)


@pytest.mark.parametrize("slow", [False, True])
def test_boundary_circle_filled_in_codomain(*, slow: bool) -> None:
	"""A triangle boundary has an essential H1 kernel after the face enters."""
	result = sixpack_from_inclusion(
		[[], [], [], [0, 1], [1, 2], [0, 2], [3, 4, 5]],
		[0, 0, 0, 1, 1, 1, 2],
		list(range(6)),
		1,
		slow,
	)
	assert_diagrams(
		result,
		{
			"domain": {0: None, 1: 3, 2: 4, 5: None},
			"codomain": {0: None, 1: 3, 2: 4, 5: 6},
			"image": {0: None, 1: 3, 2: 4, 5: 6},
			"kernel": {6: None},
			"cokernel": {},
			"relative": {6: None},
		},
	)


@pytest.mark.parametrize("slow", [False, True])
def test_face_into_tetrahedron_boundary(*, slow: bool) -> None:
	"""The old print-only tetrahedron fixture detects a finite kernel and H2 cokernel."""
	result = sixpack_from_inclusion(
		[
			[],
			[],
			[],
			[],
			[0, 1],
			[0, 2],
			[1, 2],
			[0, 3],
			[1, 3],
			[2, 3],
			[4, 7, 8],
			[5, 7, 9],
			[6, 8, 9],
			[4, 5, 6],
		],
		[0] * 4 + [1] * 6 + [2] * 4,
		[0, 1, 2, 4, 5, 6, 13],
		1,
		slow,
	)
	assert_diagrams(
		result,
		{
			"domain": {0: None, 1: 4, 2: 5, 6: 13},
			"codomain": {0: None, 1: 4, 2: 5, 3: 7, 6: 12, 8: 10, 9: 11, 13: None},
			"image": {0: None, 1: 4, 2: 5, 6: 12},
			"kernel": {12: 13},
			"cokernel": {3: 7, 8: 10, 9: 11, 13: None},
			"relative": {3: 7, 8: 10, 9: 11, 12: None},
		},
	)


@pytest.mark.parametrize(
	("matrix", "dimensions", "domain", "expected"),
	[
		([], [], [], {name: {} for name in NAMES}),
		(
			[[]],
			[0],
			[],
			{
				name: ({0: None} if name in ("codomain", "cokernel", "relative") else {})
				for name in NAMES
			},
		),
		(
			[[]],
			[0],
			[0],
			{
				name: ({0: None} if name in ("domain", "codomain", "image") else {})
				for name in NAMES
			},
		),
		(
			[[], [0]],
			[0, 1],
			[1, 0],
			{name: ({0: 1} if name in ("domain", "codomain", "image") else {}) for name in NAMES},
		),
	],
	ids=["empty", "empty-domain", "identity-point", "identity-acyclic"],
)
def test_empty_and_identity_inclusions(
	matrix: list[list[int]],
	dimensions: list[int],
	domain: list[int],
	expected: dict[str, dict[int, int | None]],
) -> None:
	"""Empty quotients have no basepoint; identity maps have zero kernel and cokernel."""
	assert_diagrams(sixpack_from_inclusion(matrix, dimensions, domain, 1), expected)


def test_dictionary_lookup_and_copy() -> None:
	"""Python getters return independent dictionaries and use None for infinity."""
	result = sixpack_from_inclusion([[], [], [0, 1]], [0, 0, 1], [0], 1)
	diagram = result.codomain
	assert diagram[1] == 2
	assert diagram[0] is None
	assert set(diagram.items()) == {(0, None), (1, 2)}
	with pytest.raises(KeyError):
		_ = diagram[2]
	diagram[1] = None
	del diagram[0]
	assert result.codomain == {0: None, 1: 2}
