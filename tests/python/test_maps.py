"""General chain maps are tested in filtration times and homological degrees."""

from typing import Any

import pytest
from phimaker import sixpack

NAMES = ("domain", "codomain", "image", "kernel", "cokernel", "relative")


def timed_bars(result: Any, metadata: Any, name: str) -> list[tuple[int, float, float]]:
	"""Remove cylinder-only zero-duration bars; retain degree and multiplicity."""
	return sorted(
		(
			metadata.dimensions[b] - (name == "kernel"),
			metadata.times[b],
			float("inf") if d is None else metadata.times[d],
		)
		for b, d in getattr(result, name).items()
		if d is None or metadata.times[b] != metadata.times[d]
	)


@pytest.mark.parametrize("slow", [False, True])
def test_identity_chain_map(*, slow: bool) -> None:
	"""Identity preserves homology and has acyclic cone, even with tied times."""
	interval = [(0.0, 0, []), (0.0, 0, []), (1.0, 1, [0, 1])]
	result, metadata = sixpack(interval, interval, [[0], [1], [2]], 1, slow)
	for name in NAMES:
		expected = [(0, 0.0, 1.0), (0, 0.0, float("inf"))] if name in NAMES[:3] else []
		assert timed_bars(result, metadata, name) == expected, name
	assert len(metadata.times) == 9
	assert metadata.times == sorted(metadata.times)
	indices = metadata.domain_indices + metadata.codomain_indices + metadata.domain_shift
	assert sorted(indices) == list(range(9))
	for original, (time, degree, _) in enumerate(interval):
		for name, shift in (("domain_indices", 0), ("codomain_indices", 0), ("domain_shift", 1)):
			index = getattr(metadata, name)[original]
			assert metadata.times[index] == time
			assert metadata.dimensions[index] == degree + shift


@pytest.mark.parametrize("slow", [False, True])
def test_zero_map_of_points(*, slow: bool) -> None:
	"""A zero map has full kernel/cokernel and a suspended domain in its cone."""
	result, metadata = sixpack([(0.0, 0, [])], [(0.0, 0, [])], [[]], 1, slow)
	point = [(0, 0.0, float("inf"))]
	expected = {
		"domain": point,
		"codomain": point,
		"image": [],
		"kernel": point,
		"cokernel": point,
		"relative": [*point, (1, 0.0, float("inf"))],
	}
	for name in NAMES:
		assert timed_bars(result, metadata, name) == expected[name], name


@pytest.mark.parametrize("slow", [False, True])
def test_square_fills_earlier_in_codomain(*, slow: bool) -> None:
	"""The old square script now checks all bars; its diagonal edge has degree one."""
	vertices = [(0.0, 0, []) for _ in range(4)]
	edges = [[0, 1], [1, 2], [2, 3], [0, 3]]
	domain = vertices + [(1.0, 1, edge) for edge in edges] + [(10.0, 2, [4, 5, 6, 7])]
	codomain = (
		vertices
		+ [(0.1, 1, edge) for edge in edges]
		+ [
			(2.0, 1, [0, 2]),
			(2.0, 2, [4, 5, 8]),
			(2.0, 2, [6, 7, 8]),
		]
	)
	result, metadata = sixpack(domain, codomain, [[i] for i in range(8)] + [[9, 10]], 1, slow)
	essential = [(0, 0.0, float("inf"))]
	expected = {
		"domain": [(0, 0.0, 1.0)] * 3 + essential + [(1, 1.0, 10.0)],
		"codomain": [(0, 0.0, 0.1)] * 3 + essential + [(1, 0.1, 2.0)],
		"image": [(0, 0.0, 0.1)] * 3 + essential + [(1, 1.0, 2.0)],
		"kernel": [(0, 0.1, 1.0)] * 3 + [(1, 2.0, 10.0)],
		"cokernel": [(1, 0.1, 1.0)],
		"relative": [(1, 0.1, 1.0)] * 4 + [(2, 2.0, 10.0)],
	}
	for name in NAMES:
		assert timed_bars(result, metadata, name) == sorted(expected[name]), name


def test_empty_general_map() -> None:
	"""An empty map has empty diagrams and empty coordinate metadata."""
	result, metadata = sixpack([], [], [], num_threads=1)
	for name in NAMES:
		assert getattr(result, name) == {}
	for name in ("times", "dimensions", "domain_indices", "codomain_indices", "domain_shift"):
		assert getattr(metadata, name) == []
