"""Relative cycles form homology bases modulo domain chains and codomain boundaries."""

import pytest
from phimaker import DiagramEnsemble, sixpack, sixpack_from_inclusion


def binary_rank(columns: list[set[int]]) -> int:
	"""Compute a small F2 rank independently of the package's reductions."""
	pivots: dict[int, set[int]] = {}
	for column in columns:
		residual = column.copy()
		while residual:
			pivot = max(residual)
			if pivot in pivots:
				residual.symmetric_difference_update(pivots[pivot])
			else:
				pivots[pivot] = residual
				break
	return len(pivots)


def quotient_boundaries(
	matrix: list[list[int]], degrees: list[int], domain: set[int], index: int, degree: int
) -> list[set[int]]:
	"""Project input boundaries by deleting domain rows; retain ambient labels."""
	return [
		set(column) - domain
		for generator, column in enumerate(matrix[: index + 1])
		if generator not in domain and degrees[generator] == degree
	]


def assert_relative_cycles(
	result: DiagramEnsemble, matrix: list[list[int]], degrees: list[int], domain_indices: list[int]
) -> None:
	"""Check relative cyclehood, degrees, quotient bases at every index, and finite deaths."""
	domain = set(domain_indices)
	assert result.relative_representatives is not None
	representatives = result.relative_representatives
	assert representatives.keys() == result.relative.keys()
	for birth, support in representatives.items():
		assert max(support) == birth
		assert len(support) == len(set(support))
		assert not set(support) & domain
		assert all(degrees[generator] == degrees[birth] for generator in support)
		boundary: set[int] = set()
		for generator in support:
			boundary.symmetric_difference_update(matrix[generator])
		assert boundary <= domain
		death = result.relative[birth]
		if death is not None:
			relations = quotient_boundaries(matrix, degrees, domain, death, degrees[birth] + 1)
			assert binary_rank([*relations, set(support)]) == binary_rank(relations)
	for index in range(len(matrix) + 1):
		for degree in range(max(degrees, default=0) + 1):
			n_chains = sum(
				generator not in domain and generator_degree == degree
				for generator, generator_degree in enumerate(degrees[: index + 1])
			)
			outgoing = quotient_boundaries(matrix, degrees, domain, index, degree)
			relations = quotient_boundaries(matrix, degrees, domain, index, degree + 1)
			boundary_rank = binary_rank(relations)
			cycle_dimension = n_chains - binary_rank(outgoing)
			active = [
				set(support)
				for birth, support in representatives.items()
				if birth <= index
				and degrees[birth] == degree
				and (result.relative[birth] is None or index < result.relative[birth])
			]
			assert binary_rank(relations + active) == boundary_rank + len(active)
			assert boundary_rank + len(active) == cycle_dimension


@pytest.mark.parametrize("slow", [False, True])
@pytest.mark.parametrize(
	("matrix", "degrees", "domain", "expected"),
	[
		([], [], [], {}),
		([[]], [0], [], {0: [0]}),
		([[], [], [0, 1]], [0, 0, 1], [2, 1, 0], {}),
		([[], [], [0, 1]], [0, 0, 1], [1], {0: [0]}),
		([[], [], [0, 1]], [0, 0, 1], [1, 0], {2: [2]}),
		([[], [], [0, 1], [0, 1]], [0, 0, 1, 1], [], {0: [0], 1: [0, 1], 3: [2, 3]}),
		(
			[[], [], [], [0, 1], [0, 2], [1, 2], [3, 4, 5]],
			[0, 0, 0, 1, 1, 1, 2],
			[3, 1, 0],
			{2: [2], 5: [4, 5]},
		),
		(
			[[], [], [], [0, 1], [0, 2], [1, 2], [3, 4, 5]],
			[0, 0, 0, 1, 1, 1, 2],
			[5, 4, 3, 2, 1, 0],
			{6: [6]},
		),
	],
	ids=[
		"empty",
		"empty-domain",
		"identity",
		"finite-vertex",
		"relative-edge",
		"essential-loop",
		"gapped-domain-finite-path",
		"triangle-relative-to-boundary",
	],
)
def test_relative_cycles_form_quotient_bases(
	matrix: list[list[int]],
	degrees: list[int],
	domain: list[int],
	expected: dict[int, list[int]],
	*,
	slow: bool,
) -> None:
	"""Finite and essential representatives give a quotient basis at every filtration index."""
	result = sixpack_from_inclusion(
		matrix, degrees, domain, get_representatives=True, num_threads=1, slow=slow
	)
	assert result.relative_representatives == expected
	assert_relative_cycles(result, matrix, degrees, domain)


@pytest.mark.parametrize("slow", [False, True])
def test_relative_cycle_need_not_be_a_codomain_cycle(*, slow: bool) -> None:
	"""An edge is an essential relative H1 cycle when both endpoints are in the domain."""
	matrix = [[], [], [0, 1]]
	result = sixpack_from_inclusion(
		matrix, [0, 0, 1], [0, 1], get_representatives=True, num_threads=1, slow=slow
	)
	assert result.relative == {2: None}
	assert result.relative_representatives == {2: [2]}
	assert matrix[2] == [0, 1]


@pytest.mark.parametrize("slow", [False, True])
@pytest.mark.parametrize("mapping", [[[0]], [[]]], ids=["identity", "zero"])
def test_relative_cycles_in_mapping_cylinders(mapping: list[list[int]], *, slow: bool) -> None:
	"""Retain cylinder indices, including the suspended domain feature for a zero map."""
	point = [(0.0, 0, [])]
	result, metadata = sixpack(
		point, point, mapping, get_representatives=True, num_threads=1, slow=slow
	)
	codomain = metadata.codomain_indices[0]
	shifted = metadata.domain_shift[0]
	expected = {codomain: [codomain]}
	if not mapping[0]:
		expected[shifted] = [shifted]
	assert result.relative_representatives == expected
	matrix: list[list[int]] = [[] for _ in metadata.times]
	matrix[shifted] = [metadata.domain_indices[0], *([codomain] if mapping[0] else [])]
	assert_relative_cycles(result, matrix, metadata.degrees, metadata.domain_indices)
