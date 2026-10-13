"""Cokernel cycles form quotient bases modulo boundaries and domain cycles."""

import pytest
from phimaker import DiagramEnsemble, sixpack, sixpack_from_inclusion


def binary_rank(columns: list[set[int]]) -> int:
	"""Compute an independent small-matrix rank oracle over F2."""
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


def quotient_relations(
	result: DiagramEnsemble, matrix: list[list[int]], index: int
) -> list[set[int]]:
	"""Keep all available domain cycles, including features already dead in homology."""
	assert result.domain_representatives is not None
	return [set(column) for column in matrix[: index + 1]] + [
		set(support) for birth, support in result.domain_representatives.items() if birth <= index
	]


def assert_cokernel_cycles(
	result: DiagramEnsemble, matrix: list[list[int]], degrees: list[int]
) -> None:
	"""Check cyclehood, degree, quotient independence/completeness, and finite deaths."""
	assert result.cokernel_representatives is not None
	representatives = result.cokernel_representatives
	assert representatives.keys() == result.cokernel.keys()
	for birth, support in representatives.items():
		assert max(support) == birth
		assert len(support) == len(set(support))
		assert all(degrees[idx] == degrees[birth] for idx in support)
		boundary: set[int] = set()
		for idx in support:
			boundary.symmetric_difference_update(matrix[idx])
		assert not boundary
		death = result.cokernel[birth]
		if death is not None:
			relations = quotient_relations(result, matrix, death)
			assert binary_rank([*relations, set(support)]) == binary_rank(relations)
	for index in range(len(matrix) + 1):
		relations = quotient_relations(result, matrix, index)
		active = [
			set(support)
			for birth, support in representatives.items()
			if birth <= index and (result.cokernel[birth] is None or index < result.cokernel[birth])
		]
		rank = binary_rank(relations)
		assert binary_rank(relations + active) == rank + len(active)
		assert rank + len(active) == sum(birth <= index for birth in result.codomain)
	assert result.image_representatives is not None
	assert not representatives.keys() & result.image_representatives.keys()
	assert binary_rank(
		[set(support) for support in result.image_representatives.values()]
		+ [set(support) for support in representatives.values()]
	) == len(result.codomain)


@pytest.mark.parametrize("slow", [False, True])
@pytest.mark.parametrize(
	("matrix", "degrees", "domain", "expected"),
	[
		([], [], [], {}),
		([[]], [0], [], {0: [0]}),
		([[], [], [0, 1]], [0, 0, 1], [2, 1, 0], {}),
		([[], [], [0, 1]], [0, 0, 1], [1], {0: [0]}),
		([[], [], [], [0, 2]], [0, 0, 0, 1], [1], {0: [0], 2: [0, 2]}),
		([[], [], [0, 1], [0, 1]], [0, 0, 1, 1], [3, 1, 0], {3: [2, 3]}),
		([[], [], [0, 1], [0, 1], [2, 3]], [0, 0, 1, 1, 2], [0, 1, 3], {3: [2, 3]}),
		(
			[[], [], [], [1, 2], [1, 2], [3, 4]],
			[0, 0, 0, 1, 1, 2],
			[4, 2, 1, 3],
			{0: [0]},
		),
	],
	ids=[
		"empty",
		"empty-domain",
		"identity",
		"finite-quotient",
		"finite-death-basis",
		"essential-loop",
		"finite-loop",
		"gapped-domain",
	],
)
def test_cokernel_cycles_form_quotient_bases(
	matrix: list[list[int]],
	degrees: list[int],
	domain: list[int],
	expected: dict[int, list[int]],
	*,
	slow: bool,
) -> None:
	"""Finite and essential cycles give the quotient basis at every filtration index."""
	result = sixpack_from_inclusion(
		matrix, degrees, domain, get_representatives=True, num_threads=1, slow=slow
	)
	assert result.cokernel_representatives == expected
	assert_cokernel_cycles(result, matrix, degrees)


@pytest.mark.parametrize("slow", [False, True])
def test_finite_cycle_is_zero_only_in_quotient_at_death(*, slow: bool) -> None:
	"""A vertex merges with the domain while remaining nonzero in codomain homology."""
	result = sixpack_from_inclusion(
		[[], [], [0, 1]], [0, 0, 1], [1], get_representatives=True, num_threads=1, slow=slow
	)
	assert result.cokernel == {0: 2}
	assert result.codomain[0] is None
	assert result.cokernel_representatives == {0: [0]}
	assert binary_rank([{0, 1}, {0}]) == 2


@pytest.mark.parametrize("slow", [False, True])
@pytest.mark.parametrize("mapping", [[[0]], [[]]], ids=["identity", "zero"])
def test_cokernel_cycles_in_mapping_cylinders(mapping: list[list[int]], *, slow: bool) -> None:
	"""Cylinders retain ambient indices and zero-duration quotient features."""
	point = [(0.0, 0, [])]
	result, metadata = sixpack(
		point, point, mapping, get_representatives=True, num_threads=1, slow=slow
	)
	birth = metadata.codomain_indices[0]
	support = [metadata.domain_indices[0], birth] if mapping[0] else [birth]
	assert result.cokernel_representatives == {birth: support}
	shift_boundary = [metadata.domain_indices[0], *([birth] if mapping[0] else [])]
	matrix = [[], [], shift_boundary]
	assert_cokernel_cycles(result, matrix, metadata.degrees)
