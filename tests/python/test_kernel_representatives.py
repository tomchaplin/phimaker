"""Kernel cycles represent independent domain classes that bound in the codomain."""

import pytest
from phimaker import DiagramEnsemble, sixpack, sixpack_from_inclusion


def interval_alive(diagram: dict[int, int | None], birth: int, index: int) -> bool:
	"""Evaluate half-open intervals, retaining essential features indefinitely."""
	death = diagram[birth]
	return birth <= index and (death is None or index < death)


def binary_rank(columns: list[set[int]]) -> int:
	"""Count independent active columns by sparse elimination over F2."""
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


def assert_kernel_cycles(
	result: DiagramEnsemble,
	matrix: list[list[int]],
	degrees: list[int],
	domain: list[int],
	*,
	times: list[float] | None = None,
) -> None:
	"""Verify cycles, independence in domain homology, and vanishing in codomain homology."""
	assert result.kernel_representatives is not None
	assert result.domain_representatives is not None
	representatives = result.kernel_representatives
	assert representatives.keys() == result.kernel.keys()
	for birth, support in representatives.items():
		assert support
		assert len(set(support)) == len(support)
		assert all(idx in domain and idx < birth for idx in support)
		assert all(degrees[idx] + 1 == degrees[birth] for idx in support)
		boundary: set[int] = set()
		for idx in support:
			boundary.symmetric_difference_update(matrix[idx])
		assert not boundary
	for index in range(len(matrix) + 1):
		if times is not None and index + 1 < len(times) and times[index] == times[index + 1]:
			continue  # A public sixpack result is evaluated after the whole time block.
		active = [birth for birth in result.kernel if interval_alive(result.kernel, birth, index)]
		cycles = [set(representatives[birth]) for birth in active]
		domain_boundaries = [set(matrix[idx]) for idx in domain if idx <= index]
		codomain_boundaries = [set(column) for column in matrix[: index + 1]]
		assert binary_rank(domain_boundaries + cycles) == binary_rank(domain_boundaries) + len(
			active
		)
		assert binary_rank(codomain_boundaries + cycles) == binary_rank(codomain_boundaries)
		active_domain = sum(interval_alive(result.domain, birth, index) for birth in result.domain)
		active_image = sum(interval_alive(result.image, birth, index) for birth in result.image)
		assert len(active) == active_domain - active_image


@pytest.mark.parametrize("slow", [False, True])
@pytest.mark.parametrize(
	("matrix", "degrees", "domain", "expected"),
	[
		(
			[[], [], [], [0, 1], [0, 2], [1, 2]],
			[0, 0, 0, 1, 1, 1],
			[5, 2, 1],
			{4: [1, 2]},
		),
		(
			[[], [], [], [0, 1], [0, 2], [1, 2], [0, 1]],
			[0, 0, 0, 1, 1, 1, 1],
			[6, 2, 0, 5, 1],
			{3: [0, 1], 4: [1, 2]},
		),
		(
			[[], [], [], [0, 1], [1, 2], [0, 2], [3, 4, 5]],
			[0, 0, 0, 1, 1, 1, 2],
			[0, 1, 2, 3, 4, 5],
			{6: [3, 4, 5]},
		),
	],
	ids=["gapped-domain", "finite-death-basis", "essential-loop"],
)
def test_kernel_representatives(
	matrix: list[list[int]],
	degrees: list[int],
	domain: list[int],
	expected: dict[int, list[int]],
	*,
	slow: bool,
) -> None:
	"""Kernel cycles use domain generators and respect finite and essential lifetimes."""
	result = sixpack_from_inclusion(
		matrix, degrees, domain, get_representatives=True, num_threads=1, slow=slow
	)
	assert result.kernel_representatives == expected
	assert_kernel_cycles(result, matrix, degrees, domain)


@pytest.mark.parametrize("slow", [False, True])
def test_kernel_representative_of_zero_map_cylinder(*, slow: bool) -> None:
	"""A shifted birth represents the original domain cycle in a zero-map cylinder."""
	point = [(0.0, 0, [])]
	result, metadata = sixpack(
		point, point, [[]], get_representatives=True, num_threads=1, slow=slow
	)
	birth = metadata.domain_shift[0]
	domain_index = metadata.domain_indices[0]
	assert result.kernel_representatives == {birth: [domain_index]}
	matrix = [[], [], [domain_index]]
	assert_kernel_cycles(
		result, matrix, metadata.degrees, metadata.domain_indices, times=metadata.times
	)
