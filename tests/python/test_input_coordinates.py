"""Check returned chains against the original complexes, independently of the cylinder."""

import pytest
from phimaker import CylinderMetadata, DiagramEnsemble, sixpack

type Matrix = list[tuple[float, int, list[int]]]

POINT: Matrix = [(0.0, 0, [])]
VERTICES: Matrix = [(0.0, 0, []), (0.0, 0, [])]
INTERVAL: Matrix = [*VERTICES, (1.0, 1, [0, 1])]
TRIANGLE_BOUNDARY: Matrix = [
	(0.0, 0, []),
	(0.0, 0, []),
	(0.0, 0, []),
	(1.0, 1, [0, 1]),
	(1.0, 1, [0, 2]),
	(1.0, 1, [1, 2]),
]
CASES = [
	pytest.param([], [], [], id="empty"),
	pytest.param([], POINT, [], id="empty-domain"),
	pytest.param(POINT, [], [[]], id="empty-codomain"),
	pytest.param(VERTICES, POINT, [[0], [0]], id="non-injective"),
	pytest.param(VERTICES, [*VERTICES, (0.0, 0, [])], [[0, 1], [1, 2]], id="overlapping-images"),
	pytest.param(VERTICES, INTERVAL, [[0], [1]], id="mixed-relative-pair"),
	pytest.param([*VERTICES, (2.0, 1, [0, 1])], INTERVAL, [[0], [1], [2]], id="finite-kernel"),
	pytest.param(POINT, [*INTERVAL, (2.0, 1, [0, 1])], [[1]], id="cokernel-loop"),
	pytest.param(
		TRIANGLE_BOUNDARY,
		[*TRIANGLE_BOUNDARY, (2.0, 2, [3, 4, 5])],
		[[i] for i in range(6)],
		id="relative-two-cycle",
	),
]
MAPS = (
	("kernel_to_domain", "kernel", "domain"),
	("domain_to_image", "domain", "image"),
	("image_to_codomain", "image", "codomain"),
	("domain_to_codomain", "domain", "codomain"),
	("codomain_to_cokernel", "codomain", "cokernel"),
)


def rank(columns: list[set[int]]) -> int:
	"""Compute an independent F2 rank using sparse Gaussian elimination."""
	pivots: dict[int, set[int]] = {}
	for column in columns:
		residual = column.copy()
		while residual:
			pivot = max(residual)
			if pivot in pivots:
				residual ^= pivots[pivot]
			else:
				pivots[pivot] = residual
				break
	return len(pivots)


def apply(columns: list[list[int]], chain: list[int] | set[int]) -> set[int]:
	"""Apply an input matrix by parity, including cancellations between columns."""
	out: set[int] = set()
	for generator in chain:
		out.symmetric_difference_update(columns[generator])
	return out


def boundaries(matrix: Matrix, time: float, degree: int) -> list[set[int]]:
	"""Boundaries of the available generators of one degree."""
	return [set(column) for entrance, dim, column in matrix if entrance <= time and dim == degree]


def cycles(matrix: Matrix, time: float, degree: int) -> list[set[int]]:
	"""Construct the input kernel of d, without any package representatives."""
	pivots: dict[int, tuple[set[int], set[int]]] = {}
	out = []
	for generator, (entrance, dim, column) in enumerate(matrix):
		if entrance > time or dim != degree:
			continue
		residual, chain = set(column), {generator}
		while residual:
			pivot = max(residual)
			if pivot not in pivots:
				pivots[pivot] = residual, chain
				break
			boundary, earlier_chain = pivots[pivot]
			residual ^= boundary
			chain ^= earlier_chain
		else:
			out.append(chain)
	return out


def alive(diagram: dict[int, int | None], birth: int, time: float, times: list[float]) -> bool:
	"""Evaluate at a supplied time, after all its cylinder columns have entered."""
	death = diagram[birth]
	return times[birth] <= time and (death is None or time < times[death])


def cone_matrix(domain: Matrix, codomain: Matrix, mapping: list[list[int]]) -> Matrix:
	"""Build the original algebraic cone in B + shifted A coordinates for rank checks."""
	offset = len(codomain)
	return codomain + [
		(time, degree + 1, mapping[idx] + [offset + row for row in boundary])
		for idx, (time, degree, boundary) in enumerate(domain)
	]


def assert_chain_coordinates(
	result: DiagramEnsemble,
	metadata: CylinderMetadata,
	domain: Matrix,
	codomain: Matrix,
	mapping: list[list[int]],
) -> None:
	"""Support indices, degrees, availability, and differentials refer to the original inputs."""
	for name in ("domain", "kernel", "image", "codomain", "cokernel"):
		matrix = domain if name in {"domain", "kernel"} else codomain
		for birth, support in getattr(result, f"{name}_representatives").items():
			assert support == sorted(set(support))
			assert support  # Every retained interval has a nonzero homology class.
			degree = metadata.degrees[birth] - (name == "kernel")
			assert all(0 <= idx < len(matrix) for idx in support)
			assert all(matrix[idx][1] == degree for idx in support)
			assert all(matrix[idx][0] <= metadata.times[birth] for idx in support)
			assert not apply([col for _, _, col in matrix], support)
	for birth, (a, b) in result.relative_representatives.items():
		assert a == sorted(set(a))
		assert b == sorted(set(b))
		assert a or b
		assert all(0 <= idx < len(domain) for idx in a)
		assert all(0 <= idx < len(codomain) for idx in b)
		assert all(domain[idx][1] + 1 == metadata.degrees[birth] for idx in a)
		assert all(codomain[idx][1] == metadata.degrees[birth] for idx in b)
		assert all(domain[idx][0] <= metadata.times[birth] for idx in a)
		assert all(codomain[idx][0] <= metadata.times[birth] for idx in b)
		assert not apply([col for _, _, col in domain], a)
		assert apply([col for _, _, col in codomain], b) == apply(mapping, a)


def assert_composites(result: DiagramEnsemble, active: dict[str, list[int]]) -> None:
	"""Compose only active interval coordinates, checking exact-sequence composites."""

	def compose(first: str, second: str, middle: str, target: str, birth: int) -> set[int]:
		out: set[int] = set()
		for mid in set(getattr(result, first)[birth]) & set(active[middle]):
			out ^= set(getattr(result, second)[mid]) & set(active[target])
		return out

	for birth in active["domain"]:
		assert compose("domain_to_image", "image_to_codomain", "image", "codomain", birth) == (
			set(result.domain_to_codomain[birth]) & set(active["codomain"])
		)
	for first, second, source, middle, target in (
		("kernel_to_domain", "domain_to_image", "kernel", "domain", "image"),
		("kernel_to_domain", "domain_to_codomain", "kernel", "domain", "codomain"),
		("image_to_codomain", "codomain_to_cokernel", "image", "codomain", "cokernel"),
		("domain_to_codomain", "codomain_to_cokernel", "domain", "codomain", "cokernel"),
	):
		for birth in active[source]:
			assert not compose(first, second, middle, target, birth)


def assert_at_time(
	result: DiagramEnsemble,
	metadata: CylinderMetadata,
	domain: Matrix,
	codomain: Matrix,
	mapping: list[list[int]],
	*,
	time: float,
	degree: int,
) -> None:
	"""Check bases, all map actions/ranks, and the cone against input-only rank oracles."""
	a_cycles = cycles(domain, time, degree)
	a_boundaries = boundaries(domain, time, degree + 1)
	b_boundaries = boundaries(codomain, time, degree + 1)
	image_cycles = [apply(mapping, chain) for chain in a_cycles]
	dim_a = len(a_cycles) - rank(a_boundaries)
	dim_b = len(cycles(codomain, time, degree)) - rank(b_boundaries)
	dim_image = rank(b_boundaries + image_cycles) - rank(b_boundaries)
	dimensions = {
		"domain": dim_a,
		"kernel": dim_a - dim_image,
		"image": dim_image,
		"codomain": dim_b,
		"cokernel": dim_b - dim_image,
	}
	relations = {
		"domain": a_boundaries,
		"kernel": a_boundaries,
		"image": b_boundaries,
		"codomain": b_boundaries,
		"cokernel": b_boundaries + image_cycles,
	}
	active = {
		name: [
			birth
			for birth in getattr(result, name)
			if metadata.degrees[birth] - (name == "kernel") == degree
			and alive(getattr(result, name), birth, time, metadata.times)
		]
		for name in dimensions
	}
	for name, expected in dimensions.items():
		supports = getattr(result, f"{name}_representatives")
		assert len(active[name]) == expected
		assert rank(relations[name] + [set(supports[birth]) for birth in active[name]]) == (
			rank(relations[name]) + expected
		)
		for birth, support in supports.items():
			death = getattr(result, name)[birth]
			if (
				metadata.degrees[birth] - (name == "kernel") == degree
				and death is not None
				and metadata.times[death] <= time
			):
				assert rank([*relations[name], set(support)]) == rank(relations[name])
	for map_name, source, target in MAPS:
		columns = []
		for birth in active[source]:
			coefficients = set(getattr(result, map_name)[birth]) & set(active[target])
			columns.append(coefficients)
			source_cycle = getattr(result, f"{source}_representatives")[birth]
			residual = (
				apply(mapping, source_cycle)
				if source == "domain" and target != "domain"
				else set(source_cycle)
			)
			for target_birth in coefficients:
				residual ^= set(getattr(result, f"{target}_representatives")[target_birth])
			assert rank(relations[target] + [residual]) == rank(relations[target])
		expected_rank = (
			dimensions["kernel"]
			if source == "kernel"
			else (dimensions["cokernel"] if target == "cokernel" else dim_image)
		)
		assert rank(columns) == expected_rank
	assert_composites(result, active)
	cone = cone_matrix(domain, codomain, mapping)
	cone_boundaries = boundaries(cone, time, degree + 1)
	cone_dimension = len(cycles(cone, time, degree)) - rank(cone_boundaries)
	active_relative = [
		birth
		for birth in result.relative
		if metadata.degrees[birth] == degree and alive(result.relative, birth, time, metadata.times)
	]
	pairs = [result.relative_representatives[birth] for birth in active_relative]
	vectors = [set(b) | {len(codomain) + idx for idx in a} for a, b in pairs]
	assert len(vectors) == cone_dimension
	assert rank(cone_boundaries + vectors) == rank(cone_boundaries) + cone_dimension
	for birth, (a, b) in result.relative_representatives.items():
		death = result.relative[birth]
		if (
			metadata.degrees[birth] == degree
			and death is not None
			and metadata.times[death] <= time
		):
			vector = set(b) | {len(codomain) + idx for idx in a}
			assert rank([*cone_boundaries, vector]) == rank(cone_boundaries)


@pytest.mark.parametrize("slow", [False, True])
@pytest.mark.parametrize("num_threads", [1, 2])
@pytest.mark.parametrize(("domain", "codomain", "mapping"), CASES)
def test_original_coordinate_bases_and_maps(
	domain: Matrix,
	codomain: Matrix,
	mapping: list[list[int]],
	*,
	slow: bool,
	num_threads: int,
) -> None:
	"""Converted interval bases and all morphisms match the original filtered chain map."""
	result, metadata = sixpack(
		domain,
		codomain,
		mapping,
		get_representatives=True,
		num_threads=num_threads,
		slow=slow,
	)
	assert_chain_coordinates(result, metadata, domain, codomain, mapping)
	for time in sorted({time for time, _, _ in domain + codomain}):
		for degree in range(max((dim for _, dim, _ in domain + codomain), default=0) + 2):
			assert_at_time(result, metadata, domain, codomain, mapping, time=time, degree=degree)


@pytest.mark.parametrize("slow", [False, True])
def test_relative_pair_retains_domain_information(*, slow: bool) -> None:
	"""The zero map's cone has a domain component even when the codomain is empty."""
	result, metadata = sixpack(POINT, [], [[]], get_representatives=True, num_threads=1, slow=slow)
	assert result.relative_representatives == {metadata.domain_shift[0]: ([0], [])}


@pytest.mark.parametrize("slow", [False, True])
def test_relative_pair_with_noncycle_codomain_chain(*, slow: bool) -> None:
	"""The relative edge pairs its two domain endpoints with a noncycle codomain edge."""
	result, _ = sixpack(
		VERTICES, INTERVAL, [[0], [1]], get_representatives=True, num_threads=1, slow=slow
	)
	assert list(result.relative_representatives.values()) == [([0, 1], [2])]
