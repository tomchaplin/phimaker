"""Persistence morphisms are sparse binary columns in the chosen interval bases."""

import copy

import pytest
from phimaker import DiagramEnsemble, get_persistence_map, sixpack, sixpack_from_inclusion


def ensemble_maps(result: DiagramEnsemble) -> tuple[dict[int, list[int]], dict[int, list[int]]]:
	"""Exercise both inclusion targets through the standalone function."""
	assert result.domain_representatives is not None
	assert result.codomain_representatives is not None
	assert result.image_representatives is not None
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


def assert_image_inclusion(
	result: DiagramEnsemble,
	n_columns: int,
	expected: dict[int, list[int]],
) -> None:
	"""Check exact coefficients, pointwise injectivity, and pointwise factorization."""
	assert result.image_representatives is not None
	assert result.codomain_representatives is not None
	identity = {
		index: [index] for support in result.image_representatives.values() for index in support
	}
	image_to_codomain = get_persistence_map(
		result.image,
		result.image_representatives,
		result.codomain,
		result.codomain_representatives,
		identity,
	)
	assert image_to_codomain == expected
	domain_to_codomain, domain_to_image = ensemble_maps(result)
	for index in range(n_columns + 1):
		active_image = [
			birth for birth in result.image if interval_alive(result.image, birth, index)
		]
		columns = [
			{
				target
				for target in image_to_codomain[birth]
				if interval_alive(result.codomain, target, index)
			}
			for birth in active_image
		]
		assert binary_rank(columns) == len(active_image), f"Image inclusion at {index}"
		for birth in result.domain:
			if not interval_alive(result.domain, birth, index):
				continue
			composed: set[int] = set()
			for image_birth in domain_to_image[birth]:
				if interval_alive(result.image, image_birth, index):
					composed.symmetric_difference_update(
						target
						for target in image_to_codomain[image_birth]
						if interval_alive(result.codomain, target, index)
					)
			direct = {
				target
				for target in domain_to_codomain[birth]
				if interval_alive(result.codomain, target, index)
			}
			assert composed == direct, f"Composition for birth {birth} at {index}"


@pytest.mark.parametrize("slow", [False, True])
@pytest.mark.parametrize(
	("matrix", "degrees", "domain", "expected"),
	[
		(
			[[], [], [], [0, 1], [0, 2], [1, 2]],
			[0, 0, 0, 1, 1, 1],
			[5, 2, 1],
			{1: [0, 1], 2: [1, 2]},
		),
		(
			[[], [], [], [], [1, 2], [2, 3], [1, 3], [2, 3]],
			[0, 0, 0, 0, 1, 1, 1, 1],
			[7, 3, 1, 6, 2],
			{1: [1], 2: [2], 3: [3]},
		),
		(
			[[], [], [0, 1], [], [0, 3], [1, 3]],
			[0, 0, 1, 0, 1, 1],
			[5, 3, 1],
			{1: [0, 1], 3: [3]},
		),
		([[]], [0], [], {}),
	],
	ids=["codomain-sums", "different-image-basis", "dead-targets", "empty-image"],
)
def test_image_to_codomain_inclusion(
	matrix: list[list[int]],
	degrees: list[int],
	domain: list[int],
	expected: dict[int, list[int]],
	*,
	slow: bool,
) -> None:
	"""Image columns are independent, and the two maps factor the original inclusion."""
	result = sixpack_from_inclusion(
		matrix, degrees, domain, get_representatives=True, num_threads=1, slow=slow
	)
	assert_image_inclusion(result, len(matrix), expected)


@pytest.mark.parametrize("slow", [False, True])
@pytest.mark.parametrize("mapping", [[[0]], [[]]], ids=["identity", "zero"])
def test_image_to_codomain_cylinder_inclusion(mapping: list[list[int]], *, slow: bool) -> None:
	"""Check both general maps at every cylinder index, including equal-time artifacts."""
	point = [(0.0, 0, [])]
	result, metadata = sixpack(
		point, point, mapping, get_representatives=True, num_threads=1, slow=slow
	)
	birth = metadata.domain_indices[0]
	assert_image_inclusion(result, len(metadata.times), {birth: [birth]})


def test_interval_composition_discards_dead_final_target() -> None:
	"""Nonzero maps compose to zero when the outer intervals do not overlap."""
	source, middle, target = {2: 4}, {1: 3}, {0: 2}
	source_reps, middle_reps, target_reps = {2: [2]}, {1: [1]}, {0: [0]}
	first = get_persistence_map(source, source_reps, middle, middle_reps, {2: [1]})
	second = get_persistence_map(middle, middle_reps, target, target_reps, {1: [0]})
	direct = get_persistence_map(source, source_reps, target, target_reps, {2: [0]})
	assert first == {2: [1]}
	assert second == {1: [0]}
	assert direct == {2: []}
	for index in range(5):
		if not interval_alive(source, 2, index):
			continue
		composed: set[int] = set()
		for mid in first[2]:
			if interval_alive(middle, mid, index):
				composed.symmetric_difference_update(
					end for end in second[mid] if interval_alive(target, end, index)
				)
		assert composed == set(direct[2])


def assert_kernel_cycles(
	result: DiagramEnsemble,
	matrix: list[list[int]],
	degrees: list[int],
	domain: list[int],
) -> None:
	"""Verify cycles, independence modulo domain boundaries, and the zero composite."""
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
	identity = {idx: [idx] for support in representatives.values() for idx in support}
	kernel_to_domain = get_persistence_map(
		result.kernel, representatives, result.domain, result.domain_representatives, identity
	)
	domain_to_codomain, _ = ensemble_maps(result)
	for index in range(len(matrix) + 1):
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
		inclusion_columns = []
		for birth in active:
			column = {
				idx for idx in kernel_to_domain[birth] if interval_alive(result.domain, idx, index)
			}
			composed: set[int] = set()
			for domain_birth in column:
				composed.symmetric_difference_update(
					target
					for target in domain_to_codomain[domain_birth]
					if interval_alive(result.codomain, target, index)
				)
			assert not composed
			inclusion_columns.append(column)
		assert binary_rank(inclusion_columns) == len(active)


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
def test_kernel_representatives_and_domain_inclusion(
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
	assert_kernel_cycles(result, matrix, metadata.degrees, metadata.domain_indices)


@pytest.mark.parametrize("slow", [False, True])
def test_sums_of_codomain_features(*, slow: bool) -> None:
	"""A domain interval has two nonzero codomain coefficients and one image coefficient."""
	result = sixpack_from_inclusion(
		[[], [], [], [0, 1], [0, 2], [1, 2]],
		[0, 0, 0, 1, 1, 1],
		[5, 2, 1],
		num_threads=1,
		slow=slow,
		get_representatives=True,
	)
	assert ensemble_maps(result) == ({1: [0, 1], 2: [1, 2]}, {1: [1], 2: [2]})


@pytest.mark.parametrize("slow", [False, True])
def test_sums_of_image_features(*, slow: bool) -> None:
	"""Different death reductions make a domain feature a sum of image features."""
	result = sixpack_from_inclusion(
		[[], [], [], [], [1, 2], [2, 3], [1, 3], [2, 3]],
		[0, 0, 0, 0, 1, 1, 1, 1],
		[7, 3, 1, 6, 2],
		num_threads=1,
		slow=slow,
		get_representatives=True,
	)
	expected = {1: [1], 2: [2], 3: [2, 3]}
	assert ensemble_maps(result) == (expected, expected)


@pytest.mark.parametrize("slow", [False, True])
def test_dead_targets_are_removed_after_solving(*, slow: bool) -> None:
	"""Exact chain expansion needs a dead cycle, which contributes zero in homology."""
	result = sixpack_from_inclusion(
		[[], [], [0, 1], [], [0, 3], [1, 3]],
		[0, 0, 1, 0, 1, 1],
		[5, 3, 1],
		num_threads=1,
		slow=slow,
		get_representatives=True,
	)
	assert ensemble_maps(result) == ({1: [0, 1], 3: [3]}, {1: [1], 3: [3]})
	# After target 3 dies, source 3 remains alive but has zero image.
	assert result.domain[3] == 5
	assert result.codomain[3] == result.image[3] == 4


@pytest.mark.parametrize("slow", [False, True])
def test_identity_and_empty_maps(*, slow: bool) -> None:
	"""Identity has diagonal columns; empty domains need no representatives."""
	result = sixpack_from_inclusion(
		[[], [], [0, 1]], [0, 0, 1], [2, 0, 1], num_threads=1, slow=slow, get_representatives=True
	)
	expected = {0: [0], 1: [1]}
	assert ensemble_maps(result) == (expected, expected)
	for matrix, degrees in (([], []), ([[]], [0])):
		empty = sixpack_from_inclusion(
			matrix, degrees, [], num_threads=1, slow=slow, get_representatives=True
		)
		assert ensemble_maps(empty) == ({}, {})


@pytest.mark.parametrize("slow", [False, True])
def test_zero_general_map_in_original_filtration(*, slow: bool) -> None:
	"""Equal-time cylinder artifacts vanish after all cells at that time enter."""
	result, metadata = sixpack(
		[(0.0, 0, [])], [(0.0, 0, [])], [[]], num_threads=1, slow=slow, get_representatives=True
	)
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
	result = sixpack_from_inclusion(
		[[], [], [0, 1]], [0, 0, 1], [1], num_threads=1, get_representatives=True
	)
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
		({4: 5}, {4: [1, 2]}, {1: None, 2: 5}, {1: [1], 2: [1, 2]}, {1: [1], 2: [2]}, {4: [2]}),
		({5: 9}, {5: [0, 1]}, {3: 7}, {3: [0, 1]}, {0: [0], 1: [1]}, {5: [3]}),
		(
			{5: 9},
			{5: [0]},
			{1: 8, 3: 7},
			{1: [0, 1], 3: [1]},
			{0: [0]},
			{5: [1, 3]},
		),
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
		({1: None}, {1: [2]}, {}, {}, {}),
		({1: None}, {1: []}, {}, {}, {}),
		({1: 1}, {1: [1]}, {}, {}, {}),
		({1: None}, {1: [1]}, {0: None}, {}, {}),
		({1: None}, {1: [1]}, {0: None}, {0: [0]}, {1: [1]}),
		({1: None}, {1: [1]}, {5: None}, {5: [5]}, {1: [5]}),
		({1: 3}, {1: [1]}, {0: 4}, {0: [0]}, {1: [0]}),
		({3: None}, {3: [0]}, {0: None, 2: None}, {0: [0], 2: [0]}, {0: [0]}),
		({2: None}, {2: [1]}, {1: 8, 3: 7}, {1: [0, 1], 3: [1]}, {1: [1]}),
		({3: None}, {3: [0]}, {1: None}, {1: [2]}, {0: [2]}),
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


@pytest.mark.parametrize("argument", ["source", "target"])
def test_persistence_map_requires_representative_dictionaries(argument: str) -> None:
	"""Passing a skipped representative attribute is a type error, rather than a zero map."""
	source_reps = None if argument == "source" else {0: [0]}
	target_reps = None if argument == "target" else {0: [0]}
	with pytest.raises(TypeError):
		get_persistence_map({0: None}, source_reps, {0: None}, target_reps, {0: [0]})
