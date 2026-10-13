"""Stored morphisms agree with chain representatives and are pointwise exact."""

import pytest
from phimaker import DiagramEnsemble, sixpack, sixpack_from_inclusion

MAPS = (
	("kernel_to_domain", "kernel", "domain"),
	("domain_to_image", "domain", "image"),
	("image_to_codomain", "image", "codomain"),
	("domain_to_codomain", "domain", "codomain"),
	("codomain_to_cokernel", "codomain", "cokernel"),
)


def rank(columns: list[set[int]]) -> int:
	"""Use independent sparse F2 elimination for small chain and map matrices."""
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


def alive(diagram: dict[int, int | None], birth: int, index: int) -> bool:
	"""Evaluate half-open intervals in the common ambient filtration index scale."""
	death = diagram[birth]
	return birth <= index and (death is None or index < death)


def assert_columns(
	result: DiagramEnsemble, matrix: list[list[int]], degrees: list[int], domain: list[int]
) -> None:
	"""Each coefficient column is the identity chain map modulo the correct relations."""
	for name, source_name, target_name in MAPS:
		mapping = getattr(result, name)
		source = getattr(result, source_name)
		target = getattr(result, target_name)
		assert mapping.keys() == source.keys()
		for birth, coefficients in mapping.items():
			assert coefficients == sorted(set(coefficients))
			residual = set(getattr(result, f"{source_name}_representatives")[birth])
			for target_birth in coefficients:
				assert alive(target, target_birth, birth)
				assert source[birth] is None or (
					target[target_birth] is not None and target[target_birth] <= source[birth]
				)
				assert degrees[target_birth] == degrees[birth] - (source_name == "kernel")
				residual.symmetric_difference_update(
					getattr(result, f"{target_name}_representatives")[target_birth]
				)
			relations = [
				set(column)
				for idx, column in enumerate(matrix)
				if idx <= birth and (target_name != "domain" or idx in domain)
			]
			if target_name == "cokernel":
				relations.extend(
					set(support)
					for idx, support in result.domain_representatives.items()
					if idx <= birth
				)
			assert rank([*relations, residual]) == rank(relations)


def composite(
	result: DiagramEnsemble,
	first: str,
	second: str,
	middle: str,
	target: str,
	*,
	birth: int,
	index: int,
) -> set[int]:
	"""Compose at an index using only alive middle and target features."""
	out: set[int] = set()
	for mid in getattr(result, first)[birth]:
		if alive(getattr(result, middle), mid, index):
			out.symmetric_difference_update(
				end
				for end in getattr(result, second)[mid]
				if alive(getattr(result, target), end, index)
			)
	return out


def assert_composites(result: DiagramEnsemble, index: int) -> None:
	"""Check factorization through image and the zero compositions of exact sequences."""
	for birth in result.domain:
		if alive(result.domain, birth, index):
			direct = {
				end
				for end in result.domain_to_codomain[birth]
				if alive(result.codomain, end, index)
			}
			assert (
				composite(
					result,
					"domain_to_image",
					"image_to_codomain",
					"image",
					"codomain",
					birth=birth,
					index=index,
				)
				== direct
			)
	for first, second, source, middle, target in (
		("kernel_to_domain", "domain_to_image", "kernel", "domain", "image"),
		("kernel_to_domain", "domain_to_codomain", "kernel", "domain", "codomain"),
		("image_to_codomain", "codomain_to_cokernel", "image", "codomain", "cokernel"),
		("domain_to_codomain", "codomain_to_cokernel", "domain", "codomain", "cokernel"),
	):
		for birth in getattr(result, source):
			if alive(getattr(result, source), birth, index):
				assert not composite(
					result, first, second, middle, target, birth=birth, index=index
				)


def assert_stored_maps(
	result: DiagramEnsemble, matrix: list[list[int]], degrees: list[int], domain: list[int]
) -> None:
	"""Check chains, interval compatibility, pointwise ranks, and exactness independently."""
	assert_columns(result, matrix, degrees, domain)
	for index in range(len(matrix) + 1):
		dimensions = {
			name: sum(alive(getattr(result, name), birth, index) for birth in getattr(result, name))
			for name in ("kernel", "domain", "image", "codomain", "cokernel")
		}
		for name, source, target in MAPS:
			columns = [
				{
					end
					for end in getattr(result, name)[birth]
					if alive(getattr(result, target), end, index)
				}
				for birth in getattr(result, source)
				if alive(getattr(result, source), birth, index)
			]
			expected = (
				dimensions["kernel"]
				if source == "kernel"
				else (dimensions["cokernel"] if target == "cokernel" else dimensions["image"])
			)
			assert rank(columns) == expected
		assert dimensions["domain"] == dimensions["kernel"] + dimensions["image"]
		assert dimensions["codomain"] == dimensions["image"] + dimensions["cokernel"]
		assert_composites(result, index)


@pytest.mark.parametrize("slow", [False, True])
@pytest.mark.parametrize(
	("matrix", "degrees", "domain"),
	[
		([], [], []),
		([[]], [0], []),
		([[], [], [0, 1]], [0, 0, 1], [2, 0, 1]),
		([[], [], [0, 1]], [0, 0, 1], [1]),
		([[], [], [], [0, 1], [0, 2], [1, 2]], [0, 0, 0, 1, 1, 1], [5, 2, 1]),
		(
			[[], [], [], [], [1, 2], [2, 3], [1, 3], [2, 3]],
			[0, 0, 0, 0, 1, 1, 1, 1],
			[7, 3, 1, 6, 2],
		),
		([[], [], [0, 1], [], [0, 3], [1, 3]], [0, 0, 1, 0, 1, 1], [5, 3, 1]),
		(
			[[], [], [], [0, 1], [1, 2], [0, 2], [3, 4, 5]],
			[0, 0, 0, 1, 1, 1, 2],
			[0, 1, 2, 3, 4, 5],
		),
		([[], [], [0, 1], [0, 1]], [0, 0, 1, 1], [0, 1, 3]),
		([[], [], [0, 1], [0, 1], [2, 3]], [0, 0, 1, 1, 2], [0, 1, 3]),
	],
	ids=[
		"empty",
		"empty-domain",
		"identity",
		"quotient",
		"finite-kernel",
		"image-sums",
		"dead-targets",
		"essential-kernel",
		"essential-cokernel",
		"finite-cokernel",
	],
)
def test_stored_maps_are_correct(
	matrix: list[list[int]], degrees: list[int], domain: list[int], *, slow: bool
) -> None:
	"""Stored maps have the correct chain action, ranks, and composites in both modes."""
	result = sixpack_from_inclusion(
		matrix, degrees, domain, get_representatives=True, num_threads=1, slow=slow
	)
	assert_stored_maps(result, matrix, degrees, domain)


@pytest.mark.parametrize("slow", [False, True])
def test_maps_have_hand_derived_coefficients(*, slow: bool) -> None:
	"""A younger domain vertex maps to a sum; both codomain features project to the quotient."""
	result = sixpack_from_inclusion(
		[[], [], [0, 1]], [0, 0, 1], [1], get_representatives=True, num_threads=1, slow=slow
	)
	assert result.kernel_to_domain == {}
	assert result.domain_to_image == {1: [1]}
	assert result.image_to_codomain == {1: [0, 1]}
	assert result.domain_to_codomain == {1: [0, 1]}
	assert result.codomain_to_cokernel == {0: [0], 1: [0]}
	identity = sixpack_from_inclusion(
		[[], [], [0, 1]], [0, 0, 1], [0, 1, 2], get_representatives=True, num_threads=1, slow=slow
	)
	assert identity.codomain_to_cokernel == {0: [], 1: []}


@pytest.mark.parametrize("slow", [False, True])
def test_factorization_requires_interval_lifetimes(*, slow: bool) -> None:
	"""Ordinary table multiplication retains a spurious dead target in an actual inclusion."""
	matrix = [[], [], [0], [], [0, 3], [1, 3]]
	degrees = [0, 0, 1, 0, 1, 1]
	domain = [1, 3, 5]
	result = sixpack_from_inclusion(
		matrix, degrees, domain, get_representatives=True, num_threads=1, slow=slow
	)
	assert result.domain_to_image[3] == [1, 3]
	assert result.image_to_codomain[1] == [0, 1]
	assert result.image_to_codomain[3] == [3]
	assert result.domain_to_codomain[3] == [1, 3]
	assert result.codomain[0] == 2
	naive: set[int] = set()
	for mid in result.domain_to_image[3]:
		naive.symmetric_difference_update(result.image_to_codomain[mid])
	assert naive == {0, 1, 3}
	assert_stored_maps(result, matrix, degrees, domain)


@pytest.mark.parametrize("slow", [False, True])
@pytest.mark.parametrize("mapping", [[[0]], [[]]], ids=["identity", "zero"])
def test_maps_in_mapping_cylinders(mapping: list[list[int]], *, slow: bool) -> None:
	"""All maps use cylinder coordinates and recover identity/zero after equal-time cells enter."""
	point = [(0.0, 0, [])]
	result, metadata = sixpack(
		point, point, mapping, get_representatives=True, num_threads=1, slow=slow
	)
	birth = metadata.domain_indices[0]
	codomain_birth = metadata.codomain_indices[0]
	shift_boundary = [birth, *([codomain_birth] if mapping[0] else [])]
	matrix = [[], [], shift_boundary]
	assert_stored_maps(result, matrix, metadata.degrees, metadata.domain_indices)
	index = len(metadata.times) - 1
	assert bool(
		[end for end in result.domain_to_codomain[birth] if alive(result.codomain, end, index)]
	) == bool(mapping[0])
