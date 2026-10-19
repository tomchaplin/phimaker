"""General-map results omit equal-time artifacts after the full map solves."""

import pytest
from phimaker import sixpack

NAMES = ("domain", "codomain", "image", "kernel", "cokernel", "relative")
MAPS = (
	("kernel_to_domain", "kernel", "domain"),
	("domain_to_image", "domain", "image"),
	("image_to_codomain", "image", "codomain"),
	("domain_to_codomain", "domain", "codomain"),
	("codomain_to_cokernel", "codomain", "cokernel"),
)


@pytest.mark.parametrize("slow", [False, True])
@pytest.mark.parametrize("get_representatives", [False, True])
@pytest.mark.parametrize("nonzero", [False, True])
def test_equal_time_results(*, slow: bool, get_representatives: bool, nonzero: bool) -> None:
	"""Prune all six diagrams, preserve zero columns, and keep extraction optional."""
	point = [(0.0, 0, [])]
	result, metadata = sixpack(
		point,
		point,
		[[0] if nonzero else []],
		get_representatives=get_representatives,
		num_threads=1,
		slow=slow,
	)
	assert metadata.times == [0.0] * 3
	expected = {
		"domain": {0: None},
		"codomain": {0 if nonzero else 1: None},
		"image": {0: None} if nonzero else {},
		"kernel": {} if nonzero else {2: None},
		"cokernel": {} if nonzero else {1: None},
		"relative": {} if nonzero else {1: None, 2: None},
	}
	for name in NAMES:
		assert getattr(result, name) == expected[name]
		representatives = getattr(result, f"{name}_representatives")
		if get_representatives:
			assert representatives.keys() == expected[name].keys()
		else:
			assert representatives is None
	for name, source, target in MAPS:
		mapping = getattr(result, name)
		if get_representatives:
			assert mapping.keys() == expected[source].keys()
			assert all(set(column) <= expected[target].keys() for column in mapping.values())
		else:
			assert mapping is None
	if get_representatives:
		assert result.domain_to_codomain == {0: [0] if nonzero else []}
		assert result.domain_to_image == {0: [0] if nonzero else []}


@pytest.mark.parametrize("slow", [False, True])
@pytest.mark.parametrize("get_representatives", [False, True])
@pytest.mark.parametrize("death", [1.0, 1e-12])
def test_positive_duration_is_retained(
	death: float, *, slow: bool, get_representatives: bool
) -> None:
	"""Tied vertices can give a short but nonzero interval that must survive."""
	result, metadata = sixpack(
		[(0.0, 0, [])],
		[(0.0, 0, []), (0.0, 0, []), (death, 1, [0, 1])],
		[[1]],
		get_representatives=get_representatives,
		num_threads=1,
		slow=slow,
	)
	for name in NAMES:
		assert all(
			end is None or metadata.times[birth] < metadata.times[end]
			for birth, end in getattr(result, name).items()
		)
	for name in ("codomain", "cokernel", "relative"):
		assert [
			(metadata.times[birth], metadata.times[end])
			for birth, end in getattr(result, name).items()
			if end is not None
		] == [(0.0, death)]
