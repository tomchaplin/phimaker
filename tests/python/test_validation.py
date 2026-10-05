"""Python bindings convert structural Rust errors to ValueError in both modes."""

import struct

import pytest
from phimaker import sixpack, sixpack_from_inclusion


@pytest.mark.parametrize("slow", [False, True])
@pytest.mark.parametrize(
	("matrix", "degrees", "domain", "message"),
	[
		([[]], [], [], "`degrees`"),
		([[]], [0, 1], [], "`degrees`"),
		([[]], [0], [1], "domain column index 1"),
		([[]], [0], [2], "domain column index 2"),
		([], [], [0], "domain column index 0"),
		([[0]], [0], [], "strictly upper-triangular"),
		([[], [9]], [0, 1], [], "strictly upper-triangular"),
		([[], [0]], [0, 0], [], "lower degree"),
		([[], [0]], [(1 << (8 * struct.calcsize("P"))) - 1, 0], [], "lower degree"),
		([[], [0]], [0, 1], [1], "closed under taking boundaries"),
	],
	ids=[
		"short-degrees",
		"long-degrees",
		"index-at-size",
		"index-above-size",
		"empty-domain-range",
		"diagonal",
		"row-out-of-range",
		"wrong-degree",
		"degree-overflow",
		"domain-not-closed",
	],
)
def test_validation_errors(
	matrix: list[list[int]],
	degrees: list[int],
	domain: list[int],
	message: str,
	*,
	slow: bool,
) -> None:
	"""Invalid structure fails before reduction or slow-mode serialization."""
	with pytest.raises(ValueError, match=message):
		sixpack_from_inclusion(matrix, degrees, domain, num_threads=1, slow=slow)


@pytest.mark.parametrize("slow", [False, True])
def test_normalized_supports_and_domain(*, slow: bool) -> None:
	"""Repeated row listings retain the nonzero entry; input lists are not mutated."""
	matrix = [[], [], [1, 0, 1]]
	domain = [1, 1]
	result = sixpack_from_inclusion(matrix, [0, 0, 1], domain, num_threads=1, slow=slow)
	expected = {
		"domain": {1: None},
		"codomain": {0: None, 1: 2},
		"image": {1: None},
		"kernel": {},
		"cokernel": {0: 2},
		"relative": {0: 2},
	}
	for name, diagram in expected.items():
		assert getattr(result, name) == diagram, name
	assert matrix == [[], [], [1, 0, 1]]
	assert domain == [1, 1]


@pytest.mark.parametrize("slow", [False, True])
def test_general_map_validation_error(*, slow: bool) -> None:
	"""The general-map binding also converts cylinder-inclusion errors to ValueError."""
	malformed = [(0.0, 0, []), (1.0, 0, [0])]
	with pytest.raises(ValueError, match="lower degree"):
		sixpack(malformed, malformed, [[0], [1]], num_threads=1, slow=slow)


def test_public_python_function_names() -> None:
	"""Internal wrapper names do not change the established Python API."""
	assert sixpack.__name__ == "sixpack"
	assert sixpack_from_inclusion.__name__ == "sixpack_from_inclusion"
