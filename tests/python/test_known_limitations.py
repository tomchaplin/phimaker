"""Track the upstream LoPhat 0.11.0 empty-decomposition serializer bug narrowly."""

import pytest
from phimaker import sixpack_from_inclusion


class EmptySerializationError(Exception):
	"""Only the known empty-column access panic is an expected failure."""


@pytest.mark.parametrize("domain", [[], [0]], ids=["empty-domain", "empty-relative"])
@pytest.mark.xfail(
	raises=EmptySerializationError,
	strict=True,
	reason="LoPhat 0.11.0 serializer accesses column zero of empty decompositions",
)
def test_slow_empty_decomposition(domain: list[int]) -> None:
	"""Expect parity with normal mode once the upstream serializer is repaired."""
	try:
		result = sixpack_from_inclusion([[]], [0], domain, num_threads=1, slow=True)
	except BaseException as error:
		if (
			type(error).__name__ == "PanicException"
			and str(error) == "index out of bounds: the len is 0 but the index is 0"
		):
			raise EmptySerializationError from error
		raise
	expected = sixpack_from_inclusion([[]], [0], domain, 1)
	for name in ("domain", "codomain", "image", "kernel", "cokernel", "relative"):
		assert getattr(result, name) == getattr(expected, name)
