"""Check slow-mode empty decompositions against normal-mode diagrams."""

import pytest
from phimaker import sixpack_from_inclusion


@pytest.mark.parametrize("domain", [[], [0]], ids=["empty-domain", "empty-relative"])
def test_slow_empty_decomposition(domain: list[int]) -> None:
	"""LoPhat handles empty domain and quotient serialization without a panic."""
	result = sixpack_from_inclusion([[]], [0], domain, num_threads=1, slow=True)
	expected = sixpack_from_inclusion([[]], [0], domain, 1)
	for name in ("domain", "codomain", "image", "kernel", "cokernel", "relative"):
		assert getattr(result, name) == getattr(expected, name)
