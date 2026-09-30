"""Ensure PyO3 publishes the shared API documentation and calling signatures."""

from phimaker import sixpack, sixpack_from_inclusion


def test_binding_documentation() -> None:
	"""Both entry points expose usage, assumptions, return types, and default arguments."""
	for function in (sixpack, sixpack_from_inclusion):
		for section in ("Parameters", "Returns", "Assumptions"):
			assert section in function.__doc__
		assert "num_threads=0" in function.__text_signature__
		assert "slow=False" in function.__text_signature__
	result = sixpack_from_inclusion([[]], [0], [0], num_threads=1)
	for name in ("domain", "codomain", "image", "kernel", "cokernel", "relative"):
		assert getattr(type(result), name).__doc__
