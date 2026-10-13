"""Check language-specific runtime docs and execute the documented examples."""

import doctest
from typing import Any

import phimaker
import phimaker.phimaker as extension
import pytest
from phimaker import CylinderMetadata, DiagramEnsemble, sixpack, sixpack_from_inclusion


def test_result_class_exports() -> None:
	"""Declared class exports match returned objects and make typed results importable."""
	result, metadata = sixpack([(0.0, 0, [])], [(0.0, 0, [])], [[0]])
	assert type(result) is DiagramEnsemble
	assert type(metadata) is CylinderMetadata
	assert DiagramEnsemble.__module__ == "phimaker"
	assert CylinderMetadata.__module__ == "phimaker"


def test_binding_documentation() -> None:
	"""Functions publish reStructuredText usage docs and their existing signatures."""
	for function in (sixpack, sixpack_from_inclusion):
		documentation = function.__doc__
		assert ":param " in documentation
		assert ":returns:" in documentation
		assert "Example::" in documentation
		assert "num_threads=0" in function.__text_signature__
		assert "slow=False" in function.__text_signature__
		assert "get_representatives=False" in function.__text_signature__
		assert ":param get_representatives:" in documentation
		assert "Rust" not in documentation
		assert "PyResult" not in documentation
		assert "LoPhat" not in documentation
		assert "#[" not in documentation


def test_shared_class_and_property_documentation() -> None:
	"""Rust comments are stripped recursively from Python classes and their fields."""
	result, metadata = sixpack([(0.0, 0, [])], [(0.0, 0, [])], [[0]])
	for value in (result, metadata):
		documentation = type(value).__doc__
		assert "Example::" in documentation
		assert "# Example" not in documentation
		assert "ExtendedUsize" not in documentation
		assert "HashMap" not in documentation
	for name in ("domain", "codomain", "image", "kernel", "cokernel", "relative"):
		documentation = getattr(type(result), name).__doc__
		assert "dictionary" in documentation
		assert "[`" not in documentation
	for name in (
		"domain_representatives",
		"codomain_representatives",
		"image_representatives",
		"kernel_representatives",
		"cokernel_representatives",
	):
		documentation = getattr(type(result), name).__doc__
		assert "dictionary" in documentation
		assert "empty dictionary" in documentation
		assert "None" in documentation
		assert "HashMap" not in documentation
		assert "V matrix" not in documentation
	for name in (
		"kernel_to_domain",
		"domain_to_image",
		"image_to_codomain",
		"domain_to_codomain",
		"codomain_to_cokernel",
	):
		documentation = getattr(type(result), name).__doc__
		assert "dictionary" in documentation
		assert "empty list denotes zero" in documentation
		assert "None" in documentation
		assert "intervals alive" in documentation
		assert "HashMap" not in documentation
	assert "maximum index" in type(result).__doc__
	assert "Index this list" in type(metadata).times.__doc__
	assert "in input order" in type(metadata).domain_indices.__doc__
	assert "in input order" in type(metadata).codomain_indices.__doc__
	assert "Original domain" not in type(metadata).domain_indices.__doc__


@pytest.mark.parametrize(
	"name",
	["sixpack", "sixpack_from_inclusion", "diagrams", "metadata", "cokernel_representatives"],
)
def test_python_documentation_examples(name: str) -> None:
	"""Every function and shared-class example is valid and has the stated result."""
	result, metadata = sixpack([(0.0, 0, [])], [(0.0, 0, [])], [[0]])
	objects: dict[str, Any] = {
		"sixpack": sixpack,
		"sixpack_from_inclusion": sixpack_from_inclusion,
		"diagrams": type(result),
		"metadata": type(metadata),
		"cokernel_representatives": type(result).cokernel_representatives,
	}
	test = doctest.DocTestParser().get_doctest(objects[name].__doc__, {}, name, None, 0)
	assert test.examples, "Usage documentation must include executable examples"
	outcome = doctest.DocTestRunner().run(test)
	assert outcome.failed == 0


def test_standalone_persistence_map_is_not_exported() -> None:
	"""Both the package and native extension expose only chain-complex computations."""
	assert "get_persistence_map" not in phimaker.__all__
	assert not hasattr(phimaker, "get_persistence_map")
	assert not hasattr(extension, "get_persistence_map")
