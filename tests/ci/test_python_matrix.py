"""Check interpreter selection without network access or an installed extension."""

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / ".github/scripts/python_matrix.py"
SPEC = importlib.util.spec_from_file_location("python_matrix", SCRIPT)
assert SPEC is not None
assert SPEC.loader is not None
matrix_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(matrix_module)


def downloads(version: str, variant: str = "default") -> list[dict[str, str]]:
	"""Supply a synthetic uv catalogue entry for each native platform."""
	return [
		{
			"implementation": "cpython",
			"version": version,
			"variant": variant,
			"os": platform["os"],
			"arch": platform["arch"],
			"libc": platform["libc"],
		}
		for platform in matrix_module.PLATFORMS
	]


def test_latest_stable_patch_and_supported_variants() -> None:
	"""Ignore prereleases, unsupported minors and experimental 3.13t interpreters."""
	catalogue = (
		downloads("3.11.15")
		+ downloads("3.12.13")
		+ downloads("3.13.1")
		+ downloads("3.13.2")
		+ downloads("3.13.2", "freethreaded")
		+ downloads("3.14.0")
		+ downloads("3.14.0", "freethreaded")
		+ downloads("3.15.0rc2")
	)
	entries = matrix_module.make_matrix(catalogue, ">=3.12,<3.15")["include"]
	assert len(entries) == 4 * len(matrix_module.PLATFORMS)
	assert {entry["label"] for entry in entries} == {"3.12", "3.13", "3.14", "3.14t"}
	assert {entry["python"] for entry in entries} == {"3.12.13", "3.13.2", "3.14.0", "3.14.0t"}
	linux_t = next(
		entry for entry in entries if entry["name"] == "linux-x64" and entry["label"] == "3.14t"
	)
	assert linux_t["interpreter"] == "/opt/python/cp314-cp314t/bin/python"


@pytest.mark.parametrize("variant", ["default", "freethreaded"])
def test_missing_interpreter_is_an_error(variant: str) -> None:
	"""Every selected interpreter must exist on every platform, including its ABI variant."""
	available_variant = [
		entry
		for entry in downloads("3.14.0", variant)
		if (entry["os"], entry["arch"]) != ("windows", "aarch64")
	]
	other_variant = "freethreaded" if variant == "default" else "default"
	catalogue = downloads("3.12.13") + available_variant + downloads("3.14.0", other_variant)
	label = "3.14t" if variant == "freethreaded" else "3.14"
	with pytest.raises(
		ValueError, match=rf"No compatible Python {label} download for windows-arm64"
	):
		matrix_module.make_matrix(catalogue, ">=3.12")


def test_missing_platform_is_an_error() -> None:
	"""An entire missing platform must fail CI instead of silently shrinking coverage."""
	with pytest.raises(ValueError, match=r"No compatible Python 3\.14 download for macos-arm64"):
		matrix_module.make_matrix(downloads("3.14.0")[:-1], ">=3.12")


def test_no_matching_stable_release() -> None:
	"""An unsatisfiable Python requirement cannot produce an empty successful matrix."""
	with pytest.raises(ValueError, match="No stable CPython releases"):
		matrix_module.make_matrix(downloads("3.14.0"), ">=3.15")
