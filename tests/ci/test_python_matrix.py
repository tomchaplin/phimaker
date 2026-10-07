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
		downloads("3.9.25")
		+ downloads("3.13.1")
		+ downloads("3.13.2")
		+ downloads("3.13.2", "freethreaded")
		+ downloads("3.14.0")
		+ downloads("3.14.0", "freethreaded")
		+ downloads("3.15.0rc2")
	)
	entries = matrix_module.make_matrix(catalogue, ">=3.10,<3.15")["include"]
	assert len(entries) == 3 * len(matrix_module.PLATFORMS)
	assert {entry["label"] for entry in entries} == {"3.13", "3.14", "3.14t"}
	assert {entry["python"] for entry in entries} == {"3.13.2", "3.14.0", "3.14.0t"}
	linux_t = next(
		entry for entry in entries if entry["name"] == "linux-x64" and entry["label"] == "3.14t"
	)
	assert linux_t["interpreter"] == "/opt/python/cp314-cp314t/bin/python"


def test_older_minor_missing_on_windows_arm() -> None:
	"""Skip an unavailable 3.10 build while still testing newer native ARM interpreters."""
	older = [
		entry
		for entry in downloads("3.10.20")
		if (entry["os"], entry["arch"]) != ("windows", "aarch64")
	]
	entries = matrix_module.make_matrix(older + downloads("3.11.15"), ">=3.10")["include"]
	arm = [entry for entry in entries if entry["name"] == "windows-arm64"]
	assert [entry["label"] for entry in arm] == ["3.11"]
	assert len(entries) == 2 * len(matrix_module.PLATFORMS) - 1


def test_missing_platform_is_an_error() -> None:
	"""An entire missing platform must fail CI instead of silently shrinking coverage."""
	with pytest.raises(ValueError, match="No supported CPython downloads"):
		matrix_module.make_matrix(downloads("3.14.0")[:-1], ">=3.10")


def test_no_matching_stable_release() -> None:
	"""An unsatisfiable Python requirement cannot produce an empty successful matrix."""
	with pytest.raises(ValueError, match="No stable CPython releases"):
		matrix_module.make_matrix(downloads("3.14.0"), ">=3.15")
