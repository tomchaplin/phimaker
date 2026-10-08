"""Build the Python API reference from the installed extension."""

from importlib.metadata import version as package_version

project = "Phimaker"
copyright = "2026, Thomas Chaplin, Abhinav Natarajan"  # noqa: A001
author = "Thomas Chaplin, Abhinav Natarajan"
release = package_version("phimaker")

extensions = ["sphinx.ext.autodoc"]
autoclass_content = "class"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
html_theme = "sphinx_rtd_theme"
html_static_path = []
