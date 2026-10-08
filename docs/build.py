"""Generate the HTML reference, treating Sphinx warnings as build failures."""

from pathlib import Path

from sphinx.cmd.build import build_main

if __name__ == "__main__":
	docs = Path(__file__).resolve().parent
	raise SystemExit(
		build_main(["-W", "--keep-going", "-b", "html", str(docs), str(docs / "_build" / "html")])
	)
