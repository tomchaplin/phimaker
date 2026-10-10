<div align="center">

<h1>phimaker</h1>

<b>P</b>ersitent <b>h</b>omology of <b>im</b>ages <b>a</b>nd (co)<b>ker</b>nels.

</div>

## Overview

Phimaker is a Rust library implementing the algorithm introduced in [[1]](#1) for computing persistent homology for kernels, images and cokernels.
Python bindings are provided via PyO3.
The Python package requires Python 3.12 or newer.

See [CHANGELOG.md](CHANGELOG.md) for changes since the last release and API
migration notes.

Install via
```
pip install phimaker
```

## Development environment and source builds

Install [Rust through rustup](https://rustup.rs/),
[uv](https://docs.astral.sh/uv/getting-started/installation/), and
[Lefthook](https://lefthook.dev/installation/). A native C linker is also
required (for example, the build tools for your operating system).

This checkout currently resolves LoPhat 0.14.0 from `../lophat`. Clone
[LoPhat](https://github.com/tomchaplin/lophat) alongside Phimaker and check out
`v0.14.0` or a compatible newer revision before building. See
[the local dependency notes](tests/README.md#local-lophat-014-development).

Run the following commands from the repository root:

```sh
rustup toolchain install stable
rustup toolchain install nightly --component rustfmt
uv python pin 3.12
uv sync --locked --group test
lefthook install
```

uv creates `.venv`, installs the development and test tools, and builds the
Python extension from source. Python 3.12 or newer is required; pin another
supported version if desired. The Git pre-commit hooks run nightly
rustfmt, Tombi, and Ruff, and stage their formatting changes. You can also run
them manually with `lefthook run pre-commit`.

Rebuild the extension and its generated type stubs after Rust or docstring
changes, then run the Python tests:

```sh
uv run --no-sync maturin develop --locked --generate-stubs
uv run --no-sync python -m pytest
```

Build a distributable wheel into `dist/` with:

```sh
uv run --no-sync maturin build --locked --release --generate-stubs --out dist
```

For the Rust library without Python bindings, use:

```sh
cargo build --locked --release --no-default-features
cargo test --locked --no-default-features
```

## Persistence diagrams

`DiagramEnsemble` exposes the diagrams as `domain`, `codomain`, `image`,
`kernel`, `cokernel`, and `relative` in both Rust and Python.

Each diagram is a mapping from a generator's birth index to its death index.
In Rust, the shared `lophat::utils::PersistenceDiagram` wraps a `HashMap<usize, ExtendedUsize>`, with
`ExtendedUsize::Finite(index)` for finite deaths and `ExtendedUsize::Infinity`
for essential classes. Both types are re-exported from `phimaker::diagrams`.
Birth and death indices refer to filtration columns.

In Python, diagrams are dictionaries with integer keys and integer or `None`
values. For example, `diagrams.codomain == {0: None, 1: 2}` describes an essential
class born at index 0 and a class born at index 1 that dies at index 2.
Use `diagram[birth]` to look up a death and `diagram.items()` to iterate over
intervals. The former `.paired` and `.unpaired` attributes have been removed.

`domain_representatives`, `codomain_representatives`, `image_representatives`,
and `kernel_representatives`
are optional. Python's `sixpack_from_inclusion` and `sixpack` default to
`get_representatives=False`, which leaves all four attributes as `None` and
skips representative extraction. Pass `get_representatives=True` to obtain
maps from each birth index to the generator indices whose sum over F2 gives a cycle.
Keys and supports use the diagram coordinates. Domain, codomain, and image cycles
have maximum support index equal to the feature's birth. Kernel cycles contain
only domain generators with indices strictly less than their kernel birth; these
cycles are boundaries in the codomain throughout their kernel intervals.
Rust stores these as `Option<HashMap<usize, Vec<usize>>>`;
Python exposes dictionaries of lists or `None`. Each requested map has exactly
its diagram's birth keys. A requested empty diagram has `Some` of an empty map
in Rust and `{}` in Python; when extraction is skipped it still has `None`.

Rust callers supply `get_representatives` immediately after the input matrices
and map or domain indices in `sixpack` and `sixpack_from_inclusion`, or as the
argument to `all_diagrams`. Pass `false` for diagrams only or
`true` for cycles too. Nonempty domain, codomain, and image decompositions must
retain V only when requesting representatives; the computation entry points
retain these matrices automatically. This flag skips representative extraction
and allocation; it does not change which matrices the reductions retain.

Call `get_persistence_map(source, source_representatives, target,
target_representatives, generator_map)` for one persistence-module map. The
representative arguments must be actual maps; `None` is not accepted. Request
representatives when obtaining the input diagrams from the computation functions.
The result maps every source birth to a sorted list of target births with binary
coefficient one. The generator map sends each source generator to a list of
target generators. Missing entries implicitly map to zero, and overlaps between
distinct mapped columns cancel over F2. Repeated listings within one supplied
column are ignored. Inputs are not modified.

Both diagrams must use one common ordered filtration index scale. Independently
numbered complexes must be reindexed before calling; the generator map does not
establish how their filtration indices compare. This is a caller contract.
Representatives must be cycles and the mapping must describe a degree-preserving
filtered chain map. These algebraic requirements are not checked. Representative
keys, availability of support generators at birth, interval endpoints, solvability, and coefficient
lifetimes are checked. Each retained coefficient satisfies
`target_birth <= source_birth < target_death <= source_death`. Targets already
dead at the source birth contribute zero after exact cycle expansion.
Representatives must be nonempty and use generator indices no greater than their
feature's birth. The maximum support index can be strictly earlier than the birth,
on either side of the map. Target representatives must be linearly independent
as chain vectors; overlapping pivots are supported. Feature labels are preserved
when expanding cycles, and targets may be born earlier than their source.

Rust returns a `Result` with `PersistenceMapError`; Python reports semantic
failures as `RuntimeError`. Python diagrams are ordinary birth-to-death
dictionaries, with `None` for essential deaths. An empty source gives an empty
map. At a filtration index, use only intervals alive there. For `sixpack`, use
`metadata.times` to select intervals in the original filtration; the diagrams
retain equal-time mapping-cylinder intervals.

For inclusion results, use identity images for the generators appearing in
source representatives and call the function separately for each target:

```python
from phimaker import get_persistence_map, sixpack_from_inclusion

diagrams = sixpack_from_inclusion([[], [], [0, 1]], [0, 0, 1], [1], get_representatives=True)
assert diagrams.domain_representatives is not None
assert diagrams.codomain_representatives is not None
assert diagrams.image_representatives is not None
generator_map = {
	index: [index] for support in diagrams.domain_representatives.values() for index in support
}
to_codomain = get_persistence_map(
	diagrams.domain,
	diagrams.domain_representatives,
	diagrams.codomain,
	diagrams.codomain_representatives,
	generator_map,
)
to_image = get_persistence_map(
	diagrams.domain,
	diagrams.domain_representatives,
	diagrams.image,
	diagrams.image_representatives,
	generator_map,
)
assert to_codomain == {1: [0, 1]}
assert to_image == {1: [1]}
```

Kernel representatives can be mapped into the domain with the same identity
generator map. Here an external edge creates a kernel class at 4, and a domain
edge kills it at 5:

```python
diagrams = sixpack_from_inclusion(
	[[], [], [], [0, 1], [0, 2], [1, 2]],
	[0, 0, 0, 1, 1, 1],
	[1, 2, 5],
	get_representatives=True,
)
assert diagrams.kernel == {4: 5}
assert diagrams.kernel_representatives == {4: [1, 2]}
assert get_persistence_map(
	diagrams.kernel,
	diagrams.kernel_representatives,
	diagrams.domain,
	diagrams.domain_representatives,
	{1: [1], 2: [2]},
) == {4: [2]}
```

## References

<a id="1">[1]</a>
Cohen-Steiner, D., Edelsbrunner, H., Harer, J. and Morozov, D., 2009, January.
Persistent homology for kernels, images, and cokernels.
In Proceedings of the twentieth annual ACM-SIAM symposium on Discrete algorithms (pp. 1011-1020).
Society for Industrial and Applied Mathematics.

## API documentation

Install the Python documentation dependencies into the project environment,
rebuild the extension, and generate the HTML reference:

```sh
uv sync --locked --group test --group docs
uv run --no-sync maturin develop --locked --generate-stubs
uv run --no-sync python docs/build.py
```

Open `docs/_build/html/index.html` in a browser. Sphinx reads the installed
extension's docstrings, so rebuild the extension after changing them. The build
script treats warnings as errors. The equivalent Sphinx command is:

```sh
uv run --no-sync python -m sphinx -W --keep-going -b html docs docs/_build/html
```

Run `cargo doc --locked --no-default-features --no-deps --open` to build and open
the Rust API reference at `target/doc/phimaker/index.html`. The public
functions document input assumptions and index coordinates, with runnable Rust
examples. Inline `///` comments supply Rust documentation, while separate
feature-gated `#[doc = python_doc!(...)]` attributes supply Python docstrings.
Shared classes retain Rust comments for rustdoc and rust-analyzer; Python builds
strip those comments so only Python usage documentation appears in `help()`.
Use `help(phimaker.sixpack_from_inclusion)`, `help(phimaker.sixpack)`, or
`help(phimaker.get_persistence_map)` after building/installing the extension
for Python examples.

The inclusion API takes boundary columns, degrees, and domain column indices.
It uses filtration indices only. The general-map API `sixpack` takes timed
complexes and returns mapping-cylinder metadata alongside its diagrams; use
that metadata to interpret the returned indices. Inclusion entry points validate
index ranges, degree-list length, triangularity, degree lowering, and domain
closure, and normalize row supports and domain indices. The boundary must square
to zero; that algebraic condition remains a caller precondition.

Rust `sixpack_from_inclusion` and `sixpack` borrow their inputs and return
`Result<_, InclusionError>`. The decomposition entry points return the same
structured errors. Python exports retain the names `sixpack_from_inclusion` and
`sixpack`; their wrappers release the GIL during computation and convert
structural errors to `ValueError`. Upstream reduction and serialization panics
are not converted to these validation errors.

Boundary supports and domain indices can be in any order, and repeated indices
are treated as repeated listings of the same element, not summed modulo two.
The Rust library is built as an `rlib` alongside the Python extension.

For reduced matrices, use `all_decompositions` or `all_decompositions_slow`.
Their codomain result `d_cod` retains both R and V in original input generator
coordinates. For a finite codomain feature `(birth, death)`, R column `death`
gives its representative cycle; for an essential feature, use V column `birth`.
Both diagram extraction modes read codomain pairs directly in these coordinates.
The relative decomposition still uses anti-transposed quotient coordinates.
Retaining codomain V adds memory use and increases the slow-mode file size.
Python convenience functions expose representative maps on the returned
`DiagramEnsemble` when called with `get_representatives=True`.

## Tests

Run `cargo test --locked` for the pure Rust suite. For Python, build the current
extension and run pytest as described in [tests/README.md](tests/README.md).
That document records each fixture's intent, including slow-mode serialization
regressions and executable documentation examples.

The Cargo dependency uses LoPhat version 0.14.0.
Building with `python-module` also enables `lophat/python-module`, providing
conversion of diagrams to dictionaries and infinite deaths to `None`.

## CI and releases

Every push, pull request, and manual workflow dispatch runs the Rust tests and
builds and tests Python wheels on native Linux and Windows x64/ARM64 runners and
macOS ARM64. The matrix selects stable CPython versions allowed by
`requires-python`, including supported free-threaded variants where available.
Each wheel includes generated type stubs and is installed before pytest runs.
`DiagramEnsemble` and `CylinderMetadata` are importable from `phimaker` for typed
results.

Publishing a GitHub release runs these checks again and publishes the tested
wheels to PyPI and the crate to crates.io. See [docs/releasing.md](docs/releasing.md)
for credentials, platform coverage, and the release procedure.
