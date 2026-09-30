# Tests

The tests use small, deterministic F2 complexes with hand-derived answers. No
random datasets, plotting libraries, external matrix files, or network access
are needed when running them. Every test has a docstring or Rust documentation
comment explaining the invariant it checks.

## Running the suites

From the repository root, install the Python test dependencies and build the
local extension (requires Rust and Python >= 3.10):

```sh
uv sync --locked --group test
uv run --no-sync maturin develop --locked
uv run --no-sync python -m pytest
```

Re-run `maturin develop` after Rust changes so pytest exercises the current
source rather than a previously installed extension. Pytest is configured to
collect only `tests/python`. There are no tests that import a development
shared library by an absolute path.

Run the Rust unit tests independently:

```sh
cargo test --locked
```

The Rust tests call builders, decompositions, and diagram extraction directly;
they do not import the Python extension or initialize the Python interpreter.
The crate still has a PyO3 build dependency; if automatic interpreter detection
chooses an unsupported Python version, set `PYO3_PYTHON` to a supported
interpreter, for example `PYO3_PYTHON="$PWD/.venv/bin/python" cargo test --locked`.

Check generated documentation with:

```sh
RUSTDOCFLAGS="-D warnings" cargo doc --locked --no-deps
```

## Coverage and intended answers

- **A vertex in an interval:** codomain pairing is always `{0: infinity, 1: 2}`,
  even when the domain is the younger vertex. This catches confusion between
  domain-first image rows and ordinary codomain persistence. All six diagrams
  are compared in both modes.
- **Unsorted domain indices:** two disjoint acyclic two-cell complexes preserve
  the same finite pairs for sorted and reversed domain input, in both modes.
- **Empty and identity inclusions:** empty diagrams remain empty, and the
  quotient of an identity inclusion has no extra basepoint or homology.
- **Triangle boundary in a filled triangle:** the face kills H1 in the image
  and starts an essential kernel class. The quotient has an essential H2 class.
- **A face in a tetrahedron boundary:** the domain H1 class dies at 13 but its
  image dies at 12, giving kernel `[12, 13)`. The resulting sphere gives an
  essential codomain/cokernel H2 class at 13. All finite H0/H1 bars and relative
  bars are checked too.
- **General identity and zero maps:** identity has no positive-duration kernel,
  cokernel, or cone homology; a zero map has full kernel and cokernel, with a
  suspended domain summand in the cone. Tests translate cylinder indices to
  both time and degree, retaining multiplicities and essential classes.
- **Square filling at different times:** verifies every diagram for a
  nontrivial chain map, including H0 kernel multiplicity and H1 kernel `[2, 10)`.
  Zero-duration cylinder intervals are removed only in these time-based
  comparisons, not in the inclusion tests that compare exact indices.
- **Rust helpers:** direct cylinder boundaries, degree shifts, tie ordering,
  strict upper triangularity and boundary squared zero; domain/image/relative
  coordinate conventions; permutation inversion; finite/infinite endpoints;
  anti-transpose recovery; stable display; serialization and explicit rewind.
- **Python bindings:** dictionary lookup, missing-key behavior, independent
  getter copies, `None` for infinity, metadata, signatures, and docstrings.

The meaningful old diagram tests are retained or expanded. The previous Rust
`ensemble_works` test opened a nonexistent fixture path and asserted only
`true == true`; it is replaced with direct assertions. The old Python binding
tests used the removed annotated-column API and expected a spurious relative
H0 class. The print-only tetrahedron and square scripts are now unit tests;
the square's diagonal had incorrectly been labeled degree two rather than one.
Random timing/plotting scripts and their unused matrix fixture were removed.
The existing Rust square test now also rules out extra essential kernel bars.

## Known upstream limitation

LoPhat 0.11.0 accesses column zero when serializing an empty decomposition.
`test_known_limitations.py` records the empty-domain and empty-relative slow
cases as strict expected failures. Only the specific PyO3 bounds panic is
converted to the expected exception; another error still fails. Success is an
unexpected pass and also fails, prompting removal of the marks after an
upstream fix. The corresponding normal-mode edge cases are ordinary passing
tests. All other slow-mode cases must pass normally.

Malformed chain complexes are outside the documented API preconditions. The
suite does not require validation errors the API does not promise, nor does it
encode performance expectations for the deferred slow-path optimization.
