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
  anti-transpose recovery; forward codomain coordinates and retained V;
  stable display; serialization and explicit rewind. A pair of parallel edges
  checks a nontrivial essential cycle, a finite H0 representative, and R = D V
  against the original boundary, including the serialized slow-mode result.
  Representative tests also cover a domain with gaps and unsorted indices,
  essential domain cycles killed in the image, essential image cycles supported
  only in the domain, cycle boundaries, maximum support indices equal to births,
  empty representative maps, and the requirement to retain V for nonempty
  decompositions in both modes.
- **Persistence maps:** sparse F2 back substitution over all combinations of a
  small basis; sums of codomain or image features; dead target cycles needed for
  exact expansion but omitted from homology coefficients; identity and empty maps;
  zero general maps after equal-time cylinder cells enter; missing or inconsistent
  representatives, non-spanning target bases, incompatible interval lifetimes,
  and normalization of unsorted or repeated support indices. Both modes and
  independent Python return copies are tested.
- **Python bindings:** dictionary lookup, missing-key behavior, independent
  getter copies, `None` for infinity, metadata, signatures, and docstrings.
- **Documentation:** Python functions, shared classes, and properties expose
  Python usage docs without Rust comments. Their examples run through Python's
  doctest parser under pytest. Rust examples run as Cargo doctests; rustdoc is
  checked with and without default features to ensure it shows Rust docs only.

The meaningful old diagram tests are retained or expanded. The previous Rust
`ensemble_works` test opened a nonexistent fixture path and asserted only
`true == true`; it is replaced with direct assertions. The old Python binding
tests used the removed annotated-column API and expected a spurious relative
H0 class. The print-only tetrahedron and square scripts are now unit tests;
the square's diagonal had incorrectly been labeled degree two rather than one.
Random timing/plotting scripts and their unused matrix fixture were removed.
The existing Rust square test now also rules out extra essential kernel bars.

## Empty decompositions

LoPhat 0.12.0 fixes empty-decomposition serialization.
`test_known_limitations.py` now checks empty-domain and empty-relative slow-mode
results against normal mode as ordinary passing regressions, without expected
failure marks. Diagram extraction uses LoPhat's shared map representation.

## Structural validation

Rust tests check all `InclusionError` variants at both decomposition entry points
and the Rust diagram convenience function. Python tests check conversion to
`ValueError`, unchanged public function names, and propagation through the
general-map wrapper. Both modes cover short/long degree lists, domain indices at
and beyond the boundary size (including empty input), diagonal/out-of-range
boundary rows, degree-zero and overflow edge cases, and domain closure.
Normalization tests check unsorted/repeated supports and domain indices, exact
six-diagram results, canonical metadata, and preservation of borrowed inputs.
The existing unsorted-domain tests now exercise validation directly from Rust.

Boundary squared zero and the full filtered chain-map identity remain caller
preconditions. The suite does not require checks for those unchecked algebraic
conditions, or performance expectations for the deferred slow-path optimization.
