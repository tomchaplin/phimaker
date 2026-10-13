# Tests

The tests use small, deterministic F2 complexes with hand-derived answers. No
random datasets, plotting libraries, external matrix files, or network access
are needed when running them. Every test has a docstring or Rust documentation
comment explaining the invariant it checks.

## Running the suites

From the repository root, install the Python test dependencies and build the
local extension (requires Rust and Python >= 3.12):

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

The CI-only matrix tests use Python >= 3.12, `packaging`, and `pytest`, and do not
import the extension. Run them separately from the Python API suite:

```sh
uv run --no-project --python 3.13 --with packaging --with pytest pytest tests/ci
```

They check version bounds, latest stable patch selection, exclusion of
prereleases and experimental free-threaded interpreters, and failure when a
selected interpreter or an entire platform is missing. The wheel workflow also
checks generated public type declarations and the `py.typed`
marker, installs the built wheel without rebuilding it, and runs `tests/python`
against that installation. The Python suite verifies that the exported result
classes match the objects returned by the functions.

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
  requested empty representative maps, and the requirement to retain V for
  nonempty decompositions when representatives are requested. Optional extraction
  tests verify that both flags give identical diagrams, that skipped extraction
  returns `None` even for empty diagrams, and that requesting empty maps works in
  both modes. Diagram-only extraction is tested with V absent separately from
  each relevant decomposition, including serialized results and cursor rewinds.
  Python tests check the default, explicit flags, and forwarding through both
  convenience functions; representative tests opt in to cycle extraction.
- **Private cycle-coordinate calculation:** sparse F2 back substitution over
  all combinations of a small basis; sums of codomain or image features; dead target cycles needed for
  exact expansion but omitted from homology coefficients; identity and empty maps;
  image-to-codomain inclusions with hand-derived coefficients, pointwise F2 rank,
  and domain-to-image-to-codomain factorization at every filtration index. These
  cover different representative bases, dead targets, empty images, and identity
  and zero-map cylinders in both modes. Composition uses only intervals alive at
  each index; a separate regression checks that nonzero maps between `[2, 4)`,
  `[1, 3)`, and `[0, 2)` compose to zero despite a nonzero ordinary matrix product;
  zero general maps after equal-time cylinder cells enter; missing or inconsistent
  representatives, non-spanning target bases, incompatible interval lifetimes,
  and normalization of unsorted or repeated support indices. Crate-private Rust
  tests also cover non-identity generator images, F2 cancellation, missing images
  as zero, ignored unused images, future-born target coefficients, and unchanged
  inputs. These tests preserve calculation coverage during the transition to
  ensemble maps; no standalone persistence-map function is publicly exported.
- **Kernel representatives:** finite and essential cycles use only domain
  generators preceding their kernel birth. A finite two-class fixture requires
  the kernel death basis rather than the image boundary at birth. Both modes
  check exact supports, cyclehood and degree, independence modulo domain
  boundaries, membership in codomain boundaries, pointwise kernel-to-domain
  injectivity, and the zero composite into codomain homology. Optional flags,
  empty maps, copied Python getters, and a zero-map cylinder are covered.
  Private cycle-coordinate tests allow earlier-supported cycles on either
  side, reduce independent target cycles sharing a pivot, and reject future
  target features, future support generators, and dependent target cycles.
- **Cokernel representatives:** finite death columns and essential codomain birth
  cycles are checked in both modes. Independent rank tests verify that active
  representatives form a basis modulo codomain boundaries and all available
  domain cycles, and that finite representatives vanish precisely at their
  deaths. A finite H0 class can die in the cokernel while remaining nonzero in
  codomain homology. A parallel-edge loop detects incorrect use of cokernel V,
  which can mix degrees and fail cyclehood. Empty/identity inclusions, permuted
  domain indices, cylinders, optional getters, and absent cokernel V are covered.
- **Python bindings:** dictionary lookup, missing-key behavior, independent
  getter copies, `None` for infinity, metadata, signatures, and docstrings. The
  retired standalone map function is absent from both the package and native
  extension, and CI checks that generated type declarations omit it.
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

## Local LoPhat 0.14 development

Until LoPhat 0.14.0 is published, the Cargo dependency includes both its version
and the sibling checkout path `../lophat`. This makes the new diagram conversion
available for local tests. Remove the path and regenerate the Cargo lockfile
against crates.io after publishing LoPhat, before running standalone CI or
releasing Phimaker. Published crate manifests use the declared registry version.
