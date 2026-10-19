# Changelog

This changelog records user-facing changes and development changes. New changes
are listed under Unreleased; released changes are recorded below.

## [Unreleased]

### Breaking changes

- Convert `sixpack` representative supports to original input coordinates
  after map extraction and equal-time filtering. Domain and kernel cycles use
  domain indices; image, codomain, and cokernel cycles use codomain indices.
  Relative pairs use domain indices in their first component and codomain
  indices in their second, satisfying `boundary(b) = f(a)`. Implement the
  pullback and F2 chain projection in `cylinder.rs`. Keep interval identifiers,
  metadata, and surviving map coefficients unchanged. Verify bases and maps
  against the original complexes and their algebraic cones in both modes.
- Return relative representatives as `(a, b)` pairs of domain cycles and
  codomain chains, with `a = boundary(b)` in inclusion coordinates. Keep both
  supports in original ambient indices and retain the optional extraction
  contract. Compute the boundary from the retained codomain decomposition in
  normal and serialized modes. Update Python getter types and wheel checks.
- Omit finite equal-time intervals from all six diagrams returned by
  `sixpack`. Compute the full representatives and persistence maps first,
  then remove the intervals, their representative entries, and their map
  rows and columns. Preserve empty columns for surviving zero maps and leave
  interval identifiers and metadata unchanged. Index-only inclusion and
  decomposition entry points continue to return complete diagrams.

### Changed

- Sum generator images and basis expansions with a shared vector-based F2
  helper that sorts and cancels once per complete sum. Replace hash-set
  accumulation in cylinder projection and repeated pairwise sums during map
  extraction; keep pivot-dependent reductions incremental.
- Simplify sparse cycle-coordinate solving and kernel birth detection. Extract
  finite image representatives directly from reduced death columns, avoiding
  extraction of an unused image representative dictionary. Image diagram
  extraction no longer requires retained image V once the kernel matrix has
  been built; cover finite image and kernel intervals in both extraction modes.
- Run Cargo builds, tests, and lockfile generation from the corresponding
  Makefile targets. Update libc to 0.2.191 and refresh the Python lockfile export
  without changing its package versions.

## [0.4.0]

### Breaking changes

- Replace the Python inclusion entry point `compute_ensemble(matrix, ...)` with
  `sixpack_from_inclusion(boundary_matrix, degrees, cols_in_domain, ...)`.
  Supply boundary supports, generator degrees, and domain indices separately
  instead of annotated `(in_domain, dimension, boundary)` columns.
- Rename `compute_ensemble_cylinder` to `sixpack` for general filtered chain
  maps. Its matrices use `(entrance_time, degree, boundary_indices)` columns.
- Use the full `DiagramEnsemble` member names `domain`, `codomain`, `image`,
  `kernel`, `cokernel`, and `relative`, replacing the abbreviated names.
- Represent each diagram as a birth-to-death mapping, with birth indices
  identifying generators. Python returns dictionaries with integer deaths or
  `None` for essential classes; replace `.paired` and `.unpaired` access with
  dictionary lookup or iteration. Rust uses `PersistenceDiagram` and
  `ExtendedUsize::{Finite, Infinity}`, now re-exported from LoPhat.
- Rename `CylinderMetadata.dimensions` to `degrees` and use `degrees` as the
  inclusion function's keyword argument, matching LoPhat's degree terminology.
- Make Rust `sixpack`, `sixpack_from_inclusion`, `all_decompositions`, and
  `all_decompositions_slow` return `Result<_, InclusionError>`. Convenience
  functions borrow their inputs and no longer require a Python interpreter.
- Store `DecompositionEnsemble.d_cod` and `FileEnsemble.d_cod` in original
  input coordinates instead of anti-transposed coordinates. Consumers of raw
  codomain decompositions must stop reversing indices. Extracted codomain
  diagrams retain their original coordinates.
- Store `DecompositionEnsemble.d_rel` and `FileEnsemble.d_rel` in forward local
  quotient coordinates instead of anti-transposed coordinates. Consumers of
  raw relative decompositions must stop reversing indices. Extracted relative
  diagrams retain their original codomain coordinates.
- Replace `VectorMapping`, `IndexMapping`, annotated-column helpers, and row
  reordering abstractions with permutation-based index translation. Make matrix
  builders and individual reduction helpers internal implementation details.
- Remove `zero_overlap` and its implementation module, and remove the washboard
  submodule and Python interface.
- Require Python 3.12 or newer, matching LoPhat's Python requirement, and
  move the Rust crate to edition 2024.
- Remove the public standalone `get_persistence_map` function in Rust and
  Python and its public `PersistenceMap` and `PersistenceMapError` types.
  Computation entry points accept filtered chain complexes and chain maps;
  there is no public solver for independently supplied persistence diagrams.
  Retain the cycle-coordinate calculation privately with Rust regression tests.
- Make domain, codomain, and image representatives optional on `DiagramEnsemble`.
  Rust `sixpack`, `sixpack_from_inclusion`, and both `all_diagrams` methods take
  a `get_representatives` flag. Python convenience functions default it to
  `False`; pass `get_representatives=True` when representative cycles are needed.

### Added

- Optional `relative_representatives` on `DiagramEnsemble` in Rust and Python,
  controlled by `get_representatives` in both normal and serialized modes.
  Reduce the quotient boundary directly and retain V; finite features use R
  death columns and essential features use V birth columns. Translate birth
  keys and supports to original codomain or mapping-cylinder coordinates.
  Supports use only generators outside the domain, have maximum index equal
  to birth, and have boundary contained in the domain. Document their classes
  modulo codomain boundaries and domain chains. Test quotient cyclehood,
  pointwise basis completeness and independence, finite deaths, empty and
  identity inclusions, permuted coordinates, cylinders, optional extraction,
  copied Python getters, and generated getter types. Retaining relative V
  adds memory use and increases serialized relative data size.
- Optional `kernel_to_domain`, `domain_to_image`, `image_to_codomain`,
  `domain_to_codomain`, and `codomain_to_cokernel` maps on `DiagramEnsemble`.
  With `get_representatives=true`, both extraction modes compute binary
  coefficients in the stored cycle bases, keyed by every source birth with
  sorted target births and explicit empty columns for zero. Skipped extraction
  leaves all five maps as `None`, including empty sources. Compute the quotient
  map using the combined image and cokernel cycle basis and discard image
  coordinates before applying cokernel lifetimes. Preserve raw cylinder indices
  and zero-duration intervals. Document interval-aware evaluation/composition,
  update Python getter types, and test chain actions, pointwise ranks,
  exactness, factorization, zero composites, and independent getter copies.
- Optional `cokernel_representatives` on `DiagramEnsemble` in Rust and Python,
  controlled by `get_representatives` in both normal and serialized modes.
  Finite features use reduced cokernel death columns; essential features use
  codomain V birth columns, giving homogeneous codomain cycles with maximum
  support index equal to birth. Classes are interpreted modulo codomain
  boundaries and domain cycles. No additional V matrices are retained.
  Add tests for quotient independence and finite deaths, empty and identity
  inclusions, gapped domains, cylinders, and an essential loop for which
  cokernel V would give a noncycle. Update both language references, examples,
  optional-field checks, and generated Python interface checks.
- Optional `kernel_representatives` on `DiagramEnsemble` in Rust and Python,
  controlled by the existing `get_representatives` flag in both computation
  modes. Cycles use only domain generators preceding the kernel birth, bound
  in the codomain, and follow finite kernel death pairings.
  Add regressions for finite and essential classes, differing birth/death cycle
  bases, domain index translation, and zero-map cylinders; check independence
  in domain homology and the zero composite into codomain homology.
- Private Rust regressions for image-to-codomain persistence maps in both
  computation modes, including sums, different representative bases, dead targets,
  empty images, and identity and zero-map cylinders. Verify pointwise injectivity
  and factorization of domain-to-codomain through the image, accounting for interval
  lifetimes when composing maps. Include nonzero interval maps whose composition
  is zero because the outer intervals do not overlap.
- Domain, codomain, and image representative maps on `DiagramEnsemble`, keyed
  by birth index with cycle supports in diagram coordinates. Extract representatives
  in both normal and slow modes using LoPhat's `diagram_with_reps`; translate domain
  and image row indices back to original generator indices and use domain cycles
  for essential image features. Store representatives as optional maps: `None`
  when extraction is skipped, or a map (empty for empty diagrams) when requested.
  Nonempty domain, codomain, and image decompositions must retain V when
  representatives are requested. Diagram-only extraction does not require V.
- A shared inclusion preparation step with typed `InclusionError` variants for
  degree-count mismatches, invalid domain indices, non-upper-triangular
  boundaries, incorrect boundary degrees, and domains not closed under boundary.
  Python bindings report these structural errors as `ValueError`.
- Normalization of unsorted boundary supports and domain indices, including
  repeated listings. Repeated indices identify the same nonzero entry and are
  ignored rather than cancelled modulo two. Borrowed inputs are not modified;
  already sorted inputs avoid an unnecessary sort.
- An `rlib` alongside the Python extension and a default `python-module` feature.
  Building with `--no-default-features` provides the Rust library without the
  Python bindings or Python dependencies. Re-export the decomposition entry
  points at the crate root.
- Separate Rust and Python usage documentation, with executable examples and
  explicit input assumptions, error behavior, diagram coordinates, and kernel
  degree conventions. Rust uses inline doc comments; Python uses gated raw doc
  attributes. Shared classes strip Rust comments in Python builds while
  retaining them for rustdoc and rust-analyzer.
- Deterministic Rust unit tests and pytest tests for all six diagrams, empty and
  identity inclusions, unsorted domains, finite and essential kernels, general
  chain maps, cylinder metadata, serialization, structural validation, and input
  normalization. Document each fixture's intent and execute documentation examples.

### Changed

- Document the representative birth guarantee for `sixpack_from_inclusion`
  in both Rust and Python: domain, codomain, image, cokernel, and relative
  representatives have maximum generator index equal to their feature's
  birth, for every thread count. Explain the corresponding maximum birth-value
  guarantee for nondecreasing filtration values, including ties, and demonstrate
  the index guarantee in the executable usage examples.
- Document representative reproducibility in the Rust and Python docstrings of
  `sixpack_from_inclusion`: multithreaded calls may choose different cycles for
  identical inputs. With one worker, domain representatives are independent of
  the ambient codomain up to order-preserving relabelling, and codomain
  representatives are independent of the selected domain. Explain using one
  worker for compatible representative bases across multiple inclusions.
- Allow the private cycle-coordinate calculation to use representatives
  supported strictly before their feature births. Validate availability by birth instead of
  requiring equality with the maximum support index. Reduce target cycle bases
  with overlapping pivots while preserving their feature labels; reject
  dependent target cycles and retain interval lifetime checks. Document the
  relaxed contract and cover kernel-style inputs and future-born target features.
- Adapt LoPhat's Python wheel and Rust crate workflows. Every push, pull request,
  and manual dispatch runs Rust tests and builds, installs, and tests each Python
  wheel. A published GitHub release uploads tested wheels to PyPI and publishes
  the tested crate to crates.io.
- Select stable CPython versions from `requires-python` and uv's interpreter
  catalogue, including supported free-threaded variants. Use native Linux and
  Windows x64/ARM64 and macOS ARM64 runners. Match LoPhat's strict matrix policy:
  every selected minor and ABI variant must be available on every platform.
  Add matrix-selection regression tests and regenerate the lockfile for Python
  3.12 and newer, removing dependencies needed only on older Python versions.
- Replace the checked-in placeholder stub with stubs generated from declarative
  PyO3 exports, explicitly enable introspection metadata, and include `py.typed`.
  Export `DiagramEnsemble` and `CylinderMetadata` from Python so typed result
  objects can be imported. Check public declarations in every built wheel,
  including optional representative getters and the absence of the retired
  standalone persistence-map function.
- Include the README in Python package metadata and correct its repository URL.
  Document publishing credentials and platform coverage. The copied workflows
  publish wheels only; they no longer build a source distribution, Intel macOS,
  PyPy, musllinux, or the other architectures from the previous wheel matrix.
- Upgrade LoPhat from 0.11.0 to 0.14.0, use its shared diagram and extended-index
  types, and adapt column access to `degree()`. Delegate Python dictionary and
  infinite-death conversion to LoPhat when `python-module` is enabled.
- Upgrade PyO3 from 0.25.1 to the 0.29 series and pyo3-log from 0.12.4 to 0.13.4.
  Update compatible Rust dependencies and the lockfile, including atomic_refcell,
  crossbeam, log, rayon, serde, serde_json, and tempfile. Python computations
  continue to release the GIL using the current PyO3 API.
- Simplify inclusion inputs and index handling, preserve original codomain
  indices in inclusion diagrams, and document metadata-based translation of
  general-map diagram endpoints from mapping-cylinder indices.
- Reduce the codomain boundary directly and retain V in both normal and slow
  modes, making R death columns and essential V birth columns available for
  representative-cycle extraction. Read codomain diagrams directly from this
  reduction. The relative reduction also now preserves forward order. Retaining V
  adds memory use and increases serialized codomain data size.
- Add GPL-3.0-or-later licensing, the license text, and package author,
  repository, and classification metadata. Update the Maturin CI action.

### Fixed

- Skip image representative extraction when representatives are disabled.
- Construct the relative complex as the quotient by the domain without an extra
  degree-zero basepoint. Identity inclusions now have empty relative homology.
- Correct kernel classification and index translation across original,
  domain-first, and anti-transposed coordinates. Normalize domain order before
  building domain matrices and interpreting their pairings.
- Support empty-decomposition serialization through LoPhat 0.12.0. Slow mode
  handles empty domains and empty quotients; the former expected-failure tests
  now run as passing regressions.
- Check invalid indices and degree-zero or overflow cases before reduction,
  with the same validation behavior in normal and slow modes.
- Correct test expectations and fixture degrees, including the square's diagonal
  and essential kernel bars. Replace a fixture-dependent Rust test with direct
  assertions.

### Development

- Document uv environment setup, Lefthook installation, source builds, and
  local Rust and Python documentation builds. Add a locked Sphinx dependency
  group and a Python API reference with a build script that treats warnings
  as errors.
- Remove plotting and random timing scripts, unused matrix fixtures, and
  washboard-specific examples. Keep separate Cargo and pytest suites that need
  no plotting dependencies or external data.
- Configure Ruff, nightly rustfmt, and Tombi, with Lefthook formatting jobs.
  Preserve Python docstring indentation while allowing Rust documentation to be
  formatted. Add scratch-file ignore rules.
- Configure locked release builds with Maturin, stub generation through uv,
  and generated Python extension stubs. Update the test dependency group and
  document how to build the extension, run tests, and generate documentation.
- Temporarily resolve LoPhat 0.14.0 from the sibling `../lophat` checkout while
  its release is unpublished. After publishing LoPhat, remove the dependency's
  `path` and regenerate the registry lockfile before standalone CI or release.

[Unreleased]: https://github.com/tomchaplin/phimaker/compare/v0.4.0...main
[0.4.0]: https://github.com/tomchaplin/phimaker/compare/v0.3.0...main
