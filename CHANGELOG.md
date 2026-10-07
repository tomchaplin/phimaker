# Changelog

This changelog records user-facing changes and development changes. New entries
belong under **Unreleased** until a version is released. Version 0.4.0 records
changes since `v0.3.0`.

## [Unreleased]

## [0.4.0] - 2026-10-07

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
- Replace `VectorMapping`, `IndexMapping`, annotated-column helpers, and row
  reordering abstractions with permutation-based index translation. Make matrix
  builders and individual reduction helpers internal implementation details.
- Remove `zero_overlap` and its implementation module, and remove the washboard
  submodule and Python interface.
- Require Python 3.10 or newer and move the Rust crate to edition 2024.

### Added

- `DiagramEnsemble.get_persistence_map()` returning sparse binary maps from domain
  features to codomain and image features. Solve in the representative cycle bases
  using private sparse F2 back substitution, then omit target features already dead
  at the source birth. Include typed `PersistenceMapError` failures in Rust and
  `RuntimeError` in Python, usage examples, and tests for both extraction modes.
- Domain, codomain, and image representative maps on `DiagramEnsemble`, keyed
  by birth index with cycle supports in diagram coordinates. Extract representatives
  in both normal and slow modes using LoPhat's `diagram_with_reps`; translate domain
  and image row indices back to original generator indices and use domain cycles
  for essential image features. Store representatives as plain maps, with empty
  maps for empty diagrams. Nonempty domain, codomain, and image decompositions
  must retain V for diagram extraction.
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

- Upgrade LoPhat from 0.11.0 to 0.13.0, use its shared diagram and extended-index
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
  reduction; only the relative reduction remains anti-transposed. Retaining V
  adds memory use and increases serialized codomain data size.
- Add GPL-3.0-or-later licensing, the license text, and package author,
  repository, and classification metadata. Update the Maturin CI action.

### Fixed

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

- Remove plotting and random timing scripts, unused matrix fixtures, and
  washboard-specific examples. Keep separate Cargo and pytest suites that need
  no plotting dependencies or external data.
- Configure Ruff, nightly rustfmt, and Tombi, with Lefthook formatting jobs.
  Preserve Python docstring indentation while allowing Rust documentation to be
  formatted. Add scratch-file ignore rules.
- Configure locked release builds with Maturin, stub generation through uv,
  and a Python extension stub. Update the test dependency group and document
  how to build the extension, run tests, and generate documentation.

[Unreleased]: https://github.com/tomchaplin/phimaker/compare/v0.4.0...dev
[0.4.0]: https://github.com/tomchaplin/phimaker/compare/v0.3.0...v0.4.0
