<div align="center">

<h1>phimaker</h1>

<b>P</b>ersitent <b>h</b>omology of <b>im</b>ages <b>a</b>nd (co)<b>ker</b>nels.

</div>

## Overview

Phimaker is a Rust library implementing the algorithm introduced in [[1]](#1) for computing persistent homology for kernels, images and cokernels.
Python bindings are provided via PyO3.

Install via
```
pip install phimaker
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

## References

<a id="1">[1]</a>
Cohen-Steiner, D., Edelsbrunner, H., Harer, J. and Morozov, D., 2009, January.
Persistent homology for kernels, images, and cokernels.
In Proceedings of the twentieth annual ACM-SIAM symposium on Discrete algorithms (pp. 1011-1020).
Society for Industrial and Applied Mathematics.

## API documentation

Run `cargo doc --no-deps --open` to build the Rust API reference. The public
functions document input assumptions and index coordinates, with runnable Rust
examples. Inline `///` comments supply Rust documentation, while separate
feature-gated `#[doc = python_doc!(...)]` attributes supply Python docstrings.
Shared classes retain Rust comments for rustdoc and rust-analyzer; Python builds
strip those comments so only Python usage documentation appears in `help()`.
Use `help(phimaker.sixpack_from_inclusion)` or `help(phimaker.sixpack)` after
building/installing the extension for Python examples.

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

## Tests

Run `cargo test --locked` for the pure Rust suite. For Python, build the current
extension and run pytest as described in [tests/README.md](tests/README.md).
That document records each fixture's intent, including slow-mode serialization
regressions and executable documentation examples.

The Cargo dependency uses LoPhat version 0.12.0.
Building with `python-module` also enables `lophat/python-module`, providing
conversion of diagrams to dictionaries and infinite deaths to `None`.
