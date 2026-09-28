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

Each diagram is a mapping from a generator's birth index to its death index.
In Rust, `PersistenceDiagram` wraps a `HashMap<usize, ExtendedUsize>`, with
`ExtendedUsize::Finite(index)` for finite deaths and `ExtendedUsize::Infinity`
for essential classes. Birth and death indices refer to filtration columns.

In Python, diagrams are dictionaries with integer keys and integer or `None`
values. For example, `diagrams.cod == {0: None, 1: 2}` describes an essential
class born at index 0 and a class born at index 1 that dies at index 2.
Use `diagram[birth]` to look up a death and `diagram.items()` to iterate over
intervals. The former `.paired` and `.unpaired` attributes have been removed.

## References

<a id="1">[1]</a>
Cohen-Steiner, D., Edelsbrunner, H., Harer, J. and Morozov, D., 2009, January.
Persistent homology for kernels, images, and cokernels.
In Proceedings of the twentieth annual ACM-SIAM symposium on Discrete algorithms (pp. 1011-1020).
Society for Industrial and Applied Mathematics.
