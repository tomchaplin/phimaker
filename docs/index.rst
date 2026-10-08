Python API reference
====================

Phimaker computes domain, codomain, image, kernel, cokernel, and relative
persistent homology over the field with two elements. Diagrams map birth
indices to death indices, with ``None`` for essential features.

For development setup and source builds, see
`the README <https://github.com/tomchaplin/phimaker#development-environment-and-source-builds>`_.
Build this reference from the repository root after installing the extension::

    uv sync --locked --group test --group docs
    uv run --no-sync maturin develop --locked --generate-stubs
    uv run --no-sync python docs/build.py

The API documentation below is read from the installed extension's docstrings.

.. automodule:: phimaker

Computing diagrams
------------------

.. autofunction:: phimaker.sixpack_from_inclusion

.. autofunction:: phimaker.sixpack

Persistence maps
----------------

.. autofunction:: phimaker.get_persistence_map

Results
-------

.. autoclass:: phimaker.DiagramEnsemble
   :members:

.. autoclass:: phimaker.CylinderMetadata
   :members:
