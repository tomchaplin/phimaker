Python API reference
====================

Phimaker computes domain, codomain, image, kernel, cokernel, and relative
persistent homology over the field with two elements. Diagrams map birth
indices to death indices, with ``None`` for essential features.

Representative cycles are optional. The computation functions default to
``get_representatives=False``, returning ``None`` for the six representative
attributes. Pass ``get_representatives=True`` when cycles are needed. A requested
empty diagram has an empty representative dictionary.
Domain, codomain, image, cokernel, and relative cycles have maximum generator index equal to birth.
Kernel cycles use only domain generators preceding their birth and are boundaries
in the codomain. Cokernel representatives are codomain cycles considered modulo
codomain boundaries and domain cycles; finite features become zero in that
quotient at their deaths.
Relative representatives use only generators outside the domain. Their boundary
lies in the domain, so they are cycles in the quotient without necessarily being
codomain cycles. Their classes are interpreted modulo codomain boundaries and
domain chains.

The same flag computes ``kernel_to_domain``, ``domain_to_image``,
``image_to_codomain``, ``domain_to_codomain``, and ``codomain_to_cokernel`` in the
stored representative bases. Each dictionary has every source birth as a key
and a sorted list of target births with coefficient one; an empty list denotes
zero. An empty source gives an empty dictionary, while skipped extraction gives
``None``. Evaluate and compose maps using only intervals alive at the desired
index. For mapping-cylinder results, use the returned metadata to recover times.
``sixpack`` omits finite intervals whose birth and death times are equal, along
with their representative entries and persistence-map rows and columns.

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

Results
-------

.. autoclass:: phimaker.DiagramEnsemble
   :members:

.. autoclass:: phimaker.CylinderMetadata
   :members:
