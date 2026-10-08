use std::collections::HashMap;

use pyo3::{
	exceptions::{PyRuntimeError, PyValueError},
	prelude::*,
};

use crate::{
	cylinder::CylinderMetadata,
	diagrams::{DiagramEnsemble, PersistenceDiagram, PersistenceMap},
	documentation::python_doc,
	get_persistence_map,
	sixpack,
	sixpack_from_inclusion,
};

#[doc = python_doc!(
r#"Compute a persistence-module map over F2 in the supplied cycle bases.

:param source: Source birth-to-death dictionary; ``None`` denotes an essential death.
:param source_representatives: Source births mapped to lists of cycle generator indices.
:param target: Target birth-to-death dictionary, using the same endpoint convention.
:param target_representatives: Target births mapped to lists of cycle generator indices.
:param generator_map: Source generator indices mapped to lists of target generators
    with coefficient one. Missing entries implicitly map to zero; an empty list
    also denotes zero. Entries unused by source representatives are ignored.
:returns: A dictionary mapping every source birth to a sorted list of target
    births with coefficient one. An empty list denotes zero. Each call returns
    an independent dictionary and lists; inputs are not modified.

All source and target births and deaths must use one common ordered filtration
index scale. Independently numbered complexes must be reindexed before calling;
a generator map alone does not establish how their filtration indices compare.
This requirement cannot be checked from these inputs. Source representatives
and map keys use the source generator basis; target representatives and map
values use the target generator basis.

Representative dictionaries must have exactly their diagram's birth keys. Each
support must have maximum index equal to its birth, and finite deaths must be
strictly later than births. Supports and individual mapping columns may be in
any order. Repeated indices within a column identify the same nonzero entry and
are ignored. Overlaps between distinct mapped columns cancel when added over F2.

Representatives must be cycles and the generator map must preserve degree,
filtration, and boundaries. Cyclehood, degrees, and the chain-map identity are
not checked. Mapped cycles must lie in the span of target representatives.
An empty source returns an empty dictionary without inspecting representatives.
Target features dead at the source birth are omitted after cycle expansion.
Every retained coefficient satisfies
``target_birth <= source_birth < target_death <= source_death``, with ``None``
interpreted as infinity. At a filtration index, use only intervals alive there.
The call blocks until results are ready; other Python threads can run meanwhile.

:raises RuntimeError: If representatives are unavailable or inconsistent, a
    mapped cycle cannot be expressed in the target basis, or a nonzero
    coefficient has incompatible interval lifetimes.
:raises TypeError: If an argument has an incompatible type or a diagram has an
    invalid key or endpoint. Diagram conversion retains the underlying error
    as its cause.
:raises OverflowError: If a representative or mapping index is negative or
    exceeds the supported integer range.

Example::

    >>> from phimaker import get_persistence_map, sixpack_from_inclusion
    >>> diagrams = sixpack_from_inclusion([[], [], [0, 1]], [0, 0, 1], [1])
    >>> get_persistence_map(
    ...     diagrams.domain, diagrams.domain_representatives,
    ...     diagrams.codomain, diagrams.codomain_representatives, {1: [1]},
    ... )
    {1: [0, 1]}
    >>> get_persistence_map(
    ...     diagrams.domain, diagrams.domain_representatives,
    ...     diagrams.image, diagrams.image_representatives, {1: [1]},
    ... )
    {1: [1]}"#
)]
#[pyfunction(name = "get_persistence_map")]
fn py_get_persistence_map(
	py: Python<'_>,
	source: PersistenceDiagram,
	source_representatives: HashMap<usize, Vec<usize>>,
	target: PersistenceDiagram,
	target_representatives: HashMap<usize, Vec<usize>>,
	generator_map: HashMap<usize, Vec<usize>>,
) -> PyResult<PersistenceMap> {
	py.detach(|| {
		get_persistence_map(
			&source,
			&source_representatives,
			&target,
			&target_representatives,
			&generator_map,
		)
	})
	.map_err(|error| PyRuntimeError::new_err(error.to_string()))
}
#[doc = python_doc!(
r#"Compute six persistence diagrams for an inclusion of filtered chain complexes over F2.

:param boundary_matrix: Boundary of each codomain generator, as a list of its
    nonzero row indices. Column order specifies filtration order; no filtration
    values are accepted. Row indices may be in any order. Repeated listings
    identify the same nonzero entry and are ignored, not added modulo two.
:param degrees: Nonnegative degree of each codomain generator, one per column.
:param cols_in_domain: Indices selecting the domain generators in the codomain.
    Any order is accepted; repeated indices are ignored.
:param num_threads: Worker limit per computation step, default ``0``. Zero
    selects the count automatically. The whole call may use more threads.
:param slow: Whether to use temporary files to reduce memory use while computing.
    Defaults to ``False``. This may be slower and does not bound memory use
    during final diagram computation.
:returns: A result with ``domain``, ``codomain``, ``image``, ``kernel``,
    ``cokernel``, and ``relative`` dictionary attributes. Each maps birth indices
    to death indices, with ``None`` for features persisting beyond the supplied
    filtration. All endpoints refer to original codomain generators.

Boundary rows must precede their column and lower degree by exactly one. The
selected domain must be closed under boundary. These conditions, index ranges,
and the degree-list length are checked. The boundary must also square to zero;
this is not checked, and violations can give incorrect diagrams even if the
call succeeds. Kernel interval degree is ``degrees[birth] - 1``; other interval
degrees are ``degrees[birth]``. Relative means homology of the quotient complex.

The input lists are not modified. Each dictionary access returns a fresh copy,
so editing it does not change the result. Dictionary order is unspecified.
Empty complexes, empty domains, and identity inclusions work in both modes.
The call blocks until results are ready; other Python threads can run meanwhile.

.. rubric:: Raises

:py:exc:`ValueError`
    If degree counts differ from the column count, a domain index is out of
    range, a boundary row does not precede its column, a boundary fails to lower
    degree by one, or the domain is not closed under boundary.
:py:exc:`TypeError`
    If an argument or an input element has an incompatible type.
:py:exc:`OverflowError`
    If a degree, index, or thread count is negative or exceeds the supported range.
``pyo3_runtime.PanicException``
    May occur if unchecked algebraic requirements are violated or computation fails.

Example::

    >>> from phimaker import sixpack_from_inclusion
    >>> diagrams = sixpack_from_inclusion([[], [], [0, 1]], [0, 0, 1], [1])
    >>> assert diagrams.domain == {1: None}
    >>> assert diagrams.codomain == {0: None, 1: 2}
    >>> assert diagrams.relative == {0: 2}"#
)]
#[pyfunction(name = "sixpack_from_inclusion")]
#[pyo3(signature = (boundary_matrix, degrees, cols_in_domain, num_threads=0, slow=false))]
fn py_sixpack_from_inclusion(
	py: Python<'_>,
	boundary_matrix: Vec<Vec<usize>>,
	degrees: Vec<usize>,
	cols_in_domain: Vec<usize>,
	num_threads: usize,
	slow: bool,
) -> PyResult<DiagramEnsemble> {
	py.detach(|| {
		sixpack_from_inclusion(
			&boundary_matrix,
			&degrees,
			&cols_in_domain,
			num_threads,
			slow,
		)
	})
	.map_err(|error| PyValueError::new_err(error.to_string()))
}

#[doc = python_doc!(
r#"Compute six persistence diagrams for a filtered chain map over F2.

:param domain_matrix: Domain columns as ``(entrance_time, degree, boundary)``
    tuples. Each boundary lists nonzero indices in this matrix.
:param codomain_matrix: Codomain columns in the same format; boundary indices
    are local to the codomain matrix.
:param map: Column ``i`` lists the codomain generators with coefficient one
    in the image of domain generator ``i``. Supply at least one column per domain
    generator; additional map columns are ignored.
:param num_threads: Worker limit per computation step, default ``0``. Zero
    selects the count automatically. The whole call may use more threads.
:param slow: Whether to use temporary files to reduce memory use while computing.
    Defaults to ``False``. This may be slower and does not bound memory use
    during final diagram computation.
:returns: ``(diagrams, metadata)``. Diagram attributes are birth-to-death
    dictionaries, with ``None`` for features persisting beyond the supplied
    filtration. Endpoints identify mapping-cylinder columns, rather than original
    input columns. Use ``metadata.times[index]`` to recover filtration times.
    ``metadata.domain_indices`` and ``metadata.codomain_indices`` locate the
    original generators. Intervals with equal birth and death times are retained.

Both matrices must be ordered by nondecreasing, non-NaN entrance time. Boundary
and map columns must list sorted, distinct indices. Boundary rows must precede
their column, lower degree by one, and square to zero. The map must preserve
degree and commute with boundary. Every image generator must enter no later than
its source generator. These requirements are not all checked; violations can
give incorrect diagrams even when the call succeeds.

Equal-time cylinder columns are ordered domain, codomain, then shifted domain,
preserving each input order. Kernel interval degree is
``metadata.degrees[birth] - 1``; other degrees are ``metadata.degrees[birth]``.
Relative means homology of the mapping cone.

The input lists are not modified. Result dictionaries and metadata lists are
independent copies on each access. Empty inputs are supported in both modes.
The call blocks until results are ready; other Python threads can run meanwhile.

.. rubric:: Raises

:py:exc:`ValueError`
    If the resulting inclusion has invalid index ranges, boundary order, degrees,
    or domain closure.
:py:exc:`TypeError`
    If an argument or an input element has an incompatible type.
:py:exc:`OverflowError`
    If a degree, index, or thread count is negative or exceeds the supported range.
``pyo3_runtime.PanicException``
    May occur for unavailable boundary or image indices, missing map columns,
    violated algebraic requirements, or computation failures.

Example::

    >>> from phimaker import sixpack
    >>> point = [(2.0, 0, [])]
    >>> diagrams, metadata = sixpack(point, point, [[0]])
    >>> birth = metadata.domain_indices[0]
    >>> assert diagrams.domain[birth] is None
    >>> assert metadata.times[birth] == 2.0"#
)]
#[pyfunction(name = "sixpack")]
#[pyo3(signature = (domain_matrix, codomain_matrix, map, num_threads=0, slow=false))]
fn py_sixpack(
	py: Python<'_>,
	domain_matrix: Vec<(f64, usize, Vec<usize>)>,
	codomain_matrix: Vec<(f64, usize, Vec<usize>)>,
	map: Vec<Vec<usize>>,
	num_threads: usize,
	slow: bool,
) -> PyResult<(DiagramEnsemble, CylinderMetadata)> {
	py.detach(|| sixpack(&domain_matrix, &codomain_matrix, &map, num_threads, slow))
		.map_err(|error| PyValueError::new_err(error.to_string()))
}

#[rustfmt::skip::attributes(doc)]
#[doc = crate::documentation::python_doc!(
r#"Persistent homology of inclusions and filtered chain maps over F2.

Use ``sixpack_from_inclusion`` for a subcomplex specified by generator indices,
or ``sixpack`` for a general chain map. Results contain domain, codomain, image,
kernel, cokernel, and relative persistence diagrams."#
)]
#[pymodule]
mod phimaker {
	use pyo3::prelude::*;

	#[pymodule_export]
	use super::{py_get_persistence_map, py_sixpack, py_sixpack_from_inclusion};
	#[pymodule_export]
	use crate::{cylinder::CylinderMetadata, diagrams::DiagramEnsemble};

	#[pymodule_init]
	fn init(_module: &Bound<'_, PyModule>) -> PyResult<()> {
		pyo3_log::init();
		Ok(())
	}
}
