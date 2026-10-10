"""
Persistent homology of inclusions and filtered chain maps over F2.

Use ``sixpack_from_inclusion`` for a subcomplex specified by generator indices,
or ``sixpack`` for a general chain map. Results contain domain, codomain, image,
kernel, cokernel, and relative persistence diagrams.
"""

from _typeshed import Incomplete
from collections.abc import Sequence
from typing import final

@final
class CylinderMetadata:
	"""
	Coordinates, degrees, and entrance times returned by :py:func:`~phimaker.sixpack`.

	Diagram endpoints identify columns in a mapping cylinder. Look up ``times[index]``
	and ``degrees[index]`` to recover the entrance time and degree of a column.
	``domain_indices[i]`` and ``codomain_indices[i]`` locate the original input
	generators; ``domain_shift[i]`` locates the degree-one-higher copy of domain
	generator ``i``. For kernel intervals, subtract one from the birth-column degree.

	Attributes are read-only and return independent lists. Editing a returned list
	does not change subsequent accesses.

	Example::

	    >>> from phimaker import sixpack
	    >>> point = [(2.0, 0, [])]
	    >>> result, metadata = sixpack(point, point, [[0]])
	    >>> birth = metadata.domain_indices[0]
	    >>> assert metadata.times[birth] == 2.0
	    >>> assert metadata.degrees[birth] == 0
	"""
	@property
	def codomain_indices(self, /) -> list[int]:
		"""
		Cylinder column index for each original codomain generator, in input order.
		"""
	@property
	def degrees(self, /) -> list[int]:
		"""
		Degree of each cylinder generator. Kernel interval degree is one less than
		the degree at its birth index.
		"""
	@property
	def domain_indices(self, /) -> list[int]:
		"""
		Cylinder column index for each original domain generator, in input order.
		"""
	@property
	def domain_shift(self, /) -> list[int]:
		"""
		Cylinder column index of the degree-one-higher copy of each domain generator.
		"""
	@property
	def times(self, /) -> list[float]:
		"""
		Entrance time of each cylinder column. Index this list by a diagram endpoint.
		"""

@final
class DiagramEnsemble:
	"""
	Six persistence diagrams returned by the computation functions.

	The ``domain``, ``codomain``, ``image``, ``kernel``, ``cokernel``, and ``relative``
	attributes are dictionaries mapping birth indices to death indices. A death of
	``None`` identifies a feature persisting beyond the supplied filtration. Dictionary
	order is unspecified; use birth keys for lookup.

	For :py:func:`~phimaker.sixpack_from_inclusion`, endpoints refer to the supplied
	codomain generators. For :py:func:`~phimaker.sixpack`, use the returned metadata
	to recover filtration times and degrees. A kernel interval has the degree of
	its birth generator minus one; other intervals have the birth-generator degree.

	The ``domain_representatives``, ``codomain_representatives``,
	``image_representatives``, and ``kernel_representatives`` attributes are ``None`` unless
	``get_representatives=True`` was requested. When requested, they map birth
	indices to lists of generator indices. Summing the listed generators over the
	field with two elements gives a representative cycle. Keys and entries use the
	same coordinates as the diagrams.
	For domain, codomain, and image features, the maximum index in a representative
	equals the feature's birth. Kernel representatives use only domain generators,
	with indices strictly less than the kernel birth. Representatives need not be minimal or
	unique. Each representative dictionary has exactly its diagram's birth keys.
	A requested empty diagram has an empty representative dictionary. If extraction
	was skipped, its representative attribute is still ``None``.

	Attributes are read-only. Each dictionary access returns an independent copy,
	including the lists in representative dictionaries. Editing a copy does not
	change this result or later attribute accesses. Support order is unspecified.

	Example::

	    >>> from phimaker import sixpack_from_inclusion
	    >>> result = sixpack_from_inclusion([[]], [0], [0], get_representatives=True)
	    >>> assert result.domain == {0: None}
	    >>> assert result.domain_representatives == {0: [0]}
	    >>> assert result.kernel_representatives == {}
	    >>> diagram = result.domain
	    >>> diagram.clear()
	    >>> assert result.domain == {0: None}
	"""
	@property
	def codomain(self, /) -> dict[int, int | None]:
		"""
		Persistent homology of the codomain, as a birth-to-death dictionary.
		"""
	@property
	def codomain_representatives(self, /) -> dict[int, list[int]] | None:
		"""
		Codomain representatives, as a dictionary from birth indices to generator lists.

		Sum the listed generators over the field with two elements to obtain a cycle.
		Indices use the codomain diagram's coordinates. When requested, an empty codomain
		diagram has an empty dictionary. ``None`` means representative extraction was
		skipped.
		"""
	@property
	def cokernel(self, /) -> dict[int, int | None]:
		"""
		Cokernel of the induced homology map, as a birth-to-death dictionary.
		"""
	@property
	def domain(self, /) -> dict[int, int | None]:
		"""
		Persistent homology of the domain, as a birth-to-death dictionary.
		"""
	@property
	def domain_representatives(self, /) -> dict[int, list[int]] | None:
		"""
		Domain representatives, as a dictionary from birth indices to generator lists.

		Sum the listed generators over the field with two elements to obtain a cycle.
		Indices use the domain diagram's coordinates. When requested, an empty domain
		diagram has an empty dictionary. ``None`` means representative extraction was
		skipped.
		"""
	@property
	def image(self, /) -> dict[int, int | None]:
		"""
		Image of the induced homology map, as a birth-to-death dictionary.
		"""
	@property
	def image_representatives(self, /) -> dict[int, list[int]] | None:
		"""
		Image representatives, as a dictionary from birth indices to generator lists.

		Sum the listed generators over the field with two elements to obtain a cycle.
		Only domain generators occur, in the image diagram's coordinates. When requested,
		an empty image diagram has an empty dictionary. ``None`` means representative
		extraction was skipped.
		"""
	@property
	def kernel(self, /) -> dict[int, int | None]:
		"""
		Kernel of the induced homology map, as a birth-to-death dictionary.

		The degree of an interval is the degree of its birth generator minus one.
		"""
	@property
	def kernel_representatives(self, /) -> dict[int, list[int]] | None:
		"""
		Kernel representatives, as a dictionary from birth indices to generator lists.

		Sum the listed generators over the field with two elements to obtain a domain
		cycle that is zero in codomain homology. Supports use the diagram's ambient
		coordinates and contain only domain generators, with indices strictly less than
		the kernel birth. When requested, an empty kernel diagram has an empty dictionary.
		``None`` means representative extraction was skipped. Use these representatives
		with :py:func:`~phimaker.get_persistence_map` and the identity map to
		compute the kernel-to-domain inclusion.
		"""
	@property
	def relative(self, /) -> dict[int, int | None]:
		"""
		Homology of the quotient complex, as a birth-to-death dictionary.

		For results from :py:func:`~phimaker.sixpack`, this is the homology of the
		mapping cone.
		"""

def get_persistence_map(
	source: dict[int, int | None],
	source_representatives: dict[int, Sequence[int]],
	target: dict[int, int | None],
	target_representatives: dict[int, Sequence[int]],
	generator_map: dict[int, Sequence[int]],
) -> dict[int, list[int]]:
	"""
	Compute a persistence-module map over F2 in the supplied cycle bases.

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

	Representative dictionaries must be supplied explicitly; ``None`` is not accepted.
	When using a computation result, request ``get_representatives=True``.
	Representative dictionaries must have exactly their diagram's birth keys. Each
	support must be nonempty and use generator indices no greater than the feature's
	birth. A feature may be born after every generator in its representative; the
	maximum support index need not equal its birth. Finite deaths must be strictly
	later than births. Supports and individual mapping columns may be in
	any order. Repeated indices within a column identify the same nonzero entry and
	are ignored. Overlaps between distinct mapped columns cancel when added over F2.

	Representatives must be cycles and the generator map must preserve degree,
	filtration, and boundaries. Cyclehood, degrees, and the chain-map identity are
	not checked. Target representatives must be linearly independent as chain vectors
	and span the mapped source cycles; their pivots need not equal their birth labels.
	An empty source returns an empty dictionary without inspecting representatives.
	Target features dead at the source birth are omitted after cycle expansion.
	Every retained coefficient satisfies
	``target_birth <= source_birth < target_death <= source_death``, with ``None``
	interpreted as infinity. At a filtration index, use only intervals alive there.
	Target features can be born earlier than source features; equal births are not
	required. Only target features already born and still alive at the source birth
	can have nonzero coefficients.
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
	    >>> diagrams = sixpack_from_inclusion([[], [], [0, 1]], [0, 0, 1], [1], get_representatives=True)
	    >>> get_persistence_map(
	    ...     diagrams.domain, diagrams.domain_representatives,
	    ...     diagrams.codomain, diagrams.codomain_representatives, {1: [1]},
	    ... )
	    {1: [0, 1]}
	    >>> get_persistence_map(
	    ...     diagrams.domain, diagrams.domain_representatives,
	    ...     diagrams.image, diagrams.image_representatives, {1: [1]},
	    ... )
	    {1: [1]}
	    >>> get_persistence_map(
	    ...     {4: 5}, {4: [1, 2]}, {2: 5}, {2: [1, 2]}, {1: [1], 2: [2]},
	    ... )
	    {4: [2]}
	"""

def sixpack(
	domain_matrix: Sequence[tuple[float, int, Sequence[int]]],
	codomain_matrix: Sequence[tuple[float, int, Sequence[int]]],
	map: Sequence[Sequence[int]],
	get_representatives: bool = False,
	num_threads: int = 0,
	slow: bool = False,
) -> tuple[DiagramEnsemble, CylinderMetadata]:
	"""
	Compute six persistence diagrams for a filtered chain map over F2.

	:param domain_matrix: Domain columns as ``(entrance_time, degree, boundary)``
	    tuples. Each boundary lists nonzero indices in this matrix.
	:param codomain_matrix: Codomain columns in the same format; boundary indices
	    are local to the codomain matrix.
	:param map: Column ``i`` lists the codomain generators with coefficient one
	    in the image of domain generator ``i``. Supply at least one column per domain
	    generator; additional map columns are ignored.
	:param get_representatives: Whether to extract domain, codomain, image, and kernel cycles.
	    Defaults to ``False``, which skips extraction and leaves the four
	    representative attributes as ``None``. With ``True``, empty diagrams have
	    empty representative dictionaries.
	:param num_threads: Worker limit per computation step, default ``0``. Zero
	    selects the count automatically. The whole call may use more threads.
	:param slow: Whether to use temporary files to reduce memory use while computing.
	    Defaults to ``False``. This may be slower and does not bound memory use
	    during final diagram computation.
	:returns: ``(diagrams, metadata)``. Diagram attributes are birth-to-death
	    dictionaries, with ``None`` for features persisting beyond the supplied
	    filtration. Interval endpoints identify mapping-cylinder columns, rather than original
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
	    >>> assert metadata.times[birth] == 2.0
	"""

def sixpack_from_inclusion(
	boundary_matrix: Sequence[Sequence[int]],
	degrees: Sequence[int],
	cols_in_domain: Sequence[int],
	get_representatives: bool = False,
	num_threads: int = 0,
	slow: bool = False,
) -> DiagramEnsemble:
	"""
	Compute six persistence diagrams for an inclusion of filtered chain complexes over F2.

	:param boundary_matrix: Boundary of each codomain generator, as a list of its
	    nonzero row indices. Column order specifies filtration order; no filtration
	    values are accepted. Row indices may be in any order. Repeated listings
	    identify the same nonzero entry and are ignored, not added modulo two.
	:param degrees: Nonnegative degree of each codomain generator, one per column.
	:param cols_in_domain: Indices selecting the domain generators in the codomain.
	    Any order is accepted; repeated indices are ignored.
	:param get_representatives: Whether to extract domain, codomain, image, and kernel cycles.
	    Defaults to ``False``, which skips extraction and leaves the four
	    representative attributes as ``None``. With ``True``, empty diagrams have
	    empty representative dictionaries.
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
	    >>> assert diagrams.relative == {0: 2}
	"""

def __getattr__(name: str) -> Incomplete: ...
