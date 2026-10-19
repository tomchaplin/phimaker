//! Persistent homology of filtered chain maps over F2.
//!
//! Use [`sixpack_from_inclusion`] for an inclusion or [`sixpack`] for a
//! filtered chain map. Both return the domain, codomain, image, kernel,
//! cokernel, and relative persistence diagrams. Diagram endpoints are generator
//! indices, with [`diagrams::ExtendedUsize::Infinity`] for essential classes.
//!
//! An inclusion is specified by the codomain boundary matrix, generator
//! degrees, and the indices selecting its domain. To supply a general map, give
//! the two timed boundary matrices and the images of the domain generators. The
//! returned [`CylinderMetadata`] translates the resulting indices to filtration
//! times.
//!
//! # Example
//!
//! ```
//! use phimaker::{diagrams::ExtendedUsize::Finite, sixpack_from_inclusion};
//!
//! let diagrams = sixpack_from_inclusion(
//! 	&[vec![], vec![], vec![0, 1]],
//! 	&[0, 0, 1],
//! 	&[1],
//! 	false,
//! 	1,
//! 	false,
//! )?;
//! assert_eq!(diagrams.codomain[&1], Finite(2));
//! # Ok::<(), phimaker::errors::InclusionError>(())
//! ```
#![recursion_limit = "1024"]
#![warn(missing_docs)]
#![allow(clippy::multiple_crate_versions)]

mod builders;
pub mod cylinder;
pub mod diagrams;
mod documentation;
pub mod ensemble;
pub mod errors;
pub mod indexing;
mod sparse;
pub mod utils; // TODO: we can remove this module if we impl the right traits

use std::ops::Deref;

use cylinder::{CylinderMetadata, build_cylinder, convert_representative_coordinates};
use diagrams::DiagramEnsemble;
pub use ensemble::{all_decompositions, all_decompositions_slow};
use errors::InclusionError;
use lophat::algorithms::LockFreeAlgorithm;

/// Compute six persistence diagrams for an inclusion A into B over F2.
///
/// `boundary_matrix[j]` lists the nonzero rows of column j of B. `degrees[j]`
/// is the degree of that generator, and `cols_in_domain` selects the generators
/// of A in B. Row lists and domain indices may have any order; repeated
/// listings identify the same nonzero entry and are ignored. Inputs are not
/// modified. Column order specifies filtration order; no filtration values are
/// accepted.
///
/// Returns all six diagrams in the original column indices of B. Essential
/// classes have death [`diagrams::ExtendedUsize::Infinity`]. The relative
/// diagram is H(B/A). Degree is `degrees[birth]`, except for kernel intervals,
/// whose degree is `degrees[birth] - 1`.
///
/// `num_threads` limits the workers for each reduction, so the whole call may
/// use more threads. Zero selects the count automatically. Set `slow` to store
/// intermediate results in temporary files. Diagram extraction still requires
/// memory for all six results. Empty inputs, domains, and quotients are
/// supported.
///
/// Set `get_representatives` to true to extract domain, codomain, image,
/// kernel, and cokernel cycles, relative chain pairs, and the five maps between
/// the first five modules. When false, all representative and morphism fields
/// are `None`, and extraction and map solving are skipped.
/// Relative representatives are pairs `(a, b)`: a domain cycle and a codomain
/// chain with `a = boundary(b)`. Both use original codomain indices; b uses
/// only generators outside A and need not be a cycle in B.
/// Requested empty diagrams have `Some` of
/// an empty map. Each morphism has exactly its source diagram's birth keys;
/// empty vectors denote zero. Reductions still retain the matrices used by
/// the six-diagram algorithm.
///
/// When representatives are requested, each domain, codomain, image, or
/// cokernel feature with birth index `b` has maximum representative
/// generator index equal to `b`: generator `b` occurs and no later generator
/// occurs. For a relative pair, this applies to its second component; its first
/// component uses only earlier domain generators. This guarantee holds for
/// every thread count. If you associate
/// nondecreasing filtration values with the input generator order, the
/// interval's birth value is therefore the maximum filtration value of a
/// generator in its representative, including when values are equal.
///
/// When `get_representatives` is true and `num_threads != 1`, representatives
/// may vary across calls with identical inputs because multithreaded reduction
/// is nondeterministic. This includes zero when it selects multiple workers.
///
/// With `num_threads = 1`, domain representatives depend only on the induced
/// ordered domain chain complex, up to an order-preserving relabelling of its
/// generators. Changing the codomain while preserving the domain boundaries,
/// degrees, and relative generator order does not change these representatives
/// after relabelling. Codomain representatives do not depend on the selected
/// domain. Use one worker when computing several domain/codomain combinations
/// whose representative bases need to be compatible.
///
/// # Errors
/// Returns [`InclusionError`] for mismatched degree counts, out-of-range domain
/// indices, boundary rows not less than their column, boundaries that do not
/// lower degree by one, or a domain not closed under boundary.
///
/// # Assumptions and panics
/// The boundary must square to zero; this is not checked. Otherwise results may
/// be incorrect or the call may panic. Computation or temporary-file failures
/// can also panic.
///
/// # Example
///
/// ```
/// use phimaker::{
/// 	diagrams::ExtendedUsize::{Finite, Infinity},
/// 	sixpack_from_inclusion,
/// };
///
/// let diagrams = sixpack_from_inclusion(
/// 	&[vec![], vec![], vec![0, 1]],
/// 	&[0, 0, 1],
/// 	&[1],
/// 	true,
/// 	1,
/// 	false,
/// )?;
/// assert_eq!(diagrams.domain[&1], Infinity);
/// assert_eq!(diagrams.codomain[&1], Finite(2));
/// assert!(diagrams.kernel.is_empty());
/// let cycle = &diagrams.codomain_representatives.as_ref().unwrap()[&1];
/// assert_eq!(cycle.iter().max(), Some(&1));
/// # Ok::<(), phimaker::errors::InclusionError>(())
/// ```
pub fn sixpack_from_inclusion<UsizeSlice: Deref<Target = [usize]>>(
	boundary_matrix: &[UsizeSlice],
	degrees: &[usize],
	cols_in_domain: &[usize],
	get_representatives: bool,
	num_threads: usize,
	slow: bool,
) -> Result<DiagramEnsemble, InclusionError> {
	if slow {
		Ok(all_decompositions_slow::<LockFreeAlgorithm<_>, _>(
			boundary_matrix,
			degrees,
			cols_in_domain,
			num_threads,
		)?
		.all_diagrams(get_representatives))
	} else {
		Ok(all_decompositions::<LockFreeAlgorithm<_>, _>(
			boundary_matrix,
			degrees,
			cols_in_domain,
			num_threads,
		)?
		.all_diagrams(get_representatives))
	}
}

/// Compute six persistence diagrams for a filtered chain map over F2.
///
/// Each input matrix contains `(entrance_time, degree, boundary_indices)`
/// columns. Boundary indices are local to that matrix. `map[i]` lists the
/// codomain generators with coefficient one in the image of domain generator i.
/// Supply at least one map column per domain generator; extras are ignored.
///
/// Both matrices must have nondecreasing, non-NaN times and sorted, distinct
/// boundary rows strictly less than their column index. Boundaries must lower
/// degree by one and square to zero. The map must preserve degree, commute with
/// boundary, and send each generator to generators entering no later than it.
/// Map columns must have sorted, distinct indices.
///
/// Diagram endpoints and representative keys identify mapping-cylinder columns.
/// Use [`CylinderMetadata::times`] to translate endpoints to filtration times
/// and its index arrays to locate the original generators in the cylinder.
/// Finite intervals with equal birth and death times are omitted, along with
/// their representatives and map rows and columns. Relative means homology of
/// the mapping cone. Kernel degree is `metadata.degrees[birth] - 1`; other
/// degrees are `metadata.degrees[birth]`. Thread counts, `slow`, and
/// `get_representatives` have the same meaning as in
/// [`sixpack_from_inclusion`].
///
/// Representative supports use original input coordinates: domain and kernel
/// cycles use domain indices; image, codomain, and cokernel cycles use codomain
/// indices. The input map is applied to image cycles. Relative representatives
/// are pairs `(a, b)` with original domain indices in a and original codomain
/// indices in b. For a degree-n feature, a has degree n-1 and b has degree n;
/// they satisfy `boundary(a) = 0` and `boundary(b) = map(a)`. Either component
/// may be empty. Persistence-map entries still identify intervals, not chain
/// generators; evaluate them using the intervals alive at the supplied times.
/// Representatives and maps are computed before filtering and converting
/// supports. Index comparisons between support entries and cylinder births
/// are not meaningful after conversion.
///
/// # Errors
/// Returns [`InclusionError`] for a structurally invalid cylinder inclusion.
/// This does not establish that the input satisfies the chain-map identity or
/// that its boundaries square to zero.
///
/// # Panics
/// Invalid boundary/image indices or missing map columns can panic. Other
/// violations of the algebraic preconditions can give incorrect results or
/// panic. Computation and temporary-file failures can also panic.
///
/// # Example
///
/// ```
/// use phimaker::sixpack;
///
/// let point = vec![(0.0, 0, vec![])];
/// let (diagrams, metadata) = sixpack(&point, &point, &[vec![0]], false, 1, false)?;
/// let birth = metadata.domain_indices[0];
/// assert!(diagrams.domain.contains_key(&birth));
/// assert_eq!(metadata.times[birth], 0.0);
/// # Ok::<(), phimaker::errors::InclusionError>(())
/// ```
///
/// A zero map to the empty codomain retains a domain component in its relative
/// representative, even though the codomain component is empty:
///
/// ```
/// use phimaker::sixpack;
/// let point = vec![(0.0, 0, vec![])];
/// let (diagrams, metadata) = sixpack(&point, &[], &[vec![]], true, 1, false)?;
/// let birth = metadata.domain_shift[0];
/// assert_eq!(
/// 	diagrams.relative_representatives.as_ref().unwrap()[&birth],
/// 	(vec![0], vec![])
/// );
/// # Ok::<(), phimaker::errors::InclusionError>(())
/// ```
pub fn sixpack<UsizeSlice: Deref<Target = [usize]>>(
	domain_matrix: &[(f64, usize, UsizeSlice)],
	codomain_matrix: &[(f64, usize, UsizeSlice)],
	map: &[UsizeSlice],
	get_representatives: bool,
	num_threads: usize,
	slow: bool,
) -> Result<(DiagramEnsemble, CylinderMetadata), InclusionError> {
	let (boundary_matrix, metadata) = build_cylinder(domain_matrix, codomain_matrix, map);
	let mut diagrams = sixpack_from_inclusion(
		&boundary_matrix,
		&metadata.degrees,
		&metadata.domain_indices,
		get_representatives,
		num_threads,
		slow,
	)?;
	// Solve in the full cylinder cycle bases before discarding intervals that
	// vanish in the supplied filtration times. Their cycles can be necessary
	// for the intermediate chain-coordinate equations.
	diagrams.remove_equal_time_intervals(&metadata.times);
	if get_representatives {
		convert_representative_coordinates(&mut diagrams, &metadata, map);
	}
	Ok((diagrams, metadata))
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod cylinder_result_tests;

#[cfg(test)]
mod input_coordinate_tests;

#[cfg(test)]
mod persistence_map_tests;

#[cfg(test)]
mod cokernel_tests;

#[cfg(test)]
mod ensemble_map_tests;

#[cfg(test)]
mod relative_tests;

#[cfg(feature = "python-module")]
mod python_bindings;
