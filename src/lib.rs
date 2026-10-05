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
//! let diagrams =
//! 	sixpack_from_inclusion(&[vec![], vec![], vec![0, 1]], &[0, 0, 1], &[1], 1, false)?;
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
pub mod utils; // TODO: we can remove this module if we impl the right traits

use std::ops::Deref;

use cylinder::{CylinderMetadata, build_cylinder};
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
/// let diagrams =
/// 	sixpack_from_inclusion(&[vec![], vec![], vec![0, 1]], &[0, 0, 1], &[1], 1, false)?;
/// assert_eq!(diagrams.domain[&1], Infinity);
/// assert_eq!(diagrams.codomain[&1], Finite(2));
/// assert!(diagrams.kernel.is_empty());
/// # Ok::<(), phimaker::errors::InclusionError>(())
/// ```
pub fn sixpack_from_inclusion<UsizeSlice: Deref<Target = [usize]>>(
	boundary_matrix: &[UsizeSlice],
	degrees: &[usize],
	cols_in_domain: &[usize],
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
		.all_diagrams())
	} else {
		Ok(all_decompositions::<LockFreeAlgorithm<_>, _>(
			boundary_matrix,
			degrees,
			cols_in_domain,
			num_threads,
		)?
		.all_diagrams())
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
/// Returns diagrams and metadata in mapping-cylinder coordinates. Use
/// [`CylinderMetadata::times`] to translate endpoints to filtration times and
/// its index arrays to locate the original generators. Equal-time intervals
/// are retained. Relative means homology of the mapping cone. Kernel degree is
/// `metadata.degrees[birth] - 1`; other degrees are `metadata.degrees[birth]`.
/// Thread counts and `slow` have the same meaning as in
/// [`sixpack_from_inclusion`].
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
/// let (diagrams, metadata) = sixpack(&point, &point, &[vec![0]], 1, false)?;
/// let birth = metadata.domain_indices[0];
/// assert!(diagrams.domain.contains_key(&birth));
/// assert_eq!(metadata.times[birth], 0.0);
/// # Ok::<(), phimaker::errors::InclusionError>(())
/// ```
pub fn sixpack<UsizeSlice: Deref<Target = [usize]>>(
	domain_matrix: &[(f64, usize, UsizeSlice)],
	codomain_matrix: &[(f64, usize, UsizeSlice)],
	map: &[UsizeSlice],
	num_threads: usize,
	slow: bool,
) -> Result<(DiagramEnsemble, CylinderMetadata), InclusionError> {
	let (boundary_matrix, metadata) = build_cylinder(domain_matrix, codomain_matrix, map);
	let diagrams = sixpack_from_inclusion(
		&boundary_matrix,
		&metadata.degrees,
		&metadata.domain_indices,
		num_threads,
		slow,
	)?;
	Ok((diagrams, metadata))
}

#[cfg(test)]
mod tests;

#[cfg(all(feature = "python-module", not(any(doc, rust_analyzer))))]
mod python_bindings;
