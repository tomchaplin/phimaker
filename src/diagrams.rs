//! Birth-to-death dictionaries and extraction from reduced matrices.

use std::{
	collections::{HashMap, HashSet},
	fs::File,
	io::BufReader,
};

use ExtendedUsize::{Finite, Infinity};
use bincode::deserialize_from;
/// Shared birth-to-death diagrams and extended indices from LoPhat.
pub use lophat::utils::{ExtendedUsize, PersistenceDiagram};
use lophat::{
	algorithms::{Decomposition, DecompositionAlgo},
	columns::Column,
	utils::DecompositionFileFormat,
};
#[cfg(feature = "python-module")]
use pyo3::prelude::*;

#[cfg(all(feature = "python-module", not(any(doc, rust_analyzer))))]
use crate::documentation::{macro_rules_apply, strip_rust_docs};
use crate::{
	documentation::python_doc,
	ensemble::{DecompositionEnsemble, EnsembleMetadata, FileEnsemble},
	errors::PersistenceMapError,
	indexing::Permutation,
};

/// Six persistence diagrams for an inclusion or filtered chain map over F2.
///
/// Each diagram maps birth indices to death indices or
/// [`ExtendedUsize::Infinity`]. Use the birth key for lookup or iterate over
/// the map; iteration order is unspecified. Essential classes persist beyond
/// the supplied filtration.
///
/// Results from [`crate::sixpack_from_inclusion`] use original codomain
/// indices. Results from [`crate::sixpack`] use mapping-cylinder indices; its
/// metadata provides filtration times and degrees. The degree of a kernel
/// interval is the birth-generator degree minus one; other interval degrees
/// equal that degree.
///
/// The domain, codomain, image, and kernel representative maps associate each
/// birth with the generator indices whose sum is its cycle over F2 when
/// requested. They are `None` when `get_representatives` is false. Their keys
/// and entries use the same coordinates as the diagrams. Domain, codomain, and
/// image representatives have maximum generator index equal to their birth.
/// Kernel representatives contain only domain generators with indices strictly
/// less than their kernel birth. Representatives are not necessarily minimal or
/// unique, and support order is unspecified. Each representative map has
/// exactly its diagram's birth keys; a requested empty diagram has `Some` of
/// an empty map. Skipping extraction still returns `None` for empty diagrams.
///
/// # Example
///
/// ```
/// use phimaker::{diagrams::ExtendedUsize::Infinity, sixpack_from_inclusion};
///
/// let diagrams = sixpack_from_inclusion(&[vec![]], &[0], &[0], true, 1, false)?;
/// assert_eq!(diagrams.domain[&0], Infinity);
/// assert_eq!(
/// 	diagrams.domain_representatives.as_ref().unwrap()[&0],
/// 	vec![0]
/// );
/// assert!(diagrams.relative.is_empty());
/// # Ok::<(), phimaker::errors::InclusionError>(())
/// ```
#[doc = python_doc!(
r#"Six persistence diagrams returned by the computation functions.

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
    >>> assert result.domain == {0: None}"#
)]
#[cfg_attr(
    all(feature = "python-module", not(any(doc, rust_analyzer))),
    macro_rules_apply(strip_rust_docs!)
)]
#[cfg_attr(
	feature = "python-module",
	pyclass(module = "phimaker", get_all, from_py_object)
)]
#[derive(Debug, Clone)]
pub struct DiagramEnsemble {
	/// Persistent homology of the codomain.
	#[doc = python_doc!(
    r#"Persistent homology of the codomain, as a birth-to-death dictionary."#
    )]
	pub codomain: PersistenceDiagram,

	/// Codomain birth indices mapped to cycle supports in diagram coordinates.
	/// Each listed generator has coefficient one over F2. An empty codomain
	/// diagram has `Some` of an empty map when requested, or `None` when
	/// representative extraction is disabled.
	#[doc = python_doc!(
    r#"Codomain representatives, as a dictionary from birth indices to generator lists.

Sum the listed generators over the field with two elements to obtain a cycle.
Indices use the codomain diagram's coordinates. When requested, an empty codomain
diagram has an empty dictionary. ``None`` means representative extraction was
skipped."#
    )]
	pub codomain_representatives: Option<HashMap<usize, Vec<usize>>>,

	/// Persistent homology of the domain.
	#[doc = python_doc!(
    r#"Persistent homology of the domain, as a birth-to-death dictionary."#
    )]
	pub domain: PersistenceDiagram,

	/// Domain birth indices mapped to cycle supports in diagram coordinates.
	/// Each listed generator has coefficient one over F2. An empty domain
	/// diagram has `Some` of an empty map when requested, or `None` when
	/// representative extraction is disabled.
	#[doc = python_doc!(
    r#"Domain representatives, as a dictionary from birth indices to generator lists.

Sum the listed generators over the field with two elements to obtain a cycle.
Indices use the domain diagram's coordinates. When requested, an empty domain
diagram has an empty dictionary. ``None`` means representative extraction was
skipped."#
    )]
	pub domain_representatives: Option<HashMap<usize, Vec<usize>>>,

	/// Image of the induced homology map.
	#[doc = python_doc!(
    r#"Image of the induced homology map, as a birth-to-death dictionary."#
    )]
	pub image: PersistenceDiagram,

	/// Image birth indices mapped to cycle supports in diagram coordinates.
	/// Supports contain only domain generators, each with coefficient one over
	/// F2. A requested empty image diagram has `Some` of an empty map;
	/// `None` means representative extraction was disabled.
	#[doc = python_doc!(
    r#"Image representatives, as a dictionary from birth indices to generator lists.

Sum the listed generators over the field with two elements to obtain a cycle.
Only domain generators occur, in the image diagram's coordinates. When requested,
an empty image diagram has an empty dictionary. ``None`` means representative
extraction was skipped."#
    )]
	pub image_representatives: Option<HashMap<usize, Vec<usize>>>,

	/// Kernel of the induced homology map. Birth indices identify killing
	/// cells, so the interval degree is the birth-cell degree minus one.
	#[doc = python_doc!(
    r#"Kernel of the induced homology map, as a birth-to-death dictionary.

The degree of an interval is the degree of its birth generator minus one."#
    )]
	pub kernel: PersistenceDiagram,

	/// Kernel birth indices mapped to cycles supported in the domain.
	/// Supports use the diagram's ambient generator coordinates and contain
	/// only indices strictly less than the kernel birth. Each listed generator
	/// has coefficient one over F2. A requested empty kernel diagram has `Some`
	/// of an empty map; `None` means representative extraction was disabled.
	#[doc = python_doc!(
    r#"Kernel representatives, as a dictionary from birth indices to generator lists.

Sum the listed generators over the field with two elements to obtain a domain
cycle that is zero in codomain homology. Supports use the diagram's ambient
coordinates and contain only domain generators, with indices strictly less than
the kernel birth. When requested, an empty kernel diagram has an empty dictionary.
``None`` means representative extraction was skipped."#
    )]
	pub kernel_representatives: Option<HashMap<usize, Vec<usize>>>,

	/// Cokernel of the induced homology map.
	#[doc = python_doc!(
    r#"Cokernel of the induced homology map, as a birth-to-death dictionary."#
    )]
	pub cokernel: PersistenceDiagram,

	/// Homology of the quotient complex; for a general map, homology of its
	/// cone.
	#[doc = python_doc!(
    r#"Homology of the quotient complex, as a birth-to-death dictionary.

For results from :py:func:`~phimaker.sixpack`, this is the homology of the
mapping cone."#
    )]
	pub relative: PersistenceDiagram,
}

// Sum of two vectors over F2.
// `left` and `right` specify the supports of the two vectors
// i.e., the indices of non-zero coordinate entries.
// This is identical to taking the symmetric difference of
// the set of supports of the two vectors.
fn sum_mod2(left: &[usize], right: &[usize]) -> Vec<usize> {
	let mut out = Vec::with_capacity(left.len() + right.len());
	let (mut i, mut j) = (0, 0);
	while i < left.len() && j < right.len() {
		match left[i].cmp(&right[j]) {
			std::cmp::Ordering::Less => {
				out.push(left[i]);
				i += 1;
			},
			std::cmp::Ordering::Greater => {
				out.push(right[j]);
				j += 1;
			},
			std::cmp::Ordering::Equal => {
				i += 1;
				j += 1;
			},
		}
	}
	out.extend_from_slice(&left[i..]);
	out.extend_from_slice(&right[j..]);
	out
}

// Solve A x = b over F2 for a sparse basis with distinct pivots.
// Each column of A and the vector b are specified by their support (indices of
// non-zero entries). A must be supplied as a mapping 'pivot index -> support
// of the column with that pivot. All supports, including b, must be sorted
// and duplicate-free. Back substitution visits strictly decreasing pivots.
// Return None if b is outside the span; otherwise return the column keys of x.
fn solve(a: &HashMap<usize, Vec<usize>>, b: &[usize]) -> Option<Vec<usize>> {
	let mut residual = b.to_vec();
	let mut solution = Vec::new();
	while let Some(&pivot) = residual.last() {
		let column = a.get(&pivot)?;
		residual = sum_mod2(&residual, column);
		solution.push(pivot);
	}
	solution.reverse();
	Some(solution)
}

type PersistenceRepresentatives = HashMap<usize, Vec<usize>>;

// Make sure that the support of a homology representative is sorted and
// duplicate-free. Verify that the support of a representative is born at or
// before the birth time of the feature itself.
//
// A feature may be born after every generator in its cycle. For example,
// a kernel feature is represented by a cycle in the domain, but is only born
// when that cycle dies in the codomain.
//
// This function does not check whether representatives form cycles.
fn prepare_representatives(
	diagram: &PersistenceDiagram,
	representatives: &HashMap<usize, Vec<usize>>,
	name: &'static str,
) -> Result<PersistenceRepresentatives, PersistenceMapError> {
	if !diagram.is_empty() && representatives.is_empty() {
		return Err(PersistenceMapError::MissingRepresentatives { diagram: name });
	}
	let inconsistent = || PersistenceMapError::InconsistentData { diagram: name };
	if representatives.len() != diagram.len() {
		return Err(inconsistent());
	}
	representatives
		.iter()
		.map(|(&birth, support)| {
			let death = diagram.get(&birth).ok_or_else(inconsistent)?;
			let mut support = support.clone();
			support.sort_unstable();
			support.dedup();
			if *death <= Finite(birth) || support.is_empty() || support.last() > Some(&birth) {
				return Err(inconsistent());
			}
			Ok((birth, support))
		})
		.collect()
}

// Given an ordered set of linearly independent representative cycles as columns
// of an upper-triangular matrix M, run left-to-right column operations on M so
// that the map taking a column to its pivot is injective. This gives a new set
// of representatives. Returns both the new set of representative cycles, and
// the expansion of each new representative cycle in terms of the original set.
// This does NOT compute an automorphism of a persistence module; it only
// computes an automorphism of the cycle space.
fn prepare_target_basis(
	representatives: &HashMap<usize, Vec<usize>>,
) -> Result<(PersistenceRepresentatives, PersistenceRepresentatives), PersistenceMapError> {
	let mut ordered: Vec<_> = representatives.iter().collect();
	ordered.sort_unstable_by_key(|&(birth, _)| *birth);
	let mut new_representatives: HashMap<usize, Vec<usize>> = HashMap::new();
	let mut new_to_old_mapping: HashMap<usize, Vec<usize>> = HashMap::new();
	for (&birth, support) in ordered {
		let mut residual = support.clone();
		let mut coefficients = vec![birth];
		loop {
			let pivot = *residual
				.last()
				.ok_or(PersistenceMapError::InconsistentData { diagram: "target" })?;
			if let Some(column) = new_representatives.get(&pivot) {
				residual = sum_mod2(&residual, column);
				coefficients = sum_mod2(&coefficients, &new_to_old_mapping[&pivot]);
			} else {
				new_representatives.insert(pivot, residual);
				new_to_old_mapping.insert(pivot, coefficients);
				break;
			}
		}
	}
	Ok((new_representatives, new_to_old_mapping))
}

// Source birth indices mapped to target births with coefficient one over F2.
pub(crate) type PersistenceMap = HashMap<usize, Vec<usize>>;

// Expand mapped cycles exactly in the complete target cycle basis, restore the
// original feature labels, then discard targets dead at each source birth.
// The supplied representatives must span the mapped cycles as chain vectors;
// this does not solve modulo an independently supplied boundary space.
// Retained privately until maps are computed during ensemble extraction.
#[allow(
	dead_code,
	reason = "Retained for the upcoming ensemble map extraction"
)]
pub(crate) fn get_persistence_map(
	source: &PersistenceDiagram,
	source_representatives: &HashMap<usize, Vec<usize>>,
	target: &PersistenceDiagram,
	target_representatives: &HashMap<usize, Vec<usize>>,
	generator_map: &HashMap<usize, Vec<usize>>,
) -> Result<PersistenceMap, PersistenceMapError> {
	if source.is_empty() {
		return Ok(HashMap::new());
	}
	let source_reps = prepare_representatives(source, source_representatives, "source")?;
	let target_reps = prepare_representatives(target, target_representatives, "target")?;
	let (basis, features) = prepare_target_basis(&target_reps)?;
	// Normalize the map of generators by sorting and de-duplicating the
	// supports.
	let mut columns = HashMap::new();
	for &generator in source_reps.values().flatten() {
		if let Some(image) = generator_map.get(&generator) {
			columns.entry(generator).or_insert_with(|| {
				let mut image = image.clone();
				image.sort_unstable();
				image.dedup();
				image
			});
		}
	}
	source_reps
		.iter()
		.map(|(&birth, support)| {
			let mut mapped = HashSet::new();
			for generator in support {
				if let Some(image) = columns.get(generator) {
					for &index in image {
						if !mapped.insert(index) {
							mapped.remove(&index);
						}
					}
				}
			}
			let mut mapped: Vec<_> = mapped.into_iter().collect();
			mapped.sort_unstable();
			let pivots = solve(&basis, &mapped).ok_or(PersistenceMapError::Unsolvable {
				target: "target",
				birth,
			})?;
			let mut coefficients = Vec::new();
			for pivot in pivots {
				coefficients = sum_mod2(&coefficients, &features[&pivot]);
			}
			coefficients.retain(|target_birth| target[target_birth] > Finite(birth));
			for &target_birth in &coefficients {
				if target_birth > birth || target[&target_birth] > source[&birth] {
					return Err(PersistenceMapError::IncompatibleIntervals {
						target: "target",
						birth,
						target_birth,
					});
				}
			}
			Ok((birth, coefficients))
		})
		.collect()
}

fn is_kernel_birth<Decomp: Decomposition<C>, C: Column>(
	idx: usize,
	metadata: &EnsembleMetadata,
	d_im_decomp: &Decomp,
) -> bool {
	let in_dom = metadata.dom_first_permutation.map(idx) < metadata.sz_domain;
	if in_dom {
		return false;
	}
	// Zero/nonzero column status agrees with the codomain reduction.
	let negative_in_cod = d_im_decomp.get_r_col(idx).is_boundary();
	if !negative_in_cod {
		return false;
	}
	let low_idx_in_dom = d_im_decomp.get_r_col(idx).pivot().unwrap() < metadata.sz_domain;
	if !low_idx_in_dom {
		return false;
	}
	true
}

fn is_kernel_death<Decomp: Decomposition<C>, C: Column>(
	idx: usize,
	metadata: &EnsembleMetadata,
	d_dom_decomp: &Decomp,
	d_im_decomp: &Decomp,
) -> bool {
	let in_dom = metadata.dom_first_permutation.map(idx) < metadata.sz_domain;
	if !in_dom {
		return false;
	}
	let dom_idx = metadata.dom_first_permutation.map(idx);
	let negative_in_dom = d_dom_decomp.get_r_col(dom_idx).is_boundary();
	if !negative_in_dom {
		return false;
	}
	// Zero/nonzero column status agrees with the codomain reduction.
	let negative_in_cod = d_im_decomp.get_r_col(idx).is_boundary();
	if negative_in_cod {
		return false;
	}
	true
}

fn kernel_diagram<Decomp: Decomposition<C>, C: Column>(
	metadata: &EnsembleMetadata,
	ker: &Decomp,
	d_dom_decomp: &Decomp,
	d_im_decomp: &Decomp,
) -> PersistenceDiagram {
	let mut dgm = PersistenceDiagram::default();
	for idx in 0..metadata.sz_codomain {
		if is_kernel_birth(idx, metadata, d_im_decomp) {
			dgm.insert(idx, Infinity);
			continue;
		}
		if is_kernel_death(idx, metadata, d_dom_decomp, d_im_decomp) {
			// TODO: Problem kernel columns have different indexing to f
			let ker_idx = *metadata.kernel_mapping.get(&idx).unwrap();
			let dom_birth_index = ker.get_r_col(ker_idx).pivot().unwrap();
			let birth_index = metadata.dom_first_permutation.inverse_map(dom_birth_index);
			dgm.insert(birth_index, Finite(idx));
		}
	}
	dgm
}

// Apply the original domain boundary using D = R V^{-1}, without rebuilding D
// or computing an inverse. V is upper triangular with diagonal entries one.
fn boundary_of_chain<Decomp: Decomposition<C>, C: Column>(
	decomposition: &Decomp,
	mut support: Vec<usize>,
) -> Vec<usize> {
	let mut boundary = Vec::new();
	while let Some(&pivot) = support.last() {
		let v = decomposition
			.get_v_col(pivot)
			.expect("Kernel representative extraction requires retained domain V");
		support = sum_mod2(&support, &v.entries().collect::<Vec<_>>());
		boundary = sum_mod2(
			&boundary,
			&decomposition.get_r_col(pivot).entries().collect::<Vec<_>>(),
		);
	}
	boundary
}

fn kernel_representatives<Decomp: Decomposition<C>, C: Column>(
	metadata: &EnsembleMetadata,
	diagram: &PersistenceDiagram,
	ker: &Decomp,
	d_dom_decomp: &Decomp,
	d_im_decomp: &Decomp,
) -> HashMap<usize, Vec<usize>> {
	diagram
		.iter()
		.map(|(&birth, &death)| {
			let support = match death {
				Infinity => d_im_decomp.get_r_col(birth).entries().collect::<Vec<_>>(),
				Finite(death) => {
					// The kernel column z = a + w is a codomain cycle, with
					// domain part a and outside part w. The pivot of w is the
					// kernel birth, so w is available then. Since D z = 0,
					// D a = D w is a domain cycle available at birth that
					// bounds in the domain at death. Using the death column
					// preserves the kernel interval basis.
					let ker_idx = metadata.kernel_mapping[&death];
					let domain_part = ker
						.get_r_col(ker_idx)
						.entries()
						.filter(|&idx| idx < metadata.sz_domain)
						.collect();
					boundary_of_chain(d_dom_decomp, domain_part)
				},
			};
			let support = support
				.into_iter()
				.map(|idx| metadata.dom_first_permutation.inverse_map(idx))
				.collect();
			(birth, support)
		})
		.collect()
}

// LoPhat supplies an empty map for an empty decomposition. Nonempty
// decompositions must retain V to supply a complete representative map.
fn decomposition_with_reps<Decomp: Decomposition<C>, C: Column>(
	decomposition: &Decomp,
) -> (PersistenceDiagram, HashMap<usize, Vec<usize>>) {
	let (diagram, representatives) = decomposition.diagram_with_reps();
	let representatives = representatives.expect("Representative extraction requires retained V");
	(diagram, representatives)
}

// Skip all representative extraction and allocation when it was not requested.
fn decomposition_diagram<Decomp: Decomposition<C>, C: Column>(
	decomposition: &Decomp,
	get_representatives: bool,
) -> (PersistenceDiagram, Option<HashMap<usize, Vec<usize>>>) {
	if get_representatives {
		let (diagram, representatives) = decomposition_with_reps(decomposition);
		(diagram, Some(representatives))
	} else {
		(decomposition.diagram(), None)
	}
}

fn domain_diagram<Decomp: Decomposition<C>, C: Column>(
	metadata: &EnsembleMetadata,
	d_dom_decomp: &Decomp,
	get_representatives: bool,
) -> (PersistenceDiagram, Option<HashMap<usize, Vec<usize>>>) {
	let (diagram, representatives) = decomposition_diagram(d_dom_decomp, get_representatives);
	let unpermute = |idx| metadata.dom_first_permutation.inverse_map(idx);
	let representatives = representatives.map(|r| {
		r.into_iter()
			.map(|(birth, support)| {
				(
					unpermute(birth),
					support.into_iter().map(unpermute).collect(),
				)
			})
			.collect()
	});
	(diagram.map_idxs(unpermute), representatives)
}

fn image_diagram<Decomp: Decomposition<C>, C: Column>(
	metadata: &EnsembleMetadata,
	d_dom_decomp: &Decomp,
	d_im_decomp: &Decomp,
	domain_representatives: &Option<HashMap<usize, Vec<usize>>>,
) -> (PersistenceDiagram, Option<HashMap<usize, Vec<usize>>>) {
	let mut im_dgm = PersistenceDiagram::default();
	(0..metadata.sz_codomain).for_each(|idx| {
		if let Some(low_idx) = d_im_decomp.get_r_col(idx).pivot() {
			let birth_idx = metadata.dom_first_permutation.inverse_map(low_idx);
			let low_idx_in_dom = low_idx < metadata.sz_domain;
			if low_idx_in_dom {
				im_dgm.insert(birth_idx, Finite(idx));
			}
		} else {
			// Check if the column is a birth in the domain.
			let idx_dom_first = metadata.dom_first_permutation.map(idx);
			let idx_in_domain = idx_dom_first < metadata.sz_domain;
			if idx_in_domain && d_dom_decomp.get_r_col(idx_dom_first).is_cycle() {
				im_dgm.insert(idx, Infinity);
			}
		}
	});
	// Image R has domain-first rows and original columns. Only its finite
	// representatives are usable here: image V may include non-domain cells.
	let representatives = domain_representatives.as_ref().map(|dom_reprs| {
		let (_, reps) = decomposition_with_reps(d_im_decomp);
		im_dgm
			.iter()
			.map(|(&birth, &death)| {
				let support = match death {
					Finite(_) => {
						reps[&metadata.dom_first_permutation.map(birth)]
							.iter()
							.map(|&idx| metadata.dom_first_permutation.inverse_map(idx))
							.collect()
					},
					Infinity => dom_reprs[&birth].clone(),
				};
				(birth, support)
			})
			.collect()
	});
	(im_dgm, representatives)
}

fn cokernel_diagram<Decomp: Decomposition<C>, C: Column>(
	metadata: &EnsembleMetadata,
	d_dom_decomp: &Decomp,
	d_im_decomp: &Decomp,
	d_cok_decomp: &Decomp,
) -> PersistenceDiagram {
	let mut dgm = PersistenceDiagram::default();
	let sz_codomain = d_im_decomp.n_cols();
	(0..sz_codomain).for_each(|idx| {
		let is_birth_in_cod = d_im_decomp.get_r_col(idx).is_cycle();
		let idx_dom_first = metadata.dom_first_permutation.map(idx);
		let idx_in_dom = idx_dom_first < metadata.sz_domain;
		let not_in_dom_or_neg_in_dom =
			(!idx_in_dom) || d_dom_decomp.get_r_col(idx_dom_first).is_boundary();
		if is_birth_in_cod && not_in_dom_or_neg_in_dom {
			dgm.insert(idx, Infinity);
			return;
		}
		if is_birth_in_cod {
			return;
		}
		let low_idx_in_dom = d_im_decomp.get_r_col(idx).pivot().unwrap() < metadata.sz_domain;
		if !low_idx_in_dom {
			let lowest_in_r_cok = d_cok_decomp.get_r_col(idx).pivot().unwrap();
			dgm.insert(lowest_in_r_cok, Finite(idx));
		}
	});
	dgm
}
impl<C: Column, Algo: DecompositionAlgo<C>> DecompositionEnsemble<C, Algo> {
	/// Return all six diagrams in original codomain generator indices.
	///
	/// The ensemble must describe one valid inclusion. Results can be looked up
	/// by birth index; an essential class has death
	/// [`ExtendedUsize::Infinity`]. For a mapping-cylinder inclusion, use
	/// its metadata to recover entrance times. Calling this method again
	/// returns the same diagrams.
	/// Domain, codomain, image, and kernel representatives use original
	/// generator indices when `get_representatives` is true. Requested
	/// empty diagrams have `Some` of an empty map. When false, all four
	/// representative maps are `None`, and no representative extraction is
	/// performed.
	///
	/// # Panics
	/// Panics if representatives are requested and a nonempty domain, codomain,
	/// or image decomposition did not retain V. The computation entry points
	/// retain these matrices.
	///
	/// # Example
	///
	/// ```
	/// use lophat::algorithms::LockFreeAlgorithm;
	/// use phimaker::all_decompositions;
	///
	/// let ensemble = all_decompositions::<LockFreeAlgorithm<_>, _>(&[vec![]], &[0], &[0], 1)?;
	/// let diagrams = ensemble.all_diagrams(false);
	/// assert_eq!(diagrams.domain.len(), 1);
	/// assert!(diagrams.domain_representatives.is_none());
	/// assert_eq!(diagrams.domain, ensemble.all_diagrams(false).domain);
	/// # Ok::<(), phimaker::errors::InclusionError>(())
	/// ```
	pub fn all_diagrams(&self, get_representatives: bool) -> DiagramEnsemble {
		let (domain, domain_representatives) =
			domain_diagram(&self.metadata, &self.d_dom, get_representatives);
		let (codomain, codomain_representatives) =
			decomposition_diagram(&self.d_cod, get_representatives);
		let (image, image_representatives) = image_diagram(
			&self.metadata,
			&self.d_dom,
			&self.d_im,
			&domain_representatives,
		);
		let kernel = kernel_diagram(&self.metadata, &self.d_ker, &self.d_dom, &self.d_im);
		let kernel_representatives = get_representatives.then(|| {
			kernel_representatives(
				&self.metadata,
				&kernel,
				&self.d_ker,
				&self.d_dom,
				&self.d_im,
			)
		});
		DiagramEnsemble {
			domain,
			domain_representatives,
			codomain,
			codomain_representatives,
			image,
			image_representatives,
			kernel,
			kernel_representatives,
			relative: {
				let f = |idx| {
					self.metadata
						.dom_first_permutation
						.inverse_map(idx + self.metadata.sz_domain)
				};
				PersistenceDiagram::from_decomposition(&self.d_rel)
					.anti_transpose(self.metadata.sz_codomain - self.metadata.sz_domain)
					.map_idxs(f)
			},
			cokernel: cokernel_diagram(&self.metadata, &self.d_dom, &self.d_im, &self.d_cok),
		}
	}
}

/// Read a serialized decomposition at the current file position.
///
/// Use a file produced by [`crate::ensemble::to_file`] for a compatible
/// decomposition. Reading advances the shared file cursor; seek back before
/// reading again.
///
/// # Panics
/// Panics on I/O errors or incompatible or corrupt serialized data.
///
/// # Example
///
/// ```
/// use lophat::{
/// 	algorithms::{Decomposition, DecompositionAlgo, SerialAlgorithm},
/// 	columns::VecColumn,
/// };
/// use phimaker::{diagrams::from_file, ensemble::to_file};
///
/// let decomposition = SerialAlgorithm::<VecColumn>::init(None)
/// 	.add_cols([VecColumn::from((0, vec![]))].into_iter())
/// 	.decompose();
/// let file = to_file(decomposition);
/// assert_eq!(from_file(&file).n_cols(), 1);
/// ```
pub fn from_file(file: &File) -> DecompositionFileFormat {
	let buf = BufReader::new(file);
	deserialize_from(buf).expect("Can't deserialize from file")
	//from_reader(file).expect("JSON deserializes")
}

impl FileEnsemble {
	/// Read all six diagrams in original codomain generator indices.
	///
	/// Files must be positioned at the start of compatible decomposition data.
	/// Reading advances every file cursor; rewind all six before calling again.
	/// This call requires memory for all six decompositions.
	/// Domain, codomain, image, and kernel representatives use original
	/// generator indices when `get_representatives` is true. Requested
	/// empty diagrams have `Some` of an empty map. When false, all four
	/// representative maps are `None`, and no representative extraction is
	/// performed.
	///
	/// # Panics
	/// Panics on read errors or incompatible or corrupt serialized data, or if
	/// representatives are requested from a nonempty domain, codomain, or image
	/// decomposition that did not retain V.
	/// The computation entry points retain these matrices.
	///
	/// # Example
	///
	/// ```
	/// use lophat::algorithms::LockFreeAlgorithm;
	/// use phimaker::all_decompositions_slow;
	///
	/// let stored = all_decompositions_slow::<LockFreeAlgorithm<_>, _>(&[vec![]], &[0], &[0], 1)?;
	/// assert_eq!(stored.all_diagrams(false).domain.len(), 1);
	/// # Ok::<(), phimaker::errors::InclusionError>(())
	/// ```
	pub fn all_diagrams(&self, get_representatives: bool) -> DiagramEnsemble {
		let d_dom_decomp = from_file(&self.d_dom);
		let d_cod_decomp = from_file(&self.d_cod);
		let d_im_decomp = from_file(&self.d_im);
		let d_cok_decomp = from_file(&self.d_cok);
		let d_ker_decomp = from_file(&self.d_ker);
		let d_rel_decomp = from_file(&self.d_rel);
		let (domain, domain_representatives) =
			domain_diagram(&self.metadata, &d_dom_decomp, get_representatives);
		let (codomain, codomain_representatives) =
			decomposition_diagram(&d_cod_decomp, get_representatives);
		let (image, image_representatives) = image_diagram(
			&self.metadata,
			&d_dom_decomp,
			&d_im_decomp,
			&domain_representatives,
		);
		let kernel = kernel_diagram(&self.metadata, &d_ker_decomp, &d_dom_decomp, &d_im_decomp);
		let kernel_representatives = get_representatives.then(|| {
			kernel_representatives(
				&self.metadata,
				&kernel,
				&d_ker_decomp,
				&d_dom_decomp,
				&d_im_decomp,
			)
		});
		DiagramEnsemble {
			domain,
			domain_representatives,
			codomain,
			codomain_representatives,
			image,
			image_representatives,
			kernel,
			kernel_representatives,
			relative: {
				let f = |idx| {
					self.metadata
						.dom_first_permutation
						.inverse_map(idx + self.metadata.sz_domain)
				};
				PersistenceDiagram::from_decomposition(&d_rel_decomp)
					.anti_transpose(self.metadata.sz_codomain - self.metadata.sz_domain)
					.map_idxs(f)
			},
			cokernel: cokernel_diagram(&self.metadata, &d_dom_decomp, &d_im_decomp, &d_cok_decomp),
		}
	}
}

#[cfg(test)]
mod tests {
	use std::collections::HashMap;

	use lophat::{
		algorithms::{DecompositionAlgo, SerialAlgorithm},
		columns::VecColumn,
		utils::anti_transpose,
	};

	use super::{
		ExtendedUsize::{Finite, Infinity},
		PersistenceDiagram,
		solve,
		sum_mod2,
	};

	/// Solve every combination of a sparse triangular basis, including zero;
	/// missing pivots detect right-hand sides outside its column span.
	#[test]
	fn sparse_f2_solve() {
		let a = HashMap::from([(2, vec![0, 2]), (4, vec![1, 2, 4]), (7, vec![0, 4, 7])]);
		for mask in 0..8 {
			let mut b = Vec::new();
			let mut expected = Vec::new();
			for (bit, key) in [2, 4, 7].into_iter().enumerate() {
				if mask & (1 << bit) != 0 {
					b = sum_mod2(&b, &a[&key]);
					expected.push(key);
				}
			}
			assert_eq!(solve(&a, &b), Some(expected));
		}
		assert_eq!(solve(&a, &[0]), None);
		assert_eq!(solve(&HashMap::new(), &[]), Some(vec![]));
		assert_eq!(solve(&HashMap::new(), &[1]), None);
	}

	/// A filled triangle checks finite and essential bars and anti-transpose
	/// recovery.
	#[test]
	fn diagrams_from_boundary_and_anti_transpose_agree() {
		// A filled triangle: two H_0 deaths, one H_1 death, one essential H_0
		// class.
		let matrix: Vec<VecColumn> = vec![
			(0, vec![]),
			(0, vec![]),
			(0, vec![]),
			(1, vec![0, 1]),
			(1, vec![1, 2]),
			(1, vec![0, 2]),
			(2, vec![3, 4, 5]),
		]
		.into_iter()
		.map(VecColumn::from)
		.collect();
		let expected = PersistenceDiagram(
			[
				(0, Infinity),
				(1, Finite(3)),
				(2, Finite(4)),
				(5, Finite(6)),
			]
			.into(),
		);
		let decomposition = SerialAlgorithm::init(None)
			.add_cols(matrix.clone().into_iter())
			.decompose();
		assert_eq!(
			PersistenceDiagram::from_decomposition(&decomposition),
			expected
		);

		let transposed = SerialAlgorithm::init(None)
			.add_cols(anti_transpose(&matrix).into_iter())
			.decompose();
		assert_eq!(
			PersistenceDiagram::from_decomposition(&transposed).anti_transpose(matrix.len()),
			expected
		);
	}

	/// An empty reduction must not underflow during anti-transpose recovery.
	#[test]
	fn empty_decomposition_has_empty_diagram() {
		let decomposition = SerialAlgorithm::<VecColumn>::init(None).decompose();
		assert_eq!(
			PersistenceDiagram::from_decomposition(&decomposition).anti_transpose(0),
			PersistenceDiagram::default()
		);
	}

	/// Infinity is larger than even the largest representable finite index.
	#[test]
	fn infinity_orders_after_all_finite_indices() {
		assert!(Finite(0) < Finite(1));
		assert!(Finite(usize::MAX) < Infinity);
		assert_eq!(Infinity.cmp(&Infinity), std::cmp::Ordering::Equal);
	}

	/// Both finite endpoints are reindexed; infinity and deterministic display
	/// survive.
	#[test]
	fn reindexing_preserves_essential_and_finite_intervals() {
		use crate::{builders::compute_dom_first_permutation, indexing::Permutation};

		let sz_codomain = 4;
		let cols_in_dom = vec![1, 3];
		let mapping = compute_dom_first_permutation(sz_codomain, &cols_in_dom);
		let diagram = PersistenceDiagram([(0, Finite(3)), (1, Infinity)].into())
			.map_idxs(|idx| mapping.inverse_map(idx));
		assert_eq!(
			diagram,
			PersistenceDiagram([(1, Finite(2)), (3, Infinity)].into())
		);
		assert_eq!(diagram.to_string(), "{1: 2, 3: Inf}");
	}
}
