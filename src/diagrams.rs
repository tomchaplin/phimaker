//! Birth-to-death dictionaries and extraction from reduced matrices.

use std::{collections::HashMap, fs::File, io::BufReader};

use ExtendedUsize::{Finite, Infinity};
use bincode::deserialize_from;
/// Shared birth-to-death diagrams and extended indices from LoPhat.
pub use lophat::utils::{ExtendedUsize, PersistenceDiagram};
use lophat::{
	algorithms::{Decomposition, DecompositionAlgo, NoVMatrixError},
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
/// The domain, codomain, and image representative maps associate each birth
/// with the generator indices whose sum is its cycle over F2. Their keys and
/// entries use the same coordinates as the diagrams. If `b` is a feature's
/// birth index in any of these diagrams, the maximum generator index in its
/// representative equals `b`. Representatives are not necessarily minimal or
/// unique, and support order is unspecified. Each representative map has
/// exactly its diagram's birth keys; an empty diagram has an empty map.
///
/// # Example
///
/// ```
/// use phimaker::{diagrams::ExtendedUsize::Infinity, sixpack_from_inclusion};
///
/// let diagrams = sixpack_from_inclusion(&[vec![]], &[0], &[0], 1, false)?;
/// assert_eq!(diagrams.domain[&0], Infinity);
/// assert_eq!(diagrams.domain_representatives[&0], vec![0]);
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

The ``domain_representatives``, ``codomain_representatives``, and
``image_representatives`` attributes map birth indices to lists of generator
indices. Summing the listed generators over the field with two elements gives
a representative cycle. Keys and entries use the same coordinates as the diagrams.
If ``b`` is a feature's birth index in any of these diagrams, the maximum index
in its representative is equal to ``b``. Representatives need not be minimal or
unique. Each representative dictionary has exactly its diagram's birth keys.
An empty diagram has an empty representative dictionary.

Attributes are read-only. Each dictionary access returns an independent copy,
including the lists in representative dictionaries. Editing a copy does not
change this result or later attribute accesses. Support order is unspecified.

Example::

    >>> from phimaker import sixpack_from_inclusion
    >>> result = sixpack_from_inclusion([[]], [0], [0])
    >>> assert result.domain == {0: None}
    >>> assert result.domain_representatives == {0: [0]}
    >>> diagram = result.domain
    >>> diagram.clear()
    >>> assert result.domain == {0: None}"#
)]
#[cfg_attr(
    all(feature = "python-module", not(any(doc, rust_analyzer))),
    macro_rules_apply(strip_rust_docs!)
)]
#[cfg_attr(feature = "python-module", pyclass(get_all, from_py_object))]
#[derive(Debug, Clone)]
pub struct DiagramEnsemble {
	/// Persistent homology of the codomain.
	#[doc = python_doc!(
    r#"Persistent homology of the codomain, as a birth-to-death dictionary."#
    )]
	pub codomain: PersistenceDiagram,
	/// Persistent homology of the domain.
	#[doc = python_doc!(
    r#"Persistent homology of the domain, as a birth-to-death dictionary."#
    )]
	pub domain: PersistenceDiagram,
	/// Image of the induced homology map.
	#[doc = python_doc!(
    r#"Image of the induced homology map, as a birth-to-death dictionary."#
    )]
	pub image: PersistenceDiagram,
	/// Domain birth indices mapped to cycle supports in diagram coordinates.
	/// Each listed generator has coefficient one over F2. An empty domain
	/// diagram has an empty map.
	#[doc = python_doc!(
    r#"Domain representatives, as a dictionary from birth indices to generator lists.

Sum the listed generators over the field with two elements to obtain a cycle.
Indices use the domain diagram's coordinates. An empty domain diagram has an
empty dictionary."#
    )]
	pub domain_representatives: HashMap<usize, Vec<usize>>,
	/// Codomain birth indices mapped to cycle supports in diagram coordinates.
	/// Each listed generator has coefficient one over F2. An empty codomain
	/// diagram has an empty map.
	#[doc = python_doc!(
    r#"Codomain representatives, as a dictionary from birth indices to generator lists.

Sum the listed generators over the field with two elements to obtain a cycle.
Indices use the codomain diagram's coordinates. An empty codomain diagram has an
empty dictionary."#
    )]
	pub codomain_representatives: HashMap<usize, Vec<usize>>,
	/// Image birth indices mapped to cycle supports in diagram coordinates.
	/// Supports contain only domain generators, each with coefficient one over
	/// F2. An empty image diagram has an empty map.
	#[doc = python_doc!(
    r#"Image representatives, as a dictionary from birth indices to generator lists.

Sum the listed generators over the field with two elements to obtain a cycle.
Only domain generators occur, in the image diagram's coordinates. An empty image
diagram has an empty dictionary."#
    )]
	pub image_representatives: HashMap<usize, Vec<usize>>,
	/// Kernel of the induced homology map. Birth indices identify killing
	/// cells, so the interval degree is the birth-cell degree minus one.
	#[doc = python_doc!(
    r#"Kernel of the induced homology map, as a birth-to-death dictionary.

The degree of an interval is the degree of its birth generator minus one."#
    )]
	pub kernel: PersistenceDiagram,
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
fn xor_sorted(left: &[usize], right: &[usize]) -> Vec<usize> {
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
// of the column with that pivot'., including b, must be sorted
// and duplicate-free. Back substitution visits strictly decreasing pivots.
// Return None if b is outside the span; otherwise return the column keys of x.
fn solve(a: &HashMap<usize, Vec<usize>>, b: &[usize]) -> Option<Vec<usize>> {
	let mut residual = b.to_vec();
	let mut solution = Vec::new();
	while let Some(&pivot) = residual.last() {
		let column = a.get(&pivot)?;
		residual = xor_sorted(&residual, column);
		solution.push(pivot);
	}
	solution.reverse();
	Some(solution)
}

// Make sure that representative support is sorted and duplicate-free.
// Normalize once for all right-hand sides and verify the pivot invariant that
// makes the representative matrix triangular. This does not check cyclehood.
fn prepare_representatives(
	diagram: &PersistenceDiagram,
	representatives: &HashMap<usize, Vec<usize>>,
	name: &'static str,
) -> Result<HashMap<usize, Vec<usize>>, PersistenceMapError> {
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
			if *death <= Finite(birth) || support.last() != Some(&birth) {
				return Err(inconsistent());
			}
			Ok((birth, support))
		})
		.collect()
}

/// Source birth indices mapped to target births with coefficient one over F2.
pub type PersistenceMap = HashMap<usize, Vec<usize>>;
impl DiagramEnsemble {
	/// Return the domain-to-codomain and domain-to-image persistence maps over
	/// F2.
	///
	/// The pair is ordered `(domain_to_codomain, domain_to_image)`. Each map
	/// has every domain birth as a key. Its value is a sorted list of
	/// target births with coefficient one; an empty list denotes zero. Keys
	/// use the diagram coordinates, including mapping-cylinder indices for
	/// [`crate::sixpack`]. At a filtration index, use only source and
	/// target intervals alive there. These coefficients are relative to
	/// this ensemble's chosen representatives.
	///
	/// Use an ensemble from the computation functions with representatives
	/// available. Domain and target cycles must share one generator basis and
	/// describe a valid inclusion. Each support must have maximum index equal
	/// to its birth. Support order does not matter, and repeated indices are
	/// treated as repeated listings of the same generator. Target
	/// cycles must span the domain cycles. Do not modify diagrams and
	/// representatives independently. Cyclehood and homological degrees are
	/// not checked here.
	///
	/// Target features dead at the source birth contribute zero and are
	/// omitted. Every remaining coefficient satisfies `target_birth <=
	/// source_birth < target_death <= source_death`, with infinity allowed
	/// for deaths. An empty domain returns two empty maps without requiring
	/// representatives.
	///
	/// # Errors
	/// Returns [`PersistenceMapError`] for unavailable representatives,
	/// inconsistent keys, supports or endpoints, an unsolvable cycle expansion,
	/// or a coefficient incompatible with the interval lifetimes.
	///
	/// # Example
	/// ```
	/// use phimaker::sixpack_from_inclusion;
	/// let diagrams =
	/// 	sixpack_from_inclusion(&[vec![], vec![], vec![0, 1]], &[0, 0, 1], &[1], 1, false)?;
	/// let (to_codomain, to_image) = diagrams.get_persistence_map()?;
	/// assert_eq!(to_codomain[&1], vec![0, 1]);
	/// assert_eq!(to_image[&1], vec![1]);
	/// # Ok::<(), Box<dyn std::error::Error>>(())
	/// ```
	pub fn get_persistence_map(
		&self,
	) -> Result<(PersistenceMap, PersistenceMap), PersistenceMapError> {
		if self.domain.is_empty() {
			return Ok((HashMap::new(), HashMap::new()));
		}
		let source = prepare_representatives(&self.domain, &self.domain_representatives, "domain")?;
		let compute = |target: &PersistenceDiagram, reps: &HashMap<usize, Vec<usize>>, name| {
			let basis = prepare_representatives(target, reps, name)?;
			source
				.iter()
				.map(|(&birth, support)| {
					let mut coefficients =
						solve(&basis, support).ok_or(PersistenceMapError::Unsolvable {
							target: name,
							birth,
						})?;
					coefficients.retain(|target_birth| target[target_birth] > Finite(birth));
					for &target_birth in &coefficients {
						if target[&target_birth] > self.domain[&birth] {
							return Err(PersistenceMapError::IncompatibleIntervals {
								target: name,
								birth,
								target_birth,
							});
						}
					}
					Ok((birth, coefficients))
				})
				.collect::<Result<HashMap<usize, Vec<usize>>, PersistenceMapError>>()
		};
		Ok((
			compute(&self.codomain, &self.codomain_representatives, "codomain")?,
			compute(&self.image, &self.image_representatives, "image")?,
		))
	}
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

// Empty decompositions cannot report V availability, but need no
// representatives. Nonempty decompositions must retain V to supply a complete
// representative map.
fn decomposition_with_reps<Decomp: Decomposition<C>, C: Column>(
	decomposition: &Decomp,
) -> (PersistenceDiagram, HashMap<usize, Vec<usize>>) {
	let (diagram, representatives) = decomposition.diagram_with_reps();
	let representatives = match representatives {
		Ok(reps) => reps,
		Err(NoVMatrixError::EmptyDecompositionError) => HashMap::new(),
		Err(NoVMatrixError::VMatrixDiscardedError) => {
			panic!("Representative extraction requires retained V")
		},
	};
	(diagram, representatives)
}

fn domain_diagram_with_reps<Decomp: Decomposition<C>, C: Column>(
	metadata: &EnsembleMetadata,
	d_dom_decomp: &Decomp,
) -> (PersistenceDiagram, HashMap<usize, Vec<usize>>) {
	let (diagram, representatives) = decomposition_with_reps(d_dom_decomp);
	let unpermute = |idx| metadata.dom_first_permutation.inverse_map(idx);
	let representatives = representatives
		.into_iter()
		.map(|(birth, support)| {
			(
				unpermute(birth),
				support.into_iter().map(unpermute).collect(),
			)
		})
		.collect();
	(diagram.map_idxs(unpermute), representatives)
}

fn image_diagram_with_reps<Decomp: Decomposition<C>, C: Column>(
	metadata: &EnsembleMetadata,
	d_dom_decomp: &Decomp,
	d_im_decomp: &Decomp,
	domain_representatives: &HashMap<usize, Vec<usize>>,
) -> (PersistenceDiagram, HashMap<usize, Vec<usize>>) {
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
	let (_, reps) = decomposition_with_reps(d_im_decomp);
	let representatives = im_dgm
		.iter()
		.map(|(&birth, &death)| {
			let support = match death {
				Finite(_) => {
					reps[&metadata.dom_first_permutation.map(birth)]
						.iter()
						.map(|&idx| metadata.dom_first_permutation.inverse_map(idx))
						.collect()
				},
				Infinity => domain_representatives[&birth].clone(),
			};
			(birth, support)
		})
		.collect();
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
	/// Domain, codomain, and image representatives use original generator
	/// indices. Empty diagrams have empty representative maps.
	///
	/// # Panics
	/// Panics if a nonempty domain, codomain, or image decomposition did not
	/// retain V. The computation entry points retain these matrices.
	///
	/// # Example
	///
	/// ```
	/// use lophat::algorithms::LockFreeAlgorithm;
	/// use phimaker::all_decompositions;
	///
	/// let ensemble = all_decompositions::<LockFreeAlgorithm<_>, _>(&[vec![]], &[0], &[0], 1)?;
	/// let diagrams = ensemble.all_diagrams();
	/// assert_eq!(diagrams.domain.len(), 1);
	/// assert_eq!(diagrams.domain, ensemble.all_diagrams().domain);
	/// # Ok::<(), phimaker::errors::InclusionError>(())
	/// ```
	pub fn all_diagrams(&self) -> DiagramEnsemble {
		let (domain, domain_representatives) =
			domain_diagram_with_reps(&self.metadata, &self.d_dom);
		let (codomain, codomain_representatives) = decomposition_with_reps(&self.d_cod);
		let (image, image_representatives) = image_diagram_with_reps(
			&self.metadata,
			&self.d_dom,
			&self.d_im,
			&domain_representatives,
		);
		DiagramEnsemble {
			domain,
			domain_representatives,
			codomain,
			codomain_representatives,
			image,
			image_representatives,
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
			kernel: kernel_diagram(&self.metadata, &self.d_ker, &self.d_dom, &self.d_im),
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
	/// Domain, codomain, and image representatives use original generator
	/// indices. Empty diagrams have empty representative maps.
	///
	/// # Panics
	/// Panics on read errors or incompatible or corrupt serialized data, or if
	/// a nonempty domain, codomain, or image decomposition did not retain V.
	/// The computation entry points retain these matrices.
	///
	/// # Example
	///
	/// ```
	/// use lophat::algorithms::LockFreeAlgorithm;
	/// use phimaker::all_decompositions_slow;
	///
	/// let stored = all_decompositions_slow::<LockFreeAlgorithm<_>, _>(&[vec![]], &[0], &[0], 1)?;
	/// assert_eq!(stored.all_diagrams().domain.len(), 1);
	/// # Ok::<(), phimaker::errors::InclusionError>(())
	/// ```
	pub fn all_diagrams(&self) -> DiagramEnsemble {
		let d_dom_decomp = from_file(&self.d_dom);
		let d_cod_decomp = from_file(&self.d_cod);
		let d_im_decomp = from_file(&self.d_im);
		let d_cok_decomp = from_file(&self.d_cok);
		let d_ker_decomp = from_file(&self.d_ker);
		let d_rel_decomp = from_file(&self.d_rel);
		let (domain, domain_representatives) =
			domain_diagram_with_reps(&self.metadata, &d_dom_decomp);
		let (codomain, codomain_representatives) = decomposition_with_reps(&d_cod_decomp);
		let (image, image_representatives) = image_diagram_with_reps(
			&self.metadata,
			&d_dom_decomp,
			&d_im_decomp,
			&domain_representatives,
		);
		DiagramEnsemble {
			domain,
			domain_representatives,
			codomain,
			codomain_representatives,
			image,
			image_representatives,
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
			kernel: kernel_diagram(&self.metadata, &d_ker_decomp, &d_dom_decomp, &d_im_decomp),
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
		xor_sorted,
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
					b = xor_sorted(&b, &a[&key]);
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
