//! Birth-to-death dictionaries and extraction from reduced matrices.

use std::{fs::File, io::BufReader};

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
	indexing::Permutation,
};

/// Six persistence diagrams for an inclusion or filtered chain map over F2.
///
/// Each field maps birth indices to death indices or
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
/// # Example
///
/// ```
/// use phimaker::{diagrams::ExtendedUsize::Infinity, sixpack_from_inclusion};
///
/// let diagrams = sixpack_from_inclusion(&[vec![]], &[0], &[0], 1, false)?;
/// assert_eq!(diagrams.domain[&0], Infinity);
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

Attributes are read-only, and each access returns an independent dictionary.
Editing a dictionary does not change this result or later attribute accesses.

Example::

    >>> from phimaker import sixpack_from_inclusion
    >>> result = sixpack_from_inclusion([[]], [0], [0])
    >>> assert result.domain == {0: None}
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

fn image_diagram<Decomp: Decomposition<C>, C: Column>(
	metadata: &EnsembleMetadata,
	d_dom_decomp: &Decomp,
	d_im_decomp: &Decomp,
) -> PersistenceDiagram {
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
	im_dgm
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
		DiagramEnsemble {
			domain: {
				PersistenceDiagram::from_decomposition(&self.d_dom)
					.map_idxs(|idx| self.metadata.dom_first_permutation.inverse_map(idx))
			},
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
			image: image_diagram(&self.metadata, &self.d_dom, &self.d_im),
			kernel: kernel_diagram(&self.metadata, &self.d_ker, &self.d_dom, &self.d_im),
			cokernel: cokernel_diagram(&self.metadata, &self.d_dom, &self.d_im, &self.d_cok),
			codomain: PersistenceDiagram::from_decomposition(&self.d_cod),
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
	///
	/// # Panics
	/// Panics on read errors or incompatible or corrupt serialized data.
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
		DiagramEnsemble {
			domain: PersistenceDiagram::from_decomposition(&d_dom_decomp)
				.map_idxs(|idx| self.metadata.dom_first_permutation.inverse_map(idx)),
			codomain: PersistenceDiagram::from_decomposition(&d_cod_decomp),
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
			image: image_diagram(&self.metadata, &d_dom_decomp, &d_im_decomp),
			kernel: kernel_diagram(&self.metadata, &d_ker_decomp, &d_dom_decomp, &d_im_decomp),
			cokernel: cokernel_diagram(&self.metadata, &d_dom_decomp, &d_im_decomp, &d_cok_decomp),
		}
	}
}

#[cfg(test)]
mod tests {
	use lophat::{
		algorithms::{DecompositionAlgo, SerialAlgorithm},
		columns::VecColumn,
		utils::anti_transpose,
	};

	use super::{
		ExtendedUsize::{Finite, Infinity},
		PersistenceDiagram,
	};

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
