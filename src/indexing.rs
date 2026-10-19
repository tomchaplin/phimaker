//! Indexing utilities, including permutations.
use itertools::Itertools;

/// A bijection on a finite index set, with its inverse.
pub trait Permutation {
	/// Map an original index to its permuted position. The index must be in
	/// range.
	fn map(&self, idx: usize) -> usize;
	/// Recover the original index from an in-range permuted position.
	fn inverse_map(&self, idx: usize) -> usize;
}

/// Translate between original and permuted generator indices.
///
/// Obtain the domain-first permutation from an inclusion's
/// [`EnsembleMetadata`](crate::ensemble::EnsembleMetadata).
/// [`Permutation::map`] translates original indices;
/// [`Permutation::inverse_map`] restores them. Out-of-range lookups panic. The
/// default permutation is empty.
///
/// # Example
///
/// ```
/// use lophat::algorithms::LockFreeAlgorithm;
/// use phimaker::{all_decompositions, indexing::Permutation};
///
/// let ensemble =
/// 	all_decompositions::<LockFreeAlgorithm<_>, _>(&[vec![], vec![]], &[0, 0], &[1], 1)?;
/// let permutation = &ensemble.metadata.dom_first_permutation;
/// assert_eq!(permutation.map(1), 0);
/// assert_eq!(permutation.inverse_map(0), 1);
/// # Ok::<(), phimaker::errors::InclusionError>(())
/// ```
#[derive(Debug, Clone, Default)]
pub struct DensePermutation {
	perm: Vec<usize>,
	inverse: Vec<usize>,
}

impl DensePermutation {
	pub(crate) fn new(perm: Vec<usize>) -> Self {
		let mut inverse = vec![0; perm.len()];
		perm.iter().enumerate().for_each(|(i, &j)| {
			inverse[j] = i;
		});
		DensePermutation { perm, inverse }
	}
}
impl Permutation for DensePermutation {
	/// Map an original index to its permuted position. The index must be in
	/// range.
	fn map(&self, idx: usize) -> usize { self.perm[idx] }

	/// Recover the original index from an in-range permuted position.
	fn inverse_map(&self, idx: usize) -> usize { self.inverse[idx] }
}

pub(crate) trait PermuteItems: Iterator<Item = usize> {
	fn permuted(self, perm: &impl Permutation) -> impl Iterator<Item = usize>;
	fn unpermuted(self, perm: &impl Permutation) -> impl Iterator<Item = usize>;
}

impl<I> PermuteItems for I
where
	I: Iterator<Item = usize>,
{
	fn permuted(self, perm: &impl Permutation) -> impl Iterator<Item = usize> {
		self.map(|row_idx| perm.map(row_idx)).sorted()
	}

	fn unpermuted(self, perm: &impl Permutation) -> impl Iterator<Item = usize> {
		self.map(|row_idx| perm.inverse_map(row_idx)).sorted()
	}
}
