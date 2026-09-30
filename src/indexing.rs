//! Index permutations between original and domain-first coordinates.

use itertools::Itertools;

/// A bijection on a finite index set, with its inverse.
pub trait Permutation {
    /// Map an original index to its permuted position. The index must be in range.
    fn map(&self, idx: usize) -> usize;
    /// Recover the original index from an in-range permuted position.
    fn inverse_map(&self, idx: usize) -> usize;
}

/// A permutation stored as forward and inverse lookup tables.
/// Out-of-range lookups panic; the default permutation is empty.
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
    /// Map an original index to its permuted position. The index must be in range.
    fn map(&self, idx: usize) -> usize {
        self.perm[idx]
    }
    /// Recover the original index from an in-range permuted position.
    fn inverse_map(&self, idx: usize) -> usize {
        self.inverse[idx]
    }
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
