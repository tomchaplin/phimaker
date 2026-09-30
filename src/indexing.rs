use itertools::Itertools;

pub trait Permutation {
    fn map(&self, idx: usize) -> usize;
    fn inverse_map(&self, idx: usize) -> usize;
}

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
    fn map(&self, idx: usize) -> usize {
        self.perm[idx]
    }
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
