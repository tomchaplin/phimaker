//! Birth-to-death dictionaries and extraction from reduced matrices.

use bincode::deserialize_from;
use std::{
    collections::HashMap,
    fmt::{self, Display},
    fs::File,
    io::BufReader,
    ops::{Deref, DerefMut},
};

use lophat::{
    algorithms::{Decomposition, DecompositionAlgo},
    columns::Column,
    utils::DecompositionFileFormat,
};
use pyo3::prelude::*;

use crate::{
    ensemble::{DecompositionEnsemble, EnsembleMetadata, FileEnsemble},
    indexing::Permutation,
};

/// A nonnegative index or infinity.
/// Infinity is greater than every finite index.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ExtendedUsize {
    /// A finite death column index.
    Finite(usize),
    /// An essential class that never dies.
    Infinity,
}

use ExtendedUsize::{Finite, Infinity};

impl Display for ExtendedUsize {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Finite(index) => write!(f, "{index}"),
            Infinity => write!(f, "Inf"),
        }
    }
}

impl<'py> IntoPyObject<'py> for ExtendedUsize {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> PyResult<Self::Output> {
        let index = match self {
            Finite(index) => Some(index),
            Infinity => None,
        };
        Ok(index.into_pyobject(py)?)
    }
}

impl<'py> IntoPyObject<'py> for &ExtendedUsize {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> PyResult<Self::Output> {
        (*self).into_pyobject(py)
    }
}

/// Maps each generator's birth index to its death index, or infinity.
/// Python receives a dictionary with `None` for infinite deaths.
#[derive(Default, Debug, Clone, PartialEq, Eq, IntoPyObject, IntoPyObjectRef)]
pub struct PersistenceDiagram(
    /// Birth-to-death entries; keys identify generators uniquely.
    pub HashMap<usize, ExtendedUsize>,
);

impl Deref for PersistenceDiagram {
    type Target = HashMap<usize, ExtendedUsize>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for PersistenceDiagram {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl Display for PersistenceDiagram {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Stable output without imposing an ordering on the underlying map.
        let mut entries: Vec<_> = self.iter().collect();
        entries.sort_unstable_by_key(|(birth, _)| **birth);
        write!(f, "{{")?;
        for (position, (birth, death)) in entries.into_iter().enumerate() {
            if position > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{birth}: {death}")?;
        }
        write!(f, "}}")
    }
}

impl FromIterator<(usize, ExtendedUsize)> for PersistenceDiagram {
    fn from_iter<T: IntoIterator<Item = (usize, ExtendedUsize)>>(iter: T) -> Self {
        PersistenceDiagram(iter.into_iter().collect())
    }
}
impl PersistenceDiagram {
    /// Extract pairings from a reduced square, strictly upper-triangular boundary matrix.
    /// Requires a valid chain-complex reduction. V is not needed. Results use the
    /// supplied matrix coordinates; reverse anti-transposed coordinates separately.
    pub fn from_decomposition<C: Column>(decomposition: &impl Decomposition<C>) -> Self {
        let mut diagram = Self((0..decomposition.n_cols()).map(|i| (i, Infinity)).collect());
        for death in 0..decomposition.n_cols() {
            if let Some(birth) = decomposition.get_r_col(death).pivot() {
                diagram.insert(birth, Finite(death));
                diagram.remove(&death);
            }
        }
        diagram
    }

    /// Restore filtration indices after reducing an anti-transposed boundary matrix.
    /// Finite pairs (b, d) become (n - 1 - d, n - 1 - b); essential birth b
    /// becomes n - 1 - b. Every finite index must be less than `matrix_size`.
    /// An empty diagram accepts size zero.
    pub fn anti_transpose(self, matrix_size: usize) -> Self {
        Self(
            self.0
                .into_iter()
                .map(|(birth, death)| match death {
                    Finite(death) => (matrix_size - 1 - death, Finite(matrix_size - 1 - birth)),
                    Infinity => (matrix_size - 1 - birth, Infinity),
                })
                .collect(),
        )
    }

    /// Apply the inverse permutation to every birth and finite death.
    /// All indices must lie in the permutation range; infinity is unchanged.
    pub fn unpermute_idxs(mut self, mapping: &impl Permutation) -> Self {
        self.0 = self
            .drain()
            .map(|(birth, death)| {
                let birth = mapping.inverse_map(birth);
                let death = match death {
                    ExtendedUsize::Finite(death) => {
                        ExtendedUsize::Finite(mapping.inverse_map(death))
                    }
                    ExtendedUsize::Infinity => ExtendedUsize::Infinity,
                };
                (birth, death)
            })
            .collect();
        self
    }

    /// Map every birth and finite death through f, leaving infinity unchanged.
    /// To preserve distinct generators, f must be injective on births. Colliding
    /// keys overwrite entries in unspecified order. No order validation is performed.
    pub fn map_idxs<F: Fn(usize) -> usize>(mut self, f: F) -> Self {
        self.0 = self
            .drain()
            .map(|(birth, death)| {
                let birth = f(birth);
                let death = match death {
                    Finite(death) => ExtendedUsize::Finite(f(death)),
                    Infinity => Infinity,
                };
                (birth, death)
            })
            .collect();
        self
    }
}

#[pyclass(get_all, from_py_object)]
#[derive(Debug, Clone)]
/// Six diagrams in original inclusion (or cylinder) indices, across all degrees.
/// Python getters return independent dicts with None for infinite death.
/// Kernel degree is birth-cell degree minus one; other degrees equal birth-cell degree.
pub struct DiagramEnsemble {
    /// Persistent homology of the codomain.
    pub codomain: PersistenceDiagram,
    /// Persistent homology of the domain.
    pub domain: PersistenceDiagram,
    /// Image of the induced homology map.
    pub image: PersistenceDiagram,
    /// Kernel of the induced homology map.
    pub kernel: PersistenceDiagram,
    /// Cokernel of the induced homology map.
    pub cokernel: PersistenceDiagram,
    /// Persistent homology of the quotient complex (no extra basepoint).
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
    // Note: we use d_im here because d_cod is anti-transposed
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
    // Note: we use d_im here because d_cod is anti-transposed
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
    /// Extract all six diagrams in original codomain column coordinates.
    /// The decompositions and metadata must come from the same valid inclusion.
    /// Does not mutate the decompositions; repeated extraction is supported.
    pub fn all_diagrams(&self) -> DiagramEnsemble {
        DiagramEnsemble {
            domain: {
                PersistenceDiagram::from_decomposition(&self.d_dom)
                    .unpermute_idxs(&self.metadata.dom_first_permutation)
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
            codomain: PersistenceDiagram::from_decomposition(&self.d_cod)
                .anti_transpose(self.metadata.sz_codomain),
        }
    }
}

/// Deserialize a LoPhat decomposition from the current file position using bincode.
/// Reading advances the shared file cursor; seek back before reading again.
///
/// # Panics
/// Panics on I/O errors or incompatible/corrupt serialized data.
pub fn from_file(file: &File) -> DecompositionFileFormat {
    let buf = BufReader::new(file);
    deserialize_from(buf).expect("Can't deserialize from file")
    //from_reader(file).expect("JSON deserializes")
}

impl FileEnsemble {
    /// Load all six decompositions and extract diagrams in original coordinates.
    /// All files must be positioned at the start of compatible serialized data.
    /// This loads all six simultaneously and advances their cursors; rewind each
    /// file before calling again. Panics on read/deserialization errors.
    pub fn all_diagrams(&self) -> DiagramEnsemble {
        let d_dom_decomp = from_file(&self.d_dom);
        let d_cod_decomp = from_file(&self.d_cod);
        let d_im_decomp = from_file(&self.d_im);
        let d_cok_decomp = from_file(&self.d_cok);
        let d_ker_decomp = from_file(&self.d_ker);
        let d_rel_decomp = from_file(&self.d_rel);
        DiagramEnsemble {
            domain: PersistenceDiagram::from_decomposition(&d_dom_decomp)
                .unpermute_idxs(&self.metadata.dom_first_permutation),
            codomain: PersistenceDiagram::from_decomposition(&d_cod_decomp)
                .anti_transpose(self.metadata.sz_codomain),
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
    use super::{
        ExtendedUsize::{Finite, Infinity},
        PersistenceDiagram,
    };
    use lophat::{
        algorithms::{DecompositionAlgo, SerialAlgorithm},
        columns::VecColumn,
        utils::anti_transpose,
    };

    #[test]
    fn diagrams_from_boundary_and_anti_transpose_agree() {
        // A filled triangle: two H_0 deaths, one H_1 death, one essential H_0 class.
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

    #[test]
    fn empty_decomposition_has_empty_diagram() {
        let decomposition = SerialAlgorithm::<VecColumn>::init(None).decompose();
        assert_eq!(
            PersistenceDiagram::from_decomposition(&decomposition).anti_transpose(0),
            PersistenceDiagram::default()
        );
    }

    #[test]
    fn infinity_orders_after_all_finite_indices() {
        assert!(Finite(0) < Finite(1));
        assert!(Finite(usize::MAX) < Infinity);
        assert_eq!(Infinity.cmp(&Infinity), std::cmp::Ordering::Equal);
    }

    #[test]
    fn reindexing_preserves_essential_and_finite_intervals() {
        use crate::builders::compute_dom_first_permutation;

        let sz_codomain = 4;
        let cols_in_dom = vec![1, 3];
        let mapping = compute_dom_first_permutation(sz_codomain, &cols_in_dom);
        let diagram =
            PersistenceDiagram([(0, Finite(3)), (1, Infinity)].into()).unpermute_idxs(&mapping);
        assert_eq!(
            diagram,
            PersistenceDiagram([(1, Finite(2)), (3, Infinity)].into())
        );
        assert_eq!(diagram.to_string(), "{1: 2, 3: Inf}");
    }
}
