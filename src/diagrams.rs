use bincode::deserialize_from;
use std::{
    collections::HashMap,
    fmt::{self, Display},
    fs::File,
    io::BufReader,
    ops::{Deref, DerefMut},
};

use log::debug;

use lophat::{
    algorithms::{Decomposition, DecompositionAlgo},
    columns::Column,
    utils::DecompositionFileFormat,
};
use pyo3::prelude::*;

use crate::{
    ensemble::{DecompositionEnsemble, EnsembleMetadata, FileEnsemble},
    indexing::{IndexMapping, unreorder_idxs},
};

/// A nonnegative index or infinity. Infinity is greater than every finite index.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ExtendedUsize {
    Finite(usize),
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
pub struct PersistenceDiagram(pub HashMap<usize, ExtendedUsize>);

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

impl PersistenceDiagram {
    /// Extract pairings from a reduced, square boundary matrix.
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
}

#[pyclass(get_all, from_py_object)]
#[derive(Debug, Clone)]
pub struct DiagramEnsemble {
    pub codomain: PersistenceDiagram,
    pub domain: PersistenceDiagram,
    pub image: PersistenceDiagram,
    pub kernel: PersistenceDiagram,
    pub cokernel: PersistenceDiagram,
    pub relative: PersistenceDiagram,
}

/// Returns the list of negative indices in the diagram, i.e., indices
/// of non-zero columns. Such columns represent deaths in the diagram.
fn compute_negative_list(metadata: &EnsembleMetadata, diagram: &PersistenceDiagram) -> Vec<bool> {
    let mut negative_list: Vec<bool> = vec![false; metadata.sz_cod];
    for death in diagram.values() {
        if let Finite(death) = death {
            negative_list[*death] = true;
        }
    }
    negative_list
}

fn is_kernel_birth<Decomp: Decomposition<C>, C: Column>(
    idx: usize,
    metadata: &EnsembleMetadata,
    cod_negative_list: &[bool],
    im_decomp: &Decomp,
) -> bool {
    let in_dom = metadata.col_in_dom[idx];
    if in_dom {
        return false;
    }
    let negative_in_cod = cod_negative_list[idx];
    if !negative_in_cod {
        return false;
    }
    let low_idx_in_dom = im_decomp.get_r_col(idx).pivot().unwrap() < metadata.sz_dom;
    if !low_idx_in_dom {
        return false;
    }
    true
}

fn is_kernel_death<Decomp: Decomposition<C>, C: Column>(
    idx: usize,
    metadata: &EnsembleMetadata,
    dom_decomp: &Decomp,
    cod_negative_list: &[bool],
) -> bool {
    let in_dom = metadata.col_in_dom[idx];
    if !in_dom {
        return false;
    }
    let dom_idx = metadata.dom_first_mapping.map(idx).unwrap();
    let negative_in_dom = dom_decomp.get_r_col(dom_idx).pivot().is_some();
    if !negative_in_dom {
        return false;
    }
    let negative_in_cod = cod_negative_list[idx];
    if negative_in_cod {
        return false;
    }
    true
}

fn kernel_diagram<Decomp: Decomposition<C>, C: Column>(
    metadata: &EnsembleMetadata,
    ker: &Decomp,
    dom_decomp: &Decomp,
    im_decomp: &Decomp,
    cod_negative_list: &[bool],
) -> PersistenceDiagram {
    let mut dgm = PersistenceDiagram::default();
    for idx in 0..metadata.sz_cod {
        if is_kernel_birth(idx, metadata, cod_negative_list, im_decomp) {
            dgm.insert(idx, Infinity);
            continue;
        }
        if is_kernel_death(idx, metadata, dom_decomp, cod_negative_list) {
            // TODO: Problem kernel columns have different indexing to f
            let ker_idx = metadata.kernel_mapping.map(idx).unwrap();
            let dom_birth_index = ker.get_r_col(ker_idx).pivot().unwrap();
            let birth_index = metadata
                .dom_first_mapping
                .inverse_map(dom_birth_index)
                .unwrap();
            dgm.insert(birth_index, Finite(idx));
        }
    }
    dgm
}

fn codomain_image_diagram<Decomp: Decomposition<C>, C: Column>(
    metadata: &EnsembleMetadata,
    dom_decomp: &Decomp,
    im_decomp: &Decomp,
) -> (PersistenceDiagram, PersistenceDiagram) {
    let mut im_dgm = PersistenceDiagram::default();
    let mut cod_dgm = PersistenceDiagram::default();
    (0..metadata.sz_cod).for_each(|idx| {
        if let Some(low_idx) = im_decomp.get_r_col(idx).pivot() {
            // The column is a death in the codomain.
            // We need to add the column to the codomain diagram.
            // The birth index, i.e., the index of the lowest entry in this column,
            // corresponds to a row in D_im, which has permuted rows.
            // We need the unpermuted index.
            let birth_idx = metadata.dom_first_mapping.inverse_map(low_idx).unwrap();
            cod_dgm.insert(birth_idx, Finite(idx));

            // Check if the birth simplex is in the domain.
            // If yes, then add a feature to the image diagram.
            let low_idx_in_dom = low_idx < metadata.sz_dom;
            if low_idx_in_dom {
                im_dgm.insert(birth_idx, Finite(idx));
            }
        } else {
            // The column is a birth in the codomain.
            cod_dgm.insert(idx, Infinity);

            // Check if the column is a birth in the domain.
            if metadata.col_in_dom[idx] {
                let dom_idx = metadata.dom_first_mapping.map(idx).unwrap();
                if dom_decomp.get_r_col(dom_idx).pivot().is_none() {
                    im_dgm.insert(idx, Infinity);
                }
            }
        }
    });
    (cod_dgm, im_dgm)
}

fn cokernel_diagram<Decomp: Decomposition<C>, C: Column>(
    metadata: &EnsembleMetadata,
    dom_decomp: &Decomp,
    im_decomp: &Decomp,
    cok_decomp: &Decomp,
    cod_negative_list: &[bool],
) -> PersistenceDiagram {
    let mut dgm = PersistenceDiagram::default();
    cod_negative_list
        .iter()
        .enumerate()
        .take(metadata.sz_cod)
        .for_each(|(idx, &is_death_in_cod)| {
            let is_birth_in_cod = !is_death_in_cod;
            let dom_idx = metadata.dom_first_mapping.map(idx).unwrap();
            let not_in_dom_or_neg_in_dom =
                (!metadata.col_in_dom[idx]) || dom_decomp.get_r_col(dom_idx).pivot().is_some();
            if is_birth_in_cod && not_in_dom_or_neg_in_dom {
                dgm.insert(idx, Infinity);
                return;
            }
            if is_birth_in_cod {
                return;
            }
            let low_idx_in_dom = im_decomp.get_r_col(idx).pivot().unwrap() < metadata.sz_dom;
            if !low_idx_in_dom {
                let lowest_in_r_cok = cok_decomp.get_r_col(idx).pivot().unwrap();
                dgm.insert(lowest_in_r_cok, Finite(idx));
            }
        });
    dgm
}
impl<C: Column, Algo: DecompositionAlgo<C>> DecompositionEnsemble<C, Algo> {
    pub fn all_diagrams(&self) -> DiagramEnsemble {
        let cod_diagram =
            PersistenceDiagram::from_decomposition(&self.cod).anti_transpose(self.metadata.sz_cod);
        let cod_negative_list = compute_negative_list(&self.metadata, &cod_diagram);

        let (cod_dgm, im_dgm) = codomain_image_diagram(&self.metadata, &self.dom, &self.im);
        DiagramEnsemble {
            domain: {
                let mut dgm = PersistenceDiagram::from_decomposition(&self.dom);
                unreorder_idxs(&mut dgm, &self.metadata.dom_first_mapping);
                dgm
            },
            relative: {
                let at_diagram = PersistenceDiagram::from_decomposition(&self.rel);
                let mut dgm =
                    at_diagram.anti_transpose(self.metadata.sz_cod - self.metadata.sz_dom + 1);
                unreorder_idxs(&mut dgm, &self.metadata.rel_mapping);
                dgm
            },
            image: im_dgm,
            kernel: kernel_diagram(
                &self.metadata,
                &self.ker,
                &self.dom,
                &self.im,
                &cod_negative_list,
            ),
            cokernel: cokernel_diagram(
                &self.metadata,
                &self.dom,
                &self.im,
                &self.cok,
                &cod_negative_list,
            ),
            codomain: cod_dgm,
        }
    }
}

pub fn from_file(file: &File) -> DecompositionFileFormat {
    let buf = BufReader::new(file);
    deserialize_from(buf).expect("Can't deserialize from file")
    //from_reader(file).expect("JSON deserializes")
}

impl FileEnsemble {
    pub fn all_diagrams(&self) -> DiagramEnsemble {
        let cod_diagram = PersistenceDiagram::from_decomposition(&from_file(&self.cod))
            .anti_transpose(self.metadata.sz_cod);
        debug!("Got cod");
        let cod_negative_list = compute_negative_list(&self.metadata, &cod_diagram);
        let rel_diagram = {
            let rel_decomp = from_file(&self.rel);
            let at_diagram = PersistenceDiagram::from_decomposition(&rel_decomp);
            let mut dgm =
                at_diagram.anti_transpose(self.metadata.sz_cod - self.metadata.sz_dom + 1);
            unreorder_idxs(&mut dgm, &self.metadata.rel_mapping);
            dgm
        };
        let dom_decomp = from_file(&self.dom);
        let dom_diagram = {
            let mut dgm = PersistenceDiagram::from_decomposition(&dom_decomp);
            unreorder_idxs(&mut dgm, &self.metadata.dom_first_mapping);
            dgm
        };
        let im_decomp = from_file(&self.im);
        let ker_decomp = from_file(&self.ker);
        let ker_diagram = kernel_diagram(
            &self.metadata,
            &ker_decomp,
            &dom_decomp,
            &im_decomp,
            &cod_negative_list,
        );
        drop(ker_decomp);
        let (cod_diagram, im_diagram) =
            codomain_image_diagram(&self.metadata, &dom_decomp, &im_decomp);
        let cok_decomp = from_file(&self.cok);
        let cok_diagram = cokernel_diagram(
            &self.metadata,
            &dom_decomp,
            &im_decomp,
            &cok_decomp,
            &cod_negative_list,
        );
        DiagramEnsemble {
            codomain: cod_diagram,
            domain: dom_diagram,
            relative: rel_diagram,
            image: im_diagram,
            kernel: ker_diagram,
            cokernel: cok_diagram,
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
        use crate::indexing::{AnnotatedColumn, compute_dom_first_mapping, unreorder_idxs};

        let matrix: Vec<_> = [false, true, false, true]
            .into_iter()
            .map(|in_domain| AnnotatedColumn {
                in_domain,
                col: VecColumn::from((0, vec![])),
            })
            .collect();
        let mapping = compute_dom_first_mapping(&matrix);
        let mut diagram = PersistenceDiagram([(0, Finite(3)), (1, Infinity)].into());
        unreorder_idxs(&mut diagram, &mapping);
        assert_eq!(
            diagram,
            PersistenceDiagram([(1, Finite(2)), (3, Infinity)].into())
        );
        assert_eq!(diagram.to_string(), "{1: 2, 3: Inf}");
    }
}
