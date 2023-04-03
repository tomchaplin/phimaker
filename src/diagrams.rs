use lophat::{Column, DiagramReadOff, PersistenceDiagram};
use pyo3::prelude::*;

use crate::{
    ensemble::DecompositionEnsemble,
    indexing::{unreorder_idxs, IndexMapping},
};

#[pyclass]
#[derive(Debug, Clone)]
pub struct DiagramEnsemble {
    #[pyo3(get)]
    pub f: PersistenceDiagram,
    #[pyo3(get)]
    pub g: PersistenceDiagram,
    #[pyo3(get)]
    pub im: PersistenceDiagram,
    #[pyo3(get)]
    pub ker: PersistenceDiagram,
    #[pyo3(get)]
    pub cok: PersistenceDiagram,
    #[pyo3(get)]
    pub rel: PersistenceDiagram,
}

impl DecompositionEnsemble {
    fn is_kernel_birth(&self, idx: usize) -> bool {
        let in_l = self.g_elements[idx];
        if in_l {
            return false;
        }
        let negative_in_f = self.f.r[idx].pivot().is_some();
        if !negative_in_f {
            return false;
        }
        let lowest_rim_in_l = self.im.r[idx].pivot().unwrap() < self.size_of_l;
        if !lowest_rim_in_l {
            return false;
        }
        return true;
    }

    fn is_kernel_death(&self, idx: usize) -> bool {
        let in_l = self.g_elements[idx];
        if !in_l {
            return false;
        }
        let g_index = self.l_first_mapping.map(idx).unwrap();
        let negative_in_g = self.g.r[g_index].pivot().is_some();
        if !negative_in_g {
            return false;
        }
        let negative_in_f = self.f.r[idx].pivot().is_some();
        if negative_in_f {
            return false;
        }
        return true;
    }

    fn kernel_diagram(&self) -> PersistenceDiagram {
        let mut dgm = PersistenceDiagram::default();
        for idx in 0..self.size_of_k {
            if self.is_kernel_birth(idx) {
                dgm.unpaired.insert(idx);
                continue;
            }
            if self.is_kernel_death(idx) {
                // TODO: Problem kernel columns have different indexing to f
                let ker_idx = self.kernel_mapping.map(idx).unwrap();
                let g_birth_index = self.ker.r[ker_idx].pivot().unwrap();
                let birth_index = self.l_first_mapping.inverse_map(g_birth_index).unwrap();
                dgm.unpaired.remove(&birth_index);
                dgm.paired.insert((birth_index, idx));
            }
        }
        dgm
    }

    fn image_diagram(&self) -> PersistenceDiagram {
        let mut dgm = PersistenceDiagram::default();
        for idx in 0..self.size_of_k {
            if self.g_elements[idx] {
                let g_idx = self.l_first_mapping.map(idx).unwrap();
                let pos_in_g = self.g.r[g_idx].pivot().is_none();
                if pos_in_g {
                    dgm.unpaired.insert(idx);
                    continue;
                }
            }
            let neg_in_f = self.f.r[idx].pivot().is_some();
            if neg_in_f {
                let lowest_in_rim = self.im.r[idx].pivot().unwrap();
                let lowest_rim_in_l = lowest_in_rim < self.size_of_l;
                if !lowest_rim_in_l {
                    continue;
                }
                let birth_idx = self.l_first_mapping.inverse_map(lowest_in_rim).unwrap();
                dgm.unpaired.remove(&birth_idx);
                dgm.paired.insert((birth_idx, idx));
            }
        }
        dgm
    }

    fn cokernel_diagram(&self) -> PersistenceDiagram {
        let mut dgm = PersistenceDiagram::default();
        for idx in 0..self.size_of_k {
            let pos_in_f = self.f.r[idx].pivot().is_none();
            let g_idx = self.l_first_mapping.map(idx).unwrap();
            let not_in_l_or_neg_in_g = (!self.g_elements[idx]) || self.g.r[g_idx].pivot().is_some();
            if pos_in_f && not_in_l_or_neg_in_g {
                dgm.unpaired.insert(idx);
                continue;
            }
            let neg_in_f = self.f.r[idx].pivot().is_some();
            if !neg_in_f {
                continue;
            }
            let lowest_rim_in_l = self.im.r[idx].pivot().unwrap() < self.size_of_l;
            if !lowest_rim_in_l {
                let lowest_in_rcok = self.cok.r[idx].pivot().unwrap();
                dgm.unpaired.remove(&lowest_in_rcok);
                dgm.paired.insert((lowest_in_rcok, idx));
            }
        }
        dgm
    }

    pub fn all_diagrams(&self) -> DiagramEnsemble {
        DiagramEnsemble {
            f: self.f.diagram(),
            g: {
                let mut dgm = self.g.diagram();
                unreorder_idxs(&mut dgm, &self.l_first_mapping);
                dgm
            },
            rel: {
                let mut dgm = self.rel.diagram();
                unreorder_idxs(&mut dgm, &self.rel_mapping);
                dgm
            },
            im: self.image_diagram(),
            ker: self.kernel_diagram(),
            cok: self.cokernel_diagram(),
        }
    }
}
