use lophat::{anti_transpose_diagram, Column, DiagramReadOff, PersistenceDiagram};
use pyo3::prelude::*;

use crate::{
    ensemble::DecompositionEnsemble,
    indexing::{unreorder_idxs, IndexMapping},
};

#[pyclass(get_all)]
#[derive(Debug, Clone)]
pub struct DiagramEnsemble {
    pub f: PersistenceDiagram,
    pub g: PersistenceDiagram,
    pub im: PersistenceDiagram,
    pub ker: PersistenceDiagram,
    pub cok: PersistenceDiagram,
    pub rel: PersistenceDiagram,
}

impl DecompositionEnsemble {
    // Since we anti-transposed f, to check whether a column is negative in f
    // we need to check the diagram of f after reindexing
    fn compute_negative_list(&self, diagram: &PersistenceDiagram) -> Vec<bool> {
        let mut negative_list: Vec<bool> = vec![false; self.size_of_k];
        for (_birth, death) in diagram.paired.iter() {
            negative_list[*death] = true;
        }
        negative_list
    }

    fn is_kernel_birth(&self, idx: usize, f_negative_list: &Vec<bool>) -> bool {
        let in_l = self.g_elements[idx];
        if in_l {
            return false;
        }
        let negative_in_f = f_negative_list[idx];
        if !negative_in_f {
            return false;
        }
        let lowest_rim_in_l = self.im.r[idx].pivot().unwrap() < self.size_of_l;
        if !lowest_rim_in_l {
            return false;
        }
        return true;
    }

    fn is_kernel_death(&self, idx: usize, f_negative_list: &Vec<bool>) -> bool {
        let in_l = self.g_elements[idx];
        if !in_l {
            return false;
        }
        let g_index = self.l_first_mapping.map(idx).unwrap();
        let negative_in_g = self.g.r[g_index].pivot().is_some();
        if !negative_in_g {
            return false;
        }
        let negative_in_f = f_negative_list[idx];
        if negative_in_f {
            return false;
        }
        return true;
    }

    fn kernel_diagram(&self, f_negative_list: &Vec<bool>) -> PersistenceDiagram {
        let mut dgm = PersistenceDiagram::default();
        for idx in 0..self.size_of_k {
            if self.is_kernel_birth(idx, f_negative_list) {
                dgm.unpaired.insert(idx);
                continue;
            }
            if self.is_kernel_death(idx, f_negative_list) {
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

    fn image_diagram(&self, f_negative_list: &Vec<bool>) -> PersistenceDiagram {
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
            let neg_in_f = f_negative_list[idx];
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

    fn cokernel_diagram(&self, f_negative_list: &Vec<bool>) -> PersistenceDiagram {
        let mut dgm = PersistenceDiagram::default();
        for idx in 0..self.size_of_k {
            let pos_in_f = !f_negative_list[idx];
            let g_idx = self.l_first_mapping.map(idx).unwrap();
            let not_in_l_or_neg_in_g = (!self.g_elements[idx]) || self.g.r[g_idx].pivot().is_some();
            if pos_in_f && not_in_l_or_neg_in_g {
                dgm.unpaired.insert(idx);
                continue;
            }
            if pos_in_f {
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
        let f_diagram = {
            let at_diagram = self.f.diagram();
            anti_transpose_diagram(at_diagram, self.size_of_k)
        };
        let f_negative_list = self.compute_negative_list(&f_diagram);
        DiagramEnsemble {
            g: {
                let mut dgm = self.g.diagram();
                unreorder_idxs(&mut dgm, &self.l_first_mapping);
                dgm
            },
            rel: {
                let at_diagram = self.rel.diagram();
                let mut dgm =
                    anti_transpose_diagram(at_diagram, self.size_of_k - self.size_of_l + 1);
                unreorder_idxs(&mut dgm, &self.rel_mapping);
                dgm
            },
            im: self.image_diagram(&f_negative_list),
            ker: self.kernel_diagram(&f_negative_list),
            cok: self.cokernel_diagram(&f_negative_list),
            f: f_diagram,
        }
    }
}
