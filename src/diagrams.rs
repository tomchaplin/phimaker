use bincode::deserialize_from;
use std::{fs::File, io::BufReader};

use log::debug;

use lophat::{
    algorithms::{Decomposition, DecompositionAlgo},
    columns::Column,
    utils::{DecompositionFileFormat, PersistenceDiagram},
};
use pyo3::prelude::*;

use crate::{
    ensemble::{DecompositionEnsemble, EnsembleMetadata, FileEnsemble},
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

/// Returns the list of negative indices in the diagram, i.e., indices
/// of non-zero columns. Such columns represent deaths in the diagram.
fn compute_negative_list(metadata: &EnsembleMetadata, diagram: &PersistenceDiagram) -> Vec<bool> {
    let mut negative_list: Vec<bool> = vec![false; metadata.sz_cod];
    for (_birth, death) in diagram.paired.iter() {
        negative_list[*death] = true;
    }
    negative_list
}

fn is_kernel_birth<Decomp: Decomposition<C>, C: Column>(
    idx: usize,
    metadata: &EnsembleMetadata,
    f_negative_list: &[bool],
    im: &Decomp,
) -> bool {
    let in_l = metadata.col_in_dom[idx];
    if in_l {
        return false;
    }
    let negative_in_f = f_negative_list[idx];
    if !negative_in_f {
        return false;
    }
    let lowest_rim_in_l = im.get_r_col(idx).pivot().unwrap() < metadata.sz_dom;
    if !lowest_rim_in_l {
        return false;
    }
    true
}

fn is_kernel_death<Decomp: Decomposition<C>, C: Column>(
    idx: usize,
    metadata: &EnsembleMetadata,
    g: &Decomp,
    f_negative_list: &[bool],
) -> bool {
    let in_l = metadata.col_in_dom[idx];
    if !in_l {
        return false;
    }
    let g_index = metadata.dom_first_mapping.map(idx).unwrap();
    let negative_in_g = g.get_r_col(g_index).pivot().is_some();
    if !negative_in_g {
        return false;
    }
    let negative_in_f = f_negative_list[idx];
    if negative_in_f {
        return false;
    }
    true
}

fn kernel_diagram<Decomp: Decomposition<C>, C: Column>(
    metadata: &EnsembleMetadata,
    ker: &Decomp,
    g: &Decomp,
    im: &Decomp,
    f_negative_list: &[bool],
) -> PersistenceDiagram {
    let mut dgm = PersistenceDiagram::default();
    for idx in 0..metadata.sz_cod {
        if is_kernel_birth(idx, metadata, f_negative_list, im) {
            dgm.unpaired.insert(idx);
            continue;
        }
        if is_kernel_death(idx, metadata, g, f_negative_list) {
            // TODO: Problem kernel columns have different indexing to f
            let ker_idx = metadata.kernel_mapping.map(idx).unwrap();
            let g_birth_index = ker.get_r_col(ker_idx).pivot().unwrap();
            let birth_index = metadata.dom_first_mapping.inverse_map(g_birth_index).unwrap();
            dgm.unpaired.remove(&birth_index);
            dgm.paired.insert((birth_index, idx));
        }
    }
    dgm
}

fn image_diagram<Decomp: Decomposition<C>, C: Column>(
    metadata: &EnsembleMetadata,
    dom_decomp: &Decomp,
    im_decomp: &Decomp,
    cod_negative_list: &[bool],
) -> PersistenceDiagram {
    // TODO: we don't need cod_negative_list or dom_decomp
    // We only need the negative columns (pivot columns) in R_im
    let mut dgm = PersistenceDiagram::default();
    cod_negative_list
        .iter()
        .enumerate()
        .take(metadata.sz_cod)
        .for_each(|(idx, &neg_in_cod)| {
            // for idx in 0..metadata.size_of_k {
            if metadata.col_in_dom[idx] {
                let dom_idx = metadata.dom_first_mapping.map(idx).unwrap();
                let is_cycle = dom_decomp.get_r_col(dom_idx).pivot().is_none();
                if is_cycle {
                    dgm.unpaired.insert(idx);
                    return;
                }
            }
            if neg_in_cod {
                let lowest_in_r_im = im_decomp.get_r_col(idx).pivot().unwrap();
                let lowest_r_im_in_l = lowest_in_r_im < metadata.sz_dom;
                if !lowest_r_im_in_l {
                    return;
                }
                let birth_idx = metadata.dom_first_mapping.inverse_map(lowest_in_r_im).unwrap();
                dgm.unpaired.remove(&birth_idx);
                dgm.paired.insert((birth_idx, idx));
            }
        });
    dgm
}

fn cokernel_diagram<Decomp: Decomposition<C>, C: Column>(
    metadata: &EnsembleMetadata,
    g: &Decomp,
    im: &Decomp,
    cok: &Decomp,
    f_negative_list: &[bool],
) -> PersistenceDiagram {
    let mut dgm = PersistenceDiagram::default();
    f_negative_list
        .iter()
        .enumerate()
        .take(metadata.sz_cod)
        .for_each(|(idx, &neg_in_f)| {
            let pos_in_f = !neg_in_f;
            let g_idx = metadata.dom_first_mapping.map(idx).unwrap();
            let not_in_l_or_neg_in_g =
                (!metadata.col_in_dom[idx]) || g.get_r_col(g_idx).pivot().is_some();
            if pos_in_f && not_in_l_or_neg_in_g {
                dgm.unpaired.insert(idx);
                return;
            }
            if pos_in_f {
                return;
            }
            let lowest_rim_in_l = im.get_r_col(idx).pivot().unwrap() < metadata.sz_dom;
            if !lowest_rim_in_l {
                let lowest_in_rcok = cok.get_r_col(idx).pivot().unwrap();
                dgm.unpaired.remove(&lowest_in_rcok);
                dgm.paired.insert((lowest_in_rcok, idx));
            }
        });
    dgm
}
impl<C: Column, Algo: DecompositionAlgo<C>> DecompositionEnsemble<C, Algo> {
    pub fn all_diagrams(&self) -> DiagramEnsemble {
        let cod_diagram = self.cod.diagram().anti_transpose(self.metadata.sz_cod);
        let cod_negative_list = compute_negative_list(&self.metadata, &cod_diagram);

        DiagramEnsemble {
            g: {
                let mut dgm = self.dom.diagram();
                unreorder_idxs(&mut dgm, &self.metadata.dom_first_mapping);
                dgm
            },
            rel: {
                let at_diagram = self.rel.diagram();
                let mut dgm = at_diagram
                    .anti_transpose(self.metadata.sz_cod - self.metadata.sz_dom + 1);
                unreorder_idxs(&mut dgm, &self.metadata.rel_mapping);
                dgm
            },
            im: image_diagram(&self.metadata, &self.dom, &self.im, &cod_negative_list),
            ker: kernel_diagram(
                &self.metadata,
                &self.ker,
                &self.dom,
                &self.im,
                &cod_negative_list,
            ),
            cok: cokernel_diagram(
                &self.metadata,
                &self.dom,
                &self.im,
                &self.cok,
                &cod_negative_list,
            ),
            f: cod_diagram,
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
        let f_diagram = {
            let f_decomp = from_file(&self.f);
            let at_diagram = f_decomp.diagram();
            at_diagram.anti_transpose(self.metadata.sz_cod)
        };
        debug!("Got f");
        let f_negative_list = compute_negative_list(&self.metadata, &f_diagram);
        let rel_diagram = {
            let rel_decomp = from_file(&self.rel);
            let at_diagram = rel_decomp.diagram();
            let mut dgm =
                at_diagram.anti_transpose(self.metadata.sz_cod - self.metadata.sz_dom + 1);
            unreorder_idxs(&mut dgm, &self.metadata.rel_mapping);
            dgm
        };
        let g_decomp = from_file(&self.g);
        let g_diagram = {
            let mut dgm = g_decomp.diagram();
            unreorder_idxs(&mut dgm, &self.metadata.dom_first_mapping);
            dgm
        };
        let im_decomp = from_file(&self.im);
        let ker_decomp = from_file(&self.ker);
        let ker_diagram = kernel_diagram(
            &self.metadata,
            &ker_decomp,
            &g_decomp,
            &im_decomp,
            &f_negative_list,
        );
        drop(ker_decomp);
        let im_diagram = image_diagram(&self.metadata, &g_decomp, &im_decomp, &f_negative_list);
        let cok_decomp = from_file(&self.cok);
        let cok_diagram = cokernel_diagram(
            &self.metadata,
            &g_decomp,
            &im_decomp,
            &cok_decomp,
            &f_negative_list,
        );
        DiagramEnsemble {
            f: f_diagram,
            g: g_diagram,
            rel: rel_diagram,
            im: im_diagram,
            ker: ker_diagram,
            cok: cok_diagram,
        }
    }
}
