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
    pub cod: PersistenceDiagram,
    pub dom: PersistenceDiagram,
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
            dgm.unpaired.insert(idx);
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
            dgm.unpaired.remove(&birth_index);
            dgm.paired.insert((birth_index, idx));
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
            cod_dgm.unpaired.remove(&birth_idx);
            cod_dgm.paired.insert((birth_idx, idx));

            // Check if the birth simplex is in the domain.
            // If yes, then add a feature to the image diagram.
            let low_idx_in_dom = low_idx < metadata.sz_dom;
            if low_idx_in_dom {
                im_dgm.unpaired.remove(&birth_idx);
                im_dgm.paired.insert((birth_idx, idx));
            }
        } else {
            // The column is a birth in the codomain.
            cod_dgm.unpaired.insert(idx);

            // Check if the column is a birth in the domain.
            if metadata.col_in_dom[idx] {
                let dom_idx = metadata.dom_first_mapping.map(idx).unwrap();
                if dom_decomp.get_r_col(dom_idx).pivot().is_none() {
                    im_dgm.unpaired.insert(idx);
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
                dgm.unpaired.insert(idx);
                return;
            }
            if is_birth_in_cod {
                return;
            }
            let low_idx_in_dom = im_decomp.get_r_col(idx).pivot().unwrap() < metadata.sz_dom;
            if !low_idx_in_dom {
                let lowest_in_r_cok = cok_decomp.get_r_col(idx).pivot().unwrap();
                dgm.unpaired.remove(&lowest_in_r_cok);
                dgm.paired.insert((lowest_in_r_cok, idx));
            }
        });
    dgm
}
impl<C: Column, Algo: DecompositionAlgo<C>> DecompositionEnsemble<C, Algo> {
    pub fn all_diagrams(&self) -> DiagramEnsemble {
        let cod_diagram = self.cod.diagram().anti_transpose(self.metadata.sz_cod);
        let cod_negative_list = compute_negative_list(&self.metadata, &cod_diagram);

        let (cod_dgm, im_dgm) = codomain_image_diagram(&self.metadata, &self.dom, &self.im);
        DiagramEnsemble {
            dom: {
                let mut dgm = self.dom.diagram();
                unreorder_idxs(&mut dgm, &self.metadata.dom_first_mapping);
                dgm
            },
            rel: {
                let at_diagram = self.rel.diagram();
                let mut dgm =
                    at_diagram.anti_transpose(self.metadata.sz_cod - self.metadata.sz_dom + 1);
                unreorder_idxs(&mut dgm, &self.metadata.rel_mapping);
                dgm
            },
            im: im_dgm,
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
            cod: cod_dgm,
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
        let cod_diagram = from_file(&self.cod)
            .diagram()
            .anti_transpose(self.metadata.sz_cod);
        debug!("Got cod");
        let cod_negative_list = compute_negative_list(&self.metadata, &cod_diagram);
        let rel_diagram = {
            let rel_decomp = from_file(&self.rel);
            let at_diagram = rel_decomp.diagram();
            let mut dgm =
                at_diagram.anti_transpose(self.metadata.sz_cod - self.metadata.sz_dom + 1);
            unreorder_idxs(&mut dgm, &self.metadata.rel_mapping);
            dgm
        };
        let dom_decomp = from_file(&self.dom);
        let dom_diagram = {
            let mut dgm = dom_decomp.diagram();
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
            cod: cod_diagram,
            dom: dom_diagram,
            rel: rel_diagram,
            im: im_diagram,
            ker: ker_diagram,
            cok: cok_diagram,
        }
    }
}
