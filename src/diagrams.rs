use bincode::deserialize_from;
use std::{fs::File, io::BufReader};

use log::debug;

use lophat::{
    algorithms::RVDecomposition, columns::Column, utils::PersistenceDiagram, utils::RVDFileFormat,
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

// Since we anti-transposed f, to check whether a column is negative in f
// we need to check the diagram of f after reindexing
fn compute_negative_list(metadata: &EnsembleMetadata, diagram: &PersistenceDiagram) -> Vec<bool> {
    let mut negative_list: Vec<bool> = vec![false; metadata.size_of_k];
    for (_birth, death) in diagram.paired.iter() {
        negative_list[*death] = true;
    }
    negative_list
}

fn is_kernel_birth<Decomp: RVDecomposition<C>, C: Column>(
    idx: usize,
    metadata: &EnsembleMetadata,
    f_negative_list: &Vec<bool>,
    im: &Decomp,
) -> bool {
    let in_l = metadata.g_elements[idx];
    if in_l {
        return false;
    }
    let negative_in_f = f_negative_list[idx];
    if !negative_in_f {
        return false;
    }
    let lowest_rim_in_l = im.get_r_col(idx).pivot().unwrap() < metadata.size_of_l;
    if !lowest_rim_in_l {
        return false;
    }
    return true;
}

fn is_kernel_death<Decomp: RVDecomposition<C>, C: Column>(
    idx: usize,
    metadata: &EnsembleMetadata,
    g: &Decomp,
    f_negative_list: &Vec<bool>,
) -> bool {
    let in_l = metadata.g_elements[idx];
    if !in_l {
        return false;
    }
    let g_index = metadata.l_first_mapping.map(idx).unwrap();
    let negative_in_g = g.get_r_col(g_index).pivot().is_some();
    if !negative_in_g {
        return false;
    }
    let negative_in_f = f_negative_list[idx];
    if negative_in_f {
        return false;
    }
    return true;
}

fn kernel_diagram<Decomp: RVDecomposition<C>, C: Column>(
    metadata: &EnsembleMetadata,
    ker: &Decomp,
    g: &Decomp,
    im: &Decomp,
    f_negative_list: &Vec<bool>,
) -> PersistenceDiagram {
    let mut dgm = PersistenceDiagram::default();
    for idx in 0..metadata.size_of_k {
        if is_kernel_birth(idx, metadata, f_negative_list, im) {
            dgm.unpaired.insert(idx);
            continue;
        }
        if is_kernel_death(idx, metadata, g, f_negative_list) {
            // TODO: Problem kernel columns have different indexing to f
            let ker_idx = metadata.kernel_mapping.map(idx).unwrap();
            let g_birth_index = ker.get_r_col(ker_idx).pivot().unwrap();
            let birth_index = metadata.l_first_mapping.inverse_map(g_birth_index).unwrap();
            dgm.unpaired.remove(&birth_index);
            dgm.paired.insert((birth_index, idx));
        }
    }
    dgm
}

fn image_diagram<Decomp: RVDecomposition<C>, C: Column>(
    metadata: &EnsembleMetadata,
    g: &Decomp,
    im: &Decomp,
    f_negative_list: &Vec<bool>,
) -> PersistenceDiagram {
    let mut dgm = PersistenceDiagram::default();
    for idx in 0..metadata.size_of_k {
        if metadata.g_elements[idx] {
            let g_idx = metadata.l_first_mapping.map(idx).unwrap();
            let pos_in_g = g.get_r_col(g_idx).pivot().is_none();
            if pos_in_g {
                dgm.unpaired.insert(idx);
                continue;
            }
        }
        let neg_in_f = f_negative_list[idx];
        if neg_in_f {
            let lowest_in_rim = im.get_r_col(idx).pivot().unwrap();
            let lowest_rim_in_l = lowest_in_rim < metadata.size_of_l;
            if !lowest_rim_in_l {
                continue;
            }
            let birth_idx = metadata.l_first_mapping.inverse_map(lowest_in_rim).unwrap();
            dgm.unpaired.remove(&birth_idx);
            dgm.paired.insert((birth_idx, idx));
        }
    }
    dgm
}

fn cokernel_diagram<Decomp: RVDecomposition<C>, C: Column>(
    metadata: &EnsembleMetadata,
    g: &Decomp,
    im: &Decomp,
    cok: &Decomp,
    f_negative_list: &Vec<bool>,
) -> PersistenceDiagram {
    let mut dgm = PersistenceDiagram::default();
    for idx in 0..metadata.size_of_k {
        let pos_in_f = !f_negative_list[idx];
        let g_idx = metadata.l_first_mapping.map(idx).unwrap();
        let not_in_l_or_neg_in_g =
            (!metadata.g_elements[idx]) || g.get_r_col(g_idx).pivot().is_some();
        if pos_in_f && not_in_l_or_neg_in_g {
            dgm.unpaired.insert(idx);
            continue;
        }
        if pos_in_f {
            continue;
        }
        let lowest_rim_in_l = im.get_r_col(idx).pivot().unwrap() < metadata.size_of_l;
        if !lowest_rim_in_l {
            let lowest_in_rcok = cok.get_r_col(idx).pivot().unwrap();
            dgm.unpaired.remove(&lowest_in_rcok);
            dgm.paired.insert((lowest_in_rcok, idx));
        }
    }
    dgm
}
impl<C: Column, Algo: RVDecomposition<C>> DecompositionEnsemble<C, Algo> {
    pub fn all_diagrams(&self) -> DiagramEnsemble {
        let f_diagram = {
            let at_diagram = self.f.diagram();
            at_diagram.anti_transpose(self.metadata.size_of_k)
        };
        let f_negative_list = compute_negative_list(&self.metadata, &f_diagram);

        DiagramEnsemble {
            g: {
                let mut dgm = self.g.diagram();
                unreorder_idxs(&mut dgm, &self.metadata.l_first_mapping);
                dgm
            },
            rel: {
                let at_diagram = self.rel.diagram();
                let mut dgm = at_diagram
                    .anti_transpose(self.metadata.size_of_k - self.metadata.size_of_l + 1);
                unreorder_idxs(&mut dgm, &self.metadata.rel_mapping);
                dgm
            },
            im: image_diagram(&self.metadata, &self.g, &self.im, &f_negative_list),
            ker: kernel_diagram(
                &self.metadata,
                &self.ker,
                &self.g,
                &self.im,
                &f_negative_list,
            ),
            cok: cokernel_diagram(
                &self.metadata,
                &self.g,
                &self.im,
                &self.cok,
                &f_negative_list,
            ),
            f: f_diagram,
        }
    }
}

pub fn from_file(file: &File) -> RVDFileFormat {
    let buf = BufReader::new(file);
    deserialize_from(buf).expect("Can desereialize from file")
    //from_reader(file).expect("JSON deserializes")
}

impl FileEnsemble {
    pub fn all_diagrams(&self) -> DiagramEnsemble {
        let f_diagram = {
            let f_decomp = from_file(&self.f);
            let at_diagram = f_decomp.diagram();
            at_diagram.anti_transpose(self.metadata.size_of_k)
        };
        debug!("Got f");
        let f_negative_list = compute_negative_list(&self.metadata, &f_diagram);
        let rel_diagram = {
            let rel_decomp = from_file(&self.rel);
            let at_diagram = rel_decomp.diagram();
            let mut dgm =
                at_diagram.anti_transpose(self.metadata.size_of_k - self.metadata.size_of_l + 1);
            unreorder_idxs(&mut dgm, &self.metadata.rel_mapping);
            dgm
        };
        let g_decomp = from_file(&self.g);
        let g_diagram = {
            let mut dgm = g_decomp.diagram();
            unreorder_idxs(&mut dgm, &self.metadata.l_first_mapping);
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
