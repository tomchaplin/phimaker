use bincode::serialize_into;
use log::debug;
use serde::Serialize;
use std::{fs::File, io::BufWriter, marker::PhantomData, thread};

use lophat::{
    algorithms::DecompositionAlgo,
    columns::{Column, VecColumn},
    options::LoPhatOptions,
    utils::anti_transpose,
};

use crate::{
    builders::{build_d_cok, build_d_dom, build_d_im, build_d_ker, build_d_rel},
    indexing::{
        build_kernel_mapping, build_rel_mapping, compute_dom_first_mapping, AnnotatedColumn,
        VectorMapping,
    },
};

#[derive(Debug)]
pub struct EnsembleMetadata {
    pub dom_first_mapping: VectorMapping,
    pub kernel_mapping: VectorMapping,
    pub rel_mapping: VectorMapping,
    pub col_in_dom: Vec<bool>,
    pub sz_dom: usize,
    pub sz_cod: usize,
}

#[derive(Debug)]
pub struct DecompositionEnsemble<C, Algo>
where
    C: Column,
    Algo: DecompositionAlgo<C>,
{
    pub cod: Algo::Decomposition,
    pub dom: Algo::Decomposition,
    pub im: Algo::Decomposition,
    pub ker: Algo::Decomposition,
    pub cok: Algo::Decomposition,
    pub rel: Algo::Decomposition,
    pub metadata: EnsembleMetadata,
    phantom: PhantomData<C>,
}

#[derive(Debug)]
pub struct FileEnsemble {
    pub f: File,
    pub g: File,
    pub im: File,
    pub ker: File,

    pub cok: File,

    pub rel: File,

    pub metadata: EnsembleMetadata,
}

pub fn decompose_cod<Algo: DecompositionAlgo<VecColumn, Options = LoPhatOptions>>(
    df: &[VecColumn],
    base_options: Algo::Options,
) -> Algo::Decomposition {
    // Decompose Df
    // Df is a chain complex so can compute anti-transpose instead
    let df_at = anti_transpose(df);
    let out = Algo::init(Some(base_options))
        .add_cols(df_at.into_iter())
        .decompose();
    debug!("Decomposed cod");
    out
}

pub fn decompose_dom_cok<Algo: DecompositionAlgo<VecColumn, Options = LoPhatOptions>>(
    d_cod: &[VecColumn],
    col_in_dom: &[bool],
    dom_first_mapping: &VectorMapping,
    base_options: Algo::Options,
) -> (Algo::Decomposition, Algo::Decomposition) {
    // Decompose D_dom
    // Need to use v columns of D_dom later, so no anti-transpose
    let d_dom = build_d_dom(d_cod, col_in_dom, dom_first_mapping);
    let d_dom_decomp_options = LoPhatOptions {
        maintain_v: true,
        ..base_options
    };
    let decomp_d_dom = Algo::init(Some(d_dom_decomp_options)).add_cols(d_dom).decompose();
    debug!("Decomposed dom");

    // Decompose d_cok
    let d_cok = build_d_cok(d_cod, &decomp_d_dom, col_in_dom, dom_first_mapping);
    let d_cok_decomp_options = LoPhatOptions {
        clearing: false,
        ..base_options
    };
    let decomp_d_cok = Algo::init(Some(d_cok_decomp_options)).add_cols(d_cok).decompose();
    debug!("Decomposed cok");
    (decomp_d_dom, decomp_d_cok)
}
pub fn decompose_ker<Algo: DecompositionAlgo<VecColumn, Options = LoPhatOptions>>(
    d_cod: &[VecColumn],
    dom_first_mapping: &VectorMapping,
    sz_cod: usize,
    base_options: Algo::Options,
) -> (Algo::Decomposition, Algo::Decomposition, VectorMapping) {
    // Decompose dim
    // Need to use v columns of Dim later, also no anti-transpose or clearing since D^2 != 0
    let d_im = build_d_im(d_cod, dom_first_mapping);
    let d_im_decomp_options = LoPhatOptions {
        maintain_v: true,
        clearing: false,
        ..base_options
    };
    let decomp_d_im = Algo::init(Some(d_im_decomp_options)).add_cols(d_im).decompose();
    debug!("Decomposed im");

    // Decompose dker
    let d_ker = build_d_ker(&decomp_d_im, dom_first_mapping);
    let d_ker_options = LoPhatOptions {
        clearing: false,                // Not a chain complex so no clearing
        column_height: Some(sz_cod), // Non-square matrix
        ..base_options
    };
    let decomp_d_ker = Algo::init(Some(d_ker_options)).add_cols(d_ker).decompose();
    let ker_mapping = build_kernel_mapping(&decomp_d_im);
    debug!("Decomposed ker");
    (decomp_d_im, decomp_d_ker, ker_mapping)
}

pub fn decompose_rel<Algo: DecompositionAlgo<VecColumn, Options = LoPhatOptions>>(
    d_cod: &[VecColumn],
    col_in_dom: &[bool],
    sz_dom: usize,
    sz_cod: usize,
    base_options: LoPhatOptions,
) -> (Algo::Decomposition, VectorMapping) {
    let (rel_mapping, l_index) = build_rel_mapping(d_cod, col_in_dom, sz_dom, sz_cod);
    let d_rel = build_d_rel(d_cod, col_in_dom, &rel_mapping, l_index).collect::<Vec<_>>();
    // Chain complex so can use clearing and anti-transpose
    let d_rel_anti_transpose = anti_transpose(&d_rel);
    let decomp_d_rel = Algo::init(Some(base_options))
        .add_cols(d_rel_anti_transpose.into_iter())
        .decompose();
    debug!("Decomposed rel");
    (decomp_d_rel, rel_mapping)
}

pub fn all_decompositions<Algo: DecompositionAlgo<VecColumn, Options = LoPhatOptions>>(
    matrix: Vec<AnnotatedColumn<VecColumn>>,
    num_threads: usize,
) -> DecompositionEnsemble<VecColumn, Algo>
where
    Algo::Decomposition: Send,
{
    let base_options = LoPhatOptions {
        maintain_v: false,   // Only turn on maintain_v on threads where we need it
        column_height: None, // Assume square unless told otherwise
        num_threads,
        min_chunk_len: 10000,
        clearing: true, // Clear whenever we can
    };

    let dom_first_mapping = compute_dom_first_mapping(&matrix);

    let (col_in_dom, d_cod): (Vec<_>, Vec<_>) = matrix
        .into_iter()
        .map(|annotated_col| (annotated_col.in_domain, annotated_col.col))
        .unzip();

    let sz_dom = col_in_dom.iter().filter(|in_dom| **in_dom).count();
    let sz_cod = d_cod.len();

    let (cod, (dom, cok), (im, ker, kernel_mapping), (rel, rel_mapping)) = thread::scope(|s| {
        let thread1 = s.spawn(|| decompose_cod::<Algo>(&d_cod, base_options));

        let thread2 = s.spawn(|| {
            decompose_dom_cok::<Algo>(&d_cod, &col_in_dom, &dom_first_mapping, base_options)
        });

        let thread3 =
            s.spawn(|| decompose_ker::<Algo>(&d_cod, &dom_first_mapping, sz_cod, base_options));

        let thread4 = s.spawn(|| {
            decompose_rel::<Algo>(&d_cod, &col_in_dom, sz_dom, sz_cod, base_options)
        });

        (
            thread1.join().unwrap(),
            thread2.join().unwrap(),
            thread3.join().unwrap(),
            thread4.join().unwrap(),
        )
    });
    DecompositionEnsemble {
        cod,
        dom,
        im,
        ker,
        cok,
        rel,
        metadata: EnsembleMetadata {
            col_in_dom,
            dom_first_mapping,
            kernel_mapping,
            rel_mapping,
            sz_dom,
            sz_cod,
        },
        phantom: PhantomData,
    }
}

pub fn to_file<Algo: Serialize>(algo: Algo) -> File {
    let mut file_write = tempfile::NamedTempFile::new().expect("Can get temp file");
    println!("Writing to {:?}", file_write.path());
    // We reopen so that we can hold onto the file for later reading
    let file_read = file_write.reopen().expect("Can reopen tempfile");
    {
        let mut buf = BufWriter::new(&mut file_write);
        serialize_into(&mut buf, &algo).expect("Can serialize to file");
    }
    // Explicitly release memory
    drop(algo);
    file_read
}

pub fn all_decompositions_slow<Algo>(
    matrix: Vec<AnnotatedColumn<VecColumn>>,
    num_threads: usize,
) -> FileEnsemble
where
    Algo: DecompositionAlgo<VecColumn, Options = LoPhatOptions>,
    Algo::Decomposition: Serialize + Send,
{
    let base_options = LoPhatOptions {
        maintain_v: false,   // Only turn on maintain_v on threads where we need it
        column_height: None, // Assume square unless told otherwise
        num_threads,
        min_chunk_len: 10000,
        clearing: true, // Clear whenever we can
    };

    let l_first_mapping = compute_l_first_mapping(&matrix);

    let (g_elements, df): (Vec<_>, Vec<_>) = matrix
        .into_iter()
        .map(|anncol| (anncol.in_g, anncol.col))
        .unzip();

    let size_of_l = g_elements.iter().filter(|in_g| **in_g).count();
    let size_of_k = df.len();

    let f = decompose_cod::<Algo>(&df, base_options);
    let f = to_file(f);
    let (g, cok) =
        decompose_dom_cok::<Algo>(&df, &g_elements, &l_first_mapping, base_options);
    let g = to_file(g);
    let cok = to_file(cok);
    let (im, ker, kernel_mapping) =
        decompose_ker::<Algo>(&df, &l_first_mapping, size_of_k, base_options);
    let im = to_file(im);
    let ker = to_file(ker);
    let (rel, rel_mapping) =
        decompose_rel::<Algo>(&df, &g_elements, size_of_l, size_of_k, base_options);
    let rel = to_file(rel);

    FileEnsemble {
        f,
        g,
        im,
        ker,
        cok,
        rel,
        metadata: EnsembleMetadata {
            col_in_dom: g_elements,
            dom_first_mapping: l_first_mapping,
            kernel_mapping,
            rel_mapping,
            sz_dom: size_of_l,
            sz_cod: size_of_k,
        },
    }
}
