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
    builders::{build_dcok, build_dg, build_dim, build_dker, build_drel},
    indexing::{
        build_kernel_mapping, build_rel_mapping, compute_l_first_mapping, AnnotatedColumn,
        VectorMapping,
    },
};

#[derive(Debug)]
pub struct EnsembleMetadata {
    pub l_first_mapping: VectorMapping,
    pub kernel_mapping: VectorMapping,
    pub rel_mapping: VectorMapping,
    pub g_elements: Vec<bool>,
    pub size_of_l: usize,
    pub size_of_k: usize,
}

#[derive(Debug)]
pub struct DecompositionEnsemble<C, Algo>
where
    C: Column,
    Algo: DecompositionAlgo<C>,
{
    pub f: Algo::Decomposition,
    pub g: Algo::Decomposition,
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

pub fn decompose_domain<Algo: DecompositionAlgo<VecColumn, Options = LoPhatOptions>>(
    df: &[VecColumn],
    base_options: Algo::Options,
) -> Algo::Decomposition {
    // Decompose Df
    // Df is a chain complex so can compute anti-transpose instead
    let df_at = anti_transpose(df);
    let out = Algo::init(Some(base_options))
        .add_cols(df_at.into_iter())
        .decompose();
    debug!("Decomposed f");
    out
}

pub fn decompose_cokernel_codomain<Algo: DecompositionAlgo<VecColumn, Options = LoPhatOptions>>(
    df: &[VecColumn],
    g_elements: &[bool],
    l_first_mapping: &VectorMapping,
    base_options: Algo::Options,
) -> (Algo::Decomposition, Algo::Decomposition) {
    // Decompose Dg
    // Need to use v columns of Dg later, so no anti-transpose
    let dg = build_dg(df, g_elements, l_first_mapping);
    let dg_options = LoPhatOptions {
        maintain_v: true,
        ..base_options
    };
    let decomp_dg = Algo::init(Some(dg_options)).add_cols(dg).decompose();
    debug!("Decomposed g");

    // Decompose d_cok
    let d_cok = build_dcok(df, &decomp_dg, g_elements, l_first_mapping);
    let dcok_options = LoPhatOptions {
        clearing: false,
        ..base_options
    };
    let decompose_dcok = Algo::init(Some(dcok_options)).add_cols(d_cok).decompose();
    debug!("Decomposed cok");
    (decomp_dg, decompose_dcok)
}
pub fn decompose_kernel<Algo: DecompositionAlgo<VecColumn, Options = LoPhatOptions>>(
    df: &[VecColumn],
    l_first_mapping: &VectorMapping,
    size_of_k: usize,
    base_options: Algo::Options,
) -> (Algo::Decomposition, Algo::Decomposition, VectorMapping) {
    // Decompose dim
    // Need to use v columns of Dim later, also no anti-transpose or clearing since D^2 != 0
    let dim = build_dim(df, l_first_mapping);
    let dim_options = LoPhatOptions {
        maintain_v: true,
        clearing: false,
        ..base_options
    };
    let decompose_dim = Algo::init(Some(dim_options)).add_cols(dim).decompose();
    debug!("Decomposed im");

    // Decompose dker
    let dker = build_dker(&decompose_dim, l_first_mapping);
    let dker_options = LoPhatOptions {
        clearing: false,                // Not a chain complex so no clearing
        column_height: Some(size_of_k), // Non-square matrix
        ..base_options
    };
    let decompose_dker = Algo::init(Some(dker_options)).add_cols(dker).decompose();
    let kernel_mapping = build_kernel_mapping(&decompose_dim);
    debug!("Decomposed ker");
    (decompose_dim, decompose_dker, kernel_mapping)
}

pub fn decompose_relative<Algo: DecompositionAlgo<VecColumn, Options = LoPhatOptions>>(
    df: &[VecColumn],
    g_elements: &[bool],
    size_of_l: usize,
    size_of_k: usize,
    base_options: LoPhatOptions,
) -> (Algo::Decomposition, VectorMapping) {
    let (rel_mapping, l_index) = build_rel_mapping(df, g_elements, size_of_l, size_of_k);
    let drel = build_drel(df, g_elements, &rel_mapping, l_index).collect::<Vec<_>>();
    // Chain complex so can use clearing and anti-transpose
    let drel_at = anti_transpose(&drel);
    let decompose_drel = Algo::init(Some(base_options))
        .add_cols(drel_at.into_iter())
        .decompose();
    debug!("Decomposed rel");
    (decompose_drel, rel_mapping)
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

    let l_first_mapping = compute_l_first_mapping(&matrix);

    let (g_elements, df): (Vec<_>, Vec<_>) = matrix
        .into_iter()
        .map(|anncol| (anncol.in_g, anncol.col))
        .unzip();

    let size_of_l = g_elements.iter().filter(|in_g| **in_g).count();
    let size_of_k = df.len();

    let (f, (g, cok), (im, ker, kernel_mapping), (rel, rel_mapping)) = thread::scope(|s| {
        let thread1 = s.spawn(|| decompose_domain::<Algo>(&df, base_options));

        let thread2 = s.spawn(|| {
            decompose_cokernel_codomain::<Algo>(&df, &g_elements, &l_first_mapping, base_options)
        });

        let thread3 =
            s.spawn(|| decompose_kernel::<Algo>(&df, &l_first_mapping, size_of_k, base_options));

        let thread4 = s.spawn(|| {
            decompose_relative::<Algo>(&df, &g_elements, size_of_l, size_of_k, base_options)
        });

        (
            thread1.join().unwrap(),
            thread2.join().unwrap(),
            thread3.join().unwrap(),
            thread4.join().unwrap(),
        )
    });
    DecompositionEnsemble {
        f,
        g,
        im,
        ker,
        cok,
        rel,
        metadata: EnsembleMetadata {
            g_elements,
            l_first_mapping,
            kernel_mapping,
            rel_mapping,
            size_of_l,
            size_of_k,
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

    let f = decompose_domain::<Algo>(&df, base_options);
    let f = to_file(f);
    let (g, cok) =
        decompose_cokernel_codomain::<Algo>(&df, &g_elements, &l_first_mapping, base_options);
    let g = to_file(g);
    let cok = to_file(cok);
    let (im, ker, kernel_mapping) =
        decompose_kernel::<Algo>(&df, &l_first_mapping, size_of_k, base_options);
    let im = to_file(im);
    let ker = to_file(ker);
    let (rel, rel_mapping) =
        decompose_relative::<Algo>(&df, &g_elements, size_of_l, size_of_k, base_options);
    let rel = to_file(rel);

    FileEnsemble {
        f,
        g,
        im,
        ker,
        cok,
        rel,
        metadata: EnsembleMetadata {
            g_elements,
            l_first_mapping,
            kernel_mapping,
            rel_mapping,
            size_of_l,
            size_of_k,
        },
    }
}
