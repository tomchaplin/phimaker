use std::{marker::PhantomData, thread};

use lophat::{
    algorithms::RVDecomposition,
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
pub struct DecompositionEnsemble<C, Algo>
where
    C: Column,
    Algo: RVDecomposition<C>,
{
    pub f: Algo,
    pub g: Algo,
    pub im: Algo,
    pub ker: Algo,
    pub cok: Algo,
    pub rel: Algo,
    pub l_first_mapping: VectorMapping,
    pub kernel_mapping: VectorMapping,
    pub rel_mapping: VectorMapping,
    pub g_elements: Vec<bool>,
    pub size_of_l: usize,
    pub size_of_k: usize,
    phantom: PhantomData<C>,
}

pub fn all_decompositions<Algo: RVDecomposition<VecColumn, Options = LoPhatOptions> + Send>(
    matrix: Vec<AnnotatedColumn<VecColumn>>,
    num_threads: usize,
) -> DecompositionEnsemble<VecColumn, Algo> {
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
        let thread1 = s.spawn(|| {
            // Decompose Df
            // Df is a chain complex so can compute anti-transpose instead
            let df_at = anti_transpose(&df);
            let out = Algo::decompose(df_at.into_iter(), Some(base_options));
            println!("Decomposed f");
            out
        });

        let thread2 = s.spawn(|| {
            // Decompose Dg
            // Need to use v columns of Dg later, so no AT
            let dg = build_dg(&df, &g_elements, &l_first_mapping);
            let mut dg_options = base_options.clone();
            dg_options.maintain_v = true;
            let decomp_dg = Algo::decompose(dg, Some(dg_options));
            println!("Decomposed g");
            // Decompose dcok
            let dcok = build_dcok(&df, &decomp_dg, &g_elements, &l_first_mapping);
            let mut dcok_options = base_options.clone();
            dcok_options.clearing = false; // Not a chain complex
            let decompose_dcok = Algo::decompose(dcok, Some(dcok_options));
            println!("Decomposed cok");
            (decomp_dg, decompose_dcok)
        });

        let thread3 = s.spawn(|| {
            // Decompose dim
            // Need to use v columns of Dim later, also no AT or clearing since D^2 != 0
            let dim = build_dim(&df, &l_first_mapping);
            let mut dim_options = base_options.clone();
            dim_options.maintain_v = true;
            dim_options.clearing = false;
            let decompose_dim = Algo::decompose(dim, Some(dim_options));
            println!("Decomposed im");
            // Decompose dker
            let dker = build_dker(&decompose_dim, &l_first_mapping);
            let mut dker_options = base_options.clone();
            dker_options.clearing = false; // Not a chain complex so no clearing
            dker_options.column_height = Some(size_of_k); // Non-square matrix
            let decompose_dker = Algo::decompose(dker, Some(dker_options));
            println!("Decomposed ker");
            let kernel_mapping = build_kernel_mapping(&decompose_dim);
            (decompose_dim, decompose_dker, kernel_mapping)
        });

        let thread4 = s.spawn(|| {
            let (rel_mapping, l_index) = build_rel_mapping(&df, &g_elements, size_of_l, size_of_k);
            let drel = build_drel(&df, &g_elements, &rel_mapping, l_index).collect();
            // Chain complex so can use clearing and AT
            let drel_at = anti_transpose(&drel);
            let decompose_drel = Algo::decompose(drel_at.into_iter(), Some(base_options));
            println!("Decomposed rel");
            (decompose_drel, rel_mapping)
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
        g_elements,
        l_first_mapping,
        kernel_mapping,
        rel_mapping,
        size_of_l,
        size_of_k,
        phantom: PhantomData,
    }
}
