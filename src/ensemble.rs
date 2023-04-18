use std::thread;

use lophat::{rv_decompose, LoPhatOptions, RVDecomposition, VecColumn};

use crate::{
    builders::{build_dcok, build_dg, build_dim, build_dker, build_drel},
    indexing::{
        build_kernel_mapping, build_rel_mapping, compute_l_first_mapping, AnnotatedColumn,
        VectorMapping,
    },
};

#[derive(Debug)]
pub struct DecompositionEnsemble {
    pub f: RVDecomposition<VecColumn>,
    pub g: RVDecomposition<VecColumn>,
    pub im: RVDecomposition<VecColumn>,
    pub ker: RVDecomposition<VecColumn>,
    pub cok: RVDecomposition<VecColumn>,
    pub rel: RVDecomposition<VecColumn>,
    pub l_first_mapping: VectorMapping,
    pub kernel_mapping: VectorMapping,
    pub rel_mapping: VectorMapping,
    pub g_elements: Vec<bool>,
    pub size_of_l: usize,
    pub size_of_k: usize,
}

fn run_decomposition(
    matrix: impl Iterator<Item = VecColumn>,
    new_column_height: Option<usize>,
    mut options: LoPhatOptions,
) -> RVDecomposition<VecColumn> {
    options.column_height = new_column_height;
    rv_decompose(matrix, &options)
}

pub fn all_decompositions(
    matrix: Vec<AnnotatedColumn<VecColumn>>,
    num_threads: usize,
) -> DecompositionEnsemble {
    let options = LoPhatOptions {
        maintain_v: true,
        column_height: None,
        num_threads,
        min_chunk_len: 1000,
        clearing: false,
    };
    // TODO: Clean this up so we aren't collecting the matrix again.
    let l_first_mapping = compute_l_first_mapping(&matrix);
    let g_elements: Vec<bool> = matrix.iter().map(|anncol| anncol.in_g).collect();
    let size_of_l = g_elements.iter().filter(|in_g| **in_g).count();
    let size_of_k = matrix.len();
    let df: Vec<VecColumn> = matrix.into_iter().map(|anncol| anncol.col).collect();
    let (f, (g, cok), (im, ker, kernel_mapping), (rel, rel_mapping)) = thread::scope(|s| {
        let thread1 = s.spawn(|| {
            // Decompose Df
            let out = run_decomposition(df.iter().cloned(), Some(size_of_k), options.clone());
            println!("Decomposed f");
            out
        });
        let thread2 = s.spawn(|| {
            // Decompose Dg
            let dg = build_dg(&df, &g_elements, &l_first_mapping);
            let decomp_dg = run_decomposition(dg, Some(size_of_l), options.clone());
            println!("Decomposed g");
            // Decompose dcok
            let dcok = build_dcok(&df, &decomp_dg, &g_elements, &l_first_mapping);
            let decompose_dcok = run_decomposition(dcok, Some(size_of_k), options.clone());
            println!("Decomposed cok");
            (decomp_dg, decompose_dcok)
        });
        let thread3 = s.spawn(|| {
            // Decompose dim
            let dim = build_dim(&df, &l_first_mapping);
            let decompose_dim = run_decomposition(dim, Some(size_of_k), options.clone());
            println!("Decomposed im");
            // Decompose dker
            let dker = build_dker(&decompose_dim, &l_first_mapping);
            let decompose_dker = run_decomposition(dker, Some(size_of_k), options.clone());
            println!("Decomposed ker");
            let kernel_mapping = build_kernel_mapping(&decompose_dim);
            (decompose_dim, decompose_dker, kernel_mapping)
        });
        let thread4 = s.spawn(|| {
            let (rel_mapping, l_index) = build_rel_mapping(&df, &g_elements, size_of_l, size_of_k);
            let drel = build_drel(&df, &g_elements, &rel_mapping, l_index);
            let decompose_drel =
                run_decomposition(drel, Some(size_of_k - size_of_l + 1), options.clone());
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
    }
}
