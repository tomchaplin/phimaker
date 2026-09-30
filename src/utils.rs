//! Diagnostic printing of matrices and decompositions.

use lophat::{
    algorithms::{Decomposition, DecompositionAlgo},
    columns::{Column, VecColumn},
};

use std::fmt::Debug;

use crate::ensemble::DecompositionEnsemble;

/// Print nonzero row indices of each matrix column to standard output.
pub fn print_matrix(matrix: &Vec<VecColumn>) {
    for col in matrix {
        println!("{:?}", col.entries());
    }
}

/// Print R and, when retained, V columns to standard output.
/// The decomposition must be nonempty: column zero is inspected to detect V.
pub fn print_decomp<C: Column + Debug, Decomp: Decomposition<C>>(decomp: &Decomp) {
    println!("R:");
    let r_matrix = (0..decomp.n_cols()).map(|idx| decomp.get_r_col(idx));
    for col in r_matrix {
        println!("{:?}", *col);
    }
    if decomp.get_v_col(0).is_ok() {
        let v_matrix = (0..decomp.n_cols()).map(|idx| decomp.get_v_col(idx));
        println!("V:");
        for col in v_matrix {
            println!("{:?}", *col.unwrap());
        }
    }
}

/// Print codomain, domain, image, kernel, and cokernel reductions.
/// Relative is omitted. Each printed decomposition must be nonempty;
/// see [`print_decomp`]. Labels D_f and D_g denote codomain and domain.
pub fn print_ensemble<C: Column + Debug, Algo: DecompositionAlgo<C>>(
    ensemble: &DecompositionEnsemble<C, Algo>,
) {
    println!("D_f:");
    print_decomp(&ensemble.d_cod);
    println!("D_g:");
    print_decomp(&ensemble.d_dom);
    println!("D_im:");
    print_decomp(&ensemble.d_im);
    println!("D_ker:");
    print_decomp(&ensemble.d_ker);
    println!("D_cok:");
    print_decomp(&ensemble.d_cok);
}
