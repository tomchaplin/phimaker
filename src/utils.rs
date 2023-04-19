use lophat::{
    algorithms::RVDecomposition,
    columns::{Column, VecColumn},
};

use std::fmt::Debug;

use crate::ensemble::DecompositionEnsemble;

pub fn print_matrix(matrix: &Vec<VecColumn>) {
    for col in matrix {
        println!("{:?}", col.entries());
    }
}

pub fn print_decomp<C: Column + Debug, Algo: RVDecomposition<C>>(decomp: &Algo) {
    println!("R:");
    let r_matrix = (0..decomp.n_cols()).map(|idx| decomp.get_r_col(idx));
    for col in r_matrix {
        println!("{:?}", *col);
    }
    if decomp.get_v_col(0).is_some() {
        let v_matrix = (0..decomp.n_cols()).map(|idx| decomp.get_v_col(idx));
        println!("V:");
        for col in v_matrix {
            println!("{:?}", *col.unwrap());
        }
    }
}

pub fn print_ensemble<C: Column + Debug, Algo: RVDecomposition<C>>(
    ensemble: &DecompositionEnsemble<C, Algo>,
) {
    println!("Df:");
    print_decomp(&ensemble.f);
    println!("Dg:");
    print_decomp(&ensemble.g);
    println!("Dim:");
    print_decomp(&ensemble.im);
    println!("Dker:");
    print_decomp(&ensemble.ker);
    println!("Dcok:");
    print_decomp(&ensemble.cok);
}
