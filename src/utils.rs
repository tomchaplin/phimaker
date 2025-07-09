use lophat::{
    algorithms::{Decomposition, DecompositionAlgo},
    columns::{Column, VecColumn},
};

use std::fmt::Debug;

use crate::ensemble::DecompositionEnsemble;

pub fn print_matrix(matrix: &Vec<VecColumn>) {
    for col in matrix {
        println!("{:?}", col.entries());
    }
}

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

pub fn print_ensemble<C: Column + Debug, Algo: DecompositionAlgo<C>>(
    ensemble: &DecompositionEnsemble<C, Algo>,
) {
    println!("D_f:");
    print_decomp(&ensemble.f);
    println!("D_g:");
    print_decomp(&ensemble.g);
    println!("D_im:");
    print_decomp(&ensemble.im);
    println!("D_ker:");
    print_decomp(&ensemble.ker);
    println!("D_cok:");
    print_decomp(&ensemble.cok);
}
