use lophat::{RVDecomposition, VecColumn};

use crate::ensemble::DecompositionEnsemble;

pub fn print_matrix(matrix: &Vec<VecColumn>) {
    for col in matrix {
        println!("{:?}", col.internal);
    }
}

pub fn print_decomp(decomp: &RVDecomposition<VecColumn>) {
    println!("R:");
    print_matrix(&decomp.r);
    println!("V:");
    print_matrix(&decomp.v.as_ref().unwrap());
}

pub fn print_ensemble(ensemble: &DecompositionEnsemble) {
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
