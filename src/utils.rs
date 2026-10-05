//! Diagnostic printing of matrices and decompositions.

use std::fmt::Debug;

use lophat::{
	algorithms::{Decomposition, DecompositionAlgo},
	columns::{Column, VecColumn},
};

use crate::ensemble::DecompositionEnsemble;

/// Print the nonzero row indices of each column to standard output.
///
/// # Example
///
/// ```
/// use lophat::columns::VecColumn;
/// use phimaker::utils::print_matrix;
///
/// let matrix = vec![VecColumn::from((0, vec![])), VecColumn::from((1, vec![0]))];
/// print_matrix(&matrix);
/// ```
pub fn print_matrix(matrix: &Vec<VecColumn>) {
	for col in matrix {
		println!("{:?}", col.entries());
	}
}

/// Print the reduced R columns and any available V columns to standard output.
///
/// # Example
///
/// ```
/// use lophat::{
/// 	algorithms::{DecompositionAlgo, SerialAlgorithm},
/// 	columns::VecColumn,
/// };
/// use phimaker::utils::print_decomp;
///
/// let decomposition = SerialAlgorithm::<VecColumn>::init(None)
/// 	.add_cols([VecColumn::from((0, vec![]))].into_iter())
/// 	.decompose();
/// print_decomp(&decomposition);
/// ```
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

/// Print codomain, domain, image, kernel, and cokernel decompositions.
///
/// Relative is omitted. Output labels `D_f` and `D_g` denote codomain and
/// domain. Use [`print_decomp`] to print an individual decomposition.
///
/// # Example
///
/// ```
/// use lophat::algorithms::LockFreeAlgorithm;
/// use phimaker::{all_decompositions, utils::print_ensemble};
///
/// let ensemble = all_decompositions::<LockFreeAlgorithm<_>, _>(&[vec![]], &[0], &[0], 1)?;
/// print_ensemble(&ensemble);
/// # Ok::<(), phimaker::errors::InclusionError>(())
/// ```
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
