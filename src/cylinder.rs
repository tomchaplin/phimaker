//! Mapping-cylinder construction for filtered chain maps over F2.

use std::{cmp::Ordering, ops::Deref};

use itertools::Itertools;
use log::debug;
#[cfg(feature = "python-module")]
use pyo3::prelude::*;


#[derive(PartialEq, Eq, Clone, Copy, Debug)]
enum CylinderColType {
	Domain,
	Codomain,
	DomainShifted,
}

impl CylinderColType {
	fn type_int(&self) -> usize {
		match self {
			CylinderColType::Domain => 1,
			CylinderColType::Codomain => 2,
			CylinderColType::DomainShifted => 3,
		}
	}
}
impl PartialOrd for CylinderColType {
	fn partial_cmp(&self, other: &Self) -> Option<Ordering> { Some(self.cmp(other)) }
}

impl Ord for CylinderColType {
	fn cmp(&self, other: &Self) -> Ordering { self.type_int().cmp(&other.type_int()) }
}

/// Coordinates, degrees, and entrance times for a mapping cylinder.
///
/// Use the index arrays to find the cylinder column corresponding to an
/// original input generator. Index [`Self::times`] and [`Self::degrees`] by
/// cylinder column, including the endpoints of diagrams returned by
/// [`crate::sixpack`].
///
/// # Example
///
/// ```
/// use phimaker::sixpack;
///
/// let point = vec![(2.0, 0, vec![])];
/// let (_, metadata) = sixpack(&point, &point, &[vec![0]], 1, false)?;
/// let index = metadata.domain_indices[0];
/// assert_eq!(metadata.times[index], 2.0);
/// assert_eq!(metadata.degrees[index], 0);
/// # Ok::<(), phimaker::errors::InclusionError>(())
/// ```
#[cfg_attr(feature = "python-module", pyclass(module = "phimaker", get_all))]
pub struct CylinderMetadata {
	/// Entrance time of each cylinder column.
	pub times: Vec<f64>,
	/// Original domain generator index to cylinder column index.
	pub domain_indices: Vec<usize>,
	/// Original codomain generator index to cylinder column index.
	pub codomain_indices: Vec<usize>,
	/// Original domain generator index to its degree-one-higher cylinder copy.
	pub domain_shift: Vec<usize>,
	/// Degree of each cylinder generator.
	pub degrees: Vec<usize>,
}

/// Build the filtered mapping cylinder of a chain map A to B over F2.
///
/// Input columns are (entrance time, degree, sorted boundary indices). Each
/// matrix must have nondecreasing non-NaN times and be strictly upper
/// triangular, with boundary squared zero and degree lowered by one.
/// `chain_morphism[i]` is a sorted, duplicate-free list of codomain indices for
/// the image of generator i. There must be at least one map column per domain
/// column; extras are ignored. The map must preserve degree, commute with
/// boundary, and not increase time.
///
/// Returns a boundary matrix with 2 * len(A) + len(B) columns and its metadata.
/// The shifted copy of a has boundary a + f(a) + shift(boundary(a)). Equal-time
/// columns are ordered domain, codomain, shifted domain, preserving input
/// order. Use metadata to translate indices; the original domain embeds as a
/// subcomplex.
///
/// # Panics
/// Panics for unavailable boundary/image indices or missing map columns.
/// Algebraic validity, sortedness, and non-NaN times are caller assumptions and
/// are not comprehensively checked.
pub(crate) fn build_cylinder<UsizeSlice: Deref<Target = [usize]>>(
	domain_matrix: &[(
		f64,        // entrance time
		usize,      // degree
		UsizeSlice, // boundary
	)],
	codomain_matrix: &[(f64, usize, UsizeSlice)], // same structure as domain
	chain_morphism: &[UsizeSlice],
) -> (
	Vec<Vec<usize>>, // Boundary matrix of mapping cylinder
	CylinderMetadata,
) {
	// Let f: A -> B be a morphism of chain complexes over F_2 vector spaces.
	// The mapping cylinder cyl(f) is defined as follows:
	// - cyl(f)_n = A_n + B_n + A_{n-1}
	// - Domain and codomain copies retain their boundaries. A shifted generator
	//   has boundary a + f(a) + shift(boundary(a)).
	let domain_size = domain_matrix.len();
	let codomain_size = codomain_matrix.len();
	let cylinder_size = 2 * domain_size + codomain_size;
	let mut domain_idxs: Vec<usize> = Vec::with_capacity(domain_size);
	let mut codomain_idxs: Vec<usize> = Vec::with_capacity(codomain_size);
	let mut domain_shift_idxs: Vec<usize> = Vec::with_capacity(domain_size);
	let mut cylinder_matrix: Vec<Vec<usize>> = Vec::with_capacity(cylinder_size);
	let mut degrees = Vec::<usize>::with_capacity(cylinder_size);
	let mut times: Vec<f64> = Vec::with_capacity(cylinder_size);

	// Compare times then compare cell type
	let cell_ordering = |x: &(
		usize,           // original index
		f64,             // entrance time
		usize,           // degree
		CylinderColType, // column type
		&UsizeSlice,     // boundary
	),
	                     y: &(
		usize,           // original index
		f64,             // entrance time
		usize,           // degree
		CylinderColType, // column type
		&UsizeSlice,     // boundary
	)| { (x.1, x.3) <= (y.1, y.3) };

	let domain_iter = domain_matrix
		.iter()
		.enumerate()
		.map(|(idx, (time, degree, col))| {
			// Generators in the domain
			(idx, *time, *degree, CylinderColType::Domain, col)
		});

	let domain_shift_iter = domain_matrix
		.iter()
		.enumerate()
		.map(|(idx, (time, degree, col))| {
			// Shifted domain complex
			(idx, *time, *degree + 1, CylinderColType::DomainShifted, col)
		});

	let codomain_iter = codomain_matrix
		.iter()
		.enumerate()
		.map(|(idx, (time, degree, col))| {
			// Codomain columns
			(idx, *time, *degree, CylinderColType::Codomain, col)
		});

	let cylinder_iter = domain_iter
		.merge_by(codomain_iter, cell_ordering)
		.merge_by(domain_shift_iter, cell_ordering);

	for (cylinder_idx, (original_idx, time, degree, col_cell_type, col)) in
		cylinder_iter.enumerate()
	{
		// Build column
		let cylinder_col = match col_cell_type {
			CylinderColType::Domain => {
				// Take normal boundary but translate to new idxs
				col.iter()
					.map(|&row_idx| {
						domain_idxs
							.get(row_idx)
							.expect("Domain matrix should be strict upper triangular")
					})
					.copied()
					.collect_vec()
			},
			CylinderColType::Codomain => {
				// Take normal boundary but translate to new idxs
				col.iter()
					.map(|&row_idx| {
						codomain_idxs
							.get(row_idx)
							.expect("Codomain matrix should be strict upper triangular")
					})
					.copied()
					.collect_vec()
			},
			CylinderColType::DomainShifted => {
				// Identity going down a degree, into the domain.
				let domain_part = vec![
					domain_idxs
						.get(original_idx)
						.expect("Map should have one column per column of domain matrix"),
				]
				.into_iter()
				.copied();
				// The mapping going down a degree
				let codomain_part = chain_morphism
					.get(original_idx)
					.unwrap() // Original_idx is already an index into map
					.iter()
					.map(|&row_idx| {
						codomain_idxs.get(row_idx).expect(
							"Map must be compatible with both filtrations i.e. entrance time of \
							 f(c) <= entrance time of c",
						)
					})
					.copied();
				// The actual boundary going down a degree,
				// into the shifted domain part.
				let domain_shift_part = col
					.iter()
					.map(|&row_idx| {
						domain_shift_idxs
							.get(row_idx)
							.expect("Domain matrix should be strict upper triangular")
					})
					.copied();
				domain_part
					.chain(codomain_part)
					.chain(domain_shift_part)
					.sorted() // Different parts might be interleaved; need to sort idxs
					.collect_vec()
			},
		};
		cylinder_matrix.push(cylinder_col);
		degrees.push(degree);
		times.push(time);
		// Add to appropriate indexing vector
		match col_cell_type {
			CylinderColType::Domain => {
				domain_idxs.push(cylinder_idx);
			},
			CylinderColType::Codomain => {
				codomain_idxs.push(cylinder_idx);
			},
			CylinderColType::DomainShifted => {
				domain_shift_idxs.push(cylinder_idx);
			},
		}
	}
	let metadata = CylinderMetadata {
		times,
		domain_indices: domain_idxs,
		codomain_indices: codomain_idxs,
		domain_shift: domain_shift_idxs,
		degrees,
	};
	debug!(
		"Built mapping cylinder with {} simplices.",
		cylinder_matrix.len()
	);
	(cylinder_matrix, metadata)
}

#[cfg(test)]
mod tests {
	use lophat::{algorithms::LockFreeAlgorithm, columns::VecColumn};

	use super::*;
	use crate::all_decompositions;

	#[test]
	/// A square filled at times one and three has exactly one H1 kernel bar.
	fn square_map_has_one_finite_kernel_bar() {
		let domain_matrix = vec![
			(0.0, 0, vec![]),
			(0.0, 0, vec![]),
			(0.0, 0, vec![]),
			(0.0, 0, vec![]),
			(0.0, 1, vec![0, 1]),
			(0.0, 1, vec![1, 3]),
			(0.0, 1, vec![0, 2]),
			(0.0, 1, vec![2, 3]),
			(3.0, 2, vec![4, 5, 6, 7]),
			(4.0, 1, vec![0, 3]),
			(4.0, 2, vec![4, 5, 9]),
		];
		let codomain_matrix = vec![
			(0.0, 0, vec![]),
			(0.0, 0, vec![]),
			(0.0, 0, vec![]),
			(0.0, 0, vec![]),
			(0.0, 1, vec![0, 1]),
			(0.0, 1, vec![1, 3]),
			(0.0, 1, vec![0, 2]),
			(0.0, 1, vec![2, 3]),
			(0.4, 1, vec![0, 3]),
			(0.4, 2, vec![4, 5, 8]),
			(1.0, 2, vec![6, 7, 8]),
		];
		let chain_morphism = vec![
			vec![0],
			vec![1],
			vec![2],
			vec![3],
			vec![4],
			vec![5],
			vec![6],
			vec![7],
			vec![9, 10], // Long square gets mapped to sum of directed triangles
			vec![8],
			vec![9],
		];
		let (cyl_matrix, metadata) =
			build_cylinder(&domain_matrix, &codomain_matrix, &chain_morphism);
		let ensemble = all_decompositions::<LockFreeAlgorithm<VecColumn>, _>(
			&cyl_matrix,
			&metadata.degrees,
			&metadata.domain_indices,
			0,
		)
		.unwrap()
		.all_diagrams();
		let pairings: Vec<_> = ensemble
			.kernel
			.iter()
			.filter_map(|(&birth, &death)| {
				match death {
					crate::diagrams::ExtendedUsize::Finite(death) => Some((birth, death)),
					crate::diagrams::ExtendedUsize::Infinity => None,
				}
			})
			.collect();
		assert_eq!(
			ensemble.kernel.len(),
			1,
			"No additional essential kernel bars"
		);
		assert_eq!(pairings.len(), 1);
		let first_pairing = pairings[0];
		let dgm_pt = (
			metadata.times[first_pairing.0],
			metadata.times[first_pairing.1],
		);
		assert_eq!(dgm_pt, (1.0, 3.0))
	}
}
