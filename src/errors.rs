//! Structural input errors shared by the Rust inclusion APIs.

use std::{error::Error, fmt};

/// Invalid structure in an inclusion of filtered chain complexes over F2.
///
/// Match a variant to identify an invalid degree count, generator index,
/// boundary entry, or domain selection. [`std::fmt::Display`] provides a
/// readable message. The boundary-squared-zero requirement is not checked by
/// this error type.
///
/// # Example
///
/// ```
/// use phimaker::{errors::InclusionError, sixpack_from_inclusion};
///
/// let error = sixpack_from_inclusion(&[vec![]], &[], &[], 1, false).unwrap_err();
/// assert_eq!(
/// 	error,
/// 	InclusionError::DegreeCount {
/// 		expected: 1,
/// 		actual: 0
/// 	}
/// );
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InclusionError {
	/// The degree list does not have one entry per boundary column.
	DegreeCount {
		/// Number of boundary columns.
		expected: usize,
		/// Number of supplied degrees.
		actual: usize,
	},
	/// A domain generator index is outside the codomain index range.
	DomainIndexOutOfBounds {
		/// Invalid original generator index.
		index: usize,
		/// Number of codomain columns; valid indices are strictly less than
		/// this.
		columns: usize,
	},
	/// A boundary row index is not strictly less than its column index.
	NonUpperTriangular {
		/// Original boundary column index.
		column: usize,
		/// Offending boundary row index.
		row: usize,
	},
	/// A boundary entry does not lower degree by exactly one.
	InvalidBoundaryDegree {
		/// Original generator index.
		column: usize,
		/// Offending boundary generator index.
		row: usize,
		/// Degree of the original generator.
		degree: usize,
		/// Degree of the boundary generator.
		boundary_degree: usize,
	},
	/// A domain generator has a boundary generator outside the domain.
	DomainNotClosed {
		/// Original index of the domain generator.
		column: usize,
		/// Original index of the boundary generator outside the domain.
		row: usize,
	},
}

impl fmt::Display for InclusionError {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		match self {
			Self::DegreeCount { expected, actual } => {
				write!(
					f,
					"`degrees` has {actual} elements, but `boundary_matrix` has {expected} columns"
				)
			},
			Self::DomainIndexOutOfBounds { index, columns } => {
				write!(
					f,
					"Boundary matrix has {columns} columns but found domain column index {index} \
					 in argument `cols_in_domain`"
				)
			},
			Self::NonUpperTriangular { column, row } => {
				write!(
					f,
					"Boundary matrix must be strictly upper-triangular, but column {column} \
					 contains entry at index {row}"
				)
			},
			Self::InvalidBoundaryDegree {
				column,
				row,
				degree,
				boundary_degree,
			} => {
				write!(
					f,
					"Boundaries must lower degree by exactly 1; column {column} has degree \
					 {degree}, but boundary generator {row} has degree {boundary_degree}"
				)
			},
			Self::DomainNotClosed { column, row } => {
				write!(
					f,
					"Domain must be closed under taking boundaries; found generator {row} in \
					 boundary column {column}"
				)
			},
		}
	}
}

impl Error for InclusionError {}
