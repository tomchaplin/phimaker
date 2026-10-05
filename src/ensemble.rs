//! Reduction of the six matrices, in memory or using temporary files.

use std::{collections::HashMap, fs::File, io::BufWriter, marker::PhantomData, ops::Deref, thread};

use bincode::serialize_into;
use log::debug;
use lophat::{
	algorithms::DecompositionAlgo,
	columns::{Column, VecColumn},
	options::LoPhatOptions,
	utils::anti_transpose,
};
use serde::Serialize;

use crate::{
	builders::{
		build_d_cok,
		build_d_dom,
		build_d_im,
		build_d_ker,
		build_d_rel,
		compute_dom_first_permutation,
		decomp_cycle_idxs,
	},
	errors::InclusionError,
	indexing::{DensePermutation, Permutation},
};

#[derive(Debug, Clone, Default)]
/// Generator indices and coordinate maps for an inclusion result.
///
/// `cols_in_dom` contains sorted, unique original domain indices. Use
/// `dom_first_permutation` to translate original indices to coordinates with
/// domain generators first. Use `kernel_mapping` to locate kernel columns
/// corresponding to image cycle columns. For diagrams in original indices, call
/// `all_diagrams` on the containing ensemble.
///
/// # Example
///
/// ```
/// use lophat::algorithms::LockFreeAlgorithm;
/// use phimaker::{all_decompositions, indexing::Permutation};
///
/// let ensemble = all_decompositions::<LockFreeAlgorithm<_>, _>(
/// 	&[vec![], vec![], vec![0, 1]],
/// 	&[0, 0, 1],
/// 	&[1],
/// 	1,
/// )?;
/// assert_eq!(ensemble.metadata.dom_first_permutation.map(1), 0);
/// # Ok::<(), phimaker::errors::InclusionError>(())
/// ```
pub struct EnsembleMetadata {
	/// Original indices to domain-first indices, preserving order within each
	/// block.
	pub dom_first_permutation: DensePermutation,
	/// Original image cycle-column index to compact kernel column index.
	pub kernel_mapping: HashMap<usize, usize>,
	/// Sorted original indices of domain generators.
	pub cols_in_dom: Vec<usize>,
	/// Number of domain generators.
	pub sz_domain: usize,
	/// Number of codomain generators.
	pub sz_codomain: usize,
}

#[derive(Debug)]
/// Six matrix decompositions and their coordinate metadata.
///
/// Obtain this value from [`all_decompositions`]. Use [`Self::all_diagrams`]
/// for diagrams in original input indices. When inspecting matrix columns
/// directly, follow the coordinate conventions documented on each field and use
/// `metadata` to translate indices. Repeated diagram extraction is supported.
///
/// # Example
///
/// ```
/// use lophat::algorithms::LockFreeAlgorithm;
/// use phimaker::all_decompositions;
///
/// let ensemble = all_decompositions::<LockFreeAlgorithm<_>, _>(&[vec![]], &[0], &[0], 1)?;
/// assert_eq!(ensemble.all_diagrams().domain.len(), 1);
/// # Ok::<(), phimaker::errors::InclusionError>(())
/// ```
pub struct DecompositionEnsemble<C, Algo>
where
	C: Column,
	Algo: DecompositionAlgo<C>,
{
	/// Anti-transposed codomain decomposition.
	pub d_cod: Algo::Decomposition,
	/// Domain decomposition in local domain coordinates, retaining V.
	pub d_dom: Algo::Decomposition,
	/// Image decomposition: original columns, domain-first rows, retaining V.
	pub d_im: Algo::Decomposition,
	/// Kernel decomposition: compact cycle columns and domain-first rows.
	pub d_ker: Algo::Decomposition,
	/// Cokernel decomposition in original codomain coordinates.
	pub d_cok: Algo::Decomposition,
	/// Anti-transposed relative decomposition in local quotient coordinates.
	pub d_rel: Algo::Decomposition,
	/// Coordinate maps for this inclusion.
	pub metadata: EnsembleMetadata,
	phantom: PhantomData<C>,
}

#[derive(Debug)]
/// Six stored decompositions and their coordinate metadata.
///
/// Obtain this value from [`all_decompositions_slow`]. Use
/// [`Self::all_diagrams`] to read the results and obtain diagrams in original
/// indices. The file handles start at byte zero. Reading advances their
/// cursors; rewind all six before extracting again. Data remains available for
/// the lifetime of the file handles.
///
/// # Example
///
/// ```
/// use lophat::algorithms::LockFreeAlgorithm;
/// use phimaker::all_decompositions_slow;
///
/// let stored = all_decompositions_slow::<LockFreeAlgorithm<_>, _>(&[vec![]], &[0], &[0], 1)?;
/// assert_eq!(stored.all_diagrams().domain.len(), 1);
/// # Ok::<(), phimaker::errors::InclusionError>(())
/// ```
pub struct FileEnsemble {
	/// Anti-transposed codomain decomposition.
	pub d_cod: File,
	/// Domain decomposition in local domain coordinates, retaining V.
	pub d_dom: File,
	/// Image decomposition: original columns, domain-first rows, retaining V.
	pub d_im: File,
	/// Kernel decomposition: compact cycle columns and domain-first rows.
	pub d_ker: File,
	/// Cokernel decomposition in original codomain coordinates.
	pub d_cok: File,
	/// Anti-transposed relative decomposition in local quotient coordinates.
	pub d_rel: File,
	/// Coordinate maps for this inclusion.
	pub metadata: EnsembleMetadata,
}

/// Reduce the anti-transpose of a square codomain boundary matrix.
/// Returned indices are reversed dual coordinates; undo with
/// [`crate::diagrams::PersistenceDiagram::anti_transpose`]. Options pass
/// through.
fn decompose_cod<Algo: DecompositionAlgo<VecColumn, Options = LoPhatOptions>>(
	d_cod: &[VecColumn],
	base_options: Algo::Options,
) -> Algo::Decomposition {
	// Decompose D_cod
	// D_cod is a chain complex so can compute anti-transpose instead
	let d_cod_anti_transpose = anti_transpose(d_cod);
	let out = Algo::init(Some(base_options))
		.add_cols(d_cod_anti_transpose.into_iter())
		.decompose();
	debug!("Decomposed d_cod");
	out
}

/// Reduce domain and cokernel matrices, returning them in that order.
/// The domain indices must be sorted and consistent with the domain-first
/// permutation. Retains domain V and disables cokernel clearing; other options
/// come from `base_options`. See [`build_d_dom`] and [`build_d_cok`].
fn decompose_dom_cok<Algo: DecompositionAlgo<VecColumn, Options = LoPhatOptions>>(
	d_cod: &[VecColumn],
	cols_in_dom: &[usize], // WARNING: assumes that this is sorted
	dom_first_mapping: &impl Permutation,
	base_options: Algo::Options,
) -> (Algo::Decomposition, Algo::Decomposition) {
	// Decompose D_dom
	// Need to use v columns of D_dom later, so no anti-transpose
	let d_dom = build_d_dom(d_cod, cols_in_dom, dom_first_mapping);
	let d_dom_decomp_options = LoPhatOptions {
		maintain_v: true,
		..base_options
	};
	let d_dom_decomp = Algo::init(Some(d_dom_decomp_options))
		.add_cols(d_dom)
		.decompose();
	debug!("Decomposed d_dom");

	// Decompose d_cok
	let d_cok = build_d_cok(d_cod, &d_dom_decomp, dom_first_mapping);
	let d_cok_decomp_options = LoPhatOptions {
		clearing: false,
		..base_options
	};
	let decomp_d_cok = Algo::init(Some(d_cok_decomp_options))
		.add_cols(d_cok)
		.decompose();
	debug!("Decomposed d_cok");
	(d_dom_decomp, decomp_d_cok)
}
/// Reduce image and kernel matrices and return their column correspondence.
/// The third result maps original image cycle-column indices to compact kernel
/// column indices. `sz_cod` must equal the codomain size. Retains image V,
/// disables clearing for both matrices, and sets kernel height to `sz_cod`.
/// The permutation must match the inclusion; see [`build_d_im`] and
/// [`build_d_ker`].
fn decompose_im_ker<Algo: DecompositionAlgo<VecColumn, Options = LoPhatOptions>>(
	d_cod: &[VecColumn],
	dom_first_mapping: &impl Permutation,
	sz_cod: usize,
	base_options: Algo::Options,
) -> (
	Algo::Decomposition,   // Decomposition of D_im
	Algo::Decomposition,   // Decomposition of D_ker
	HashMap<usize, usize>, // cycle columns of D_im
) {
	// Decompose dim
	// Need to use v columns of Dim later, also no anti-transpose or clearing
	// since D^2 != 0
	let d_im = build_d_im(d_cod, dom_first_mapping);
	let d_im_decomp_options = LoPhatOptions {
		maintain_v: true,
		clearing: false,
		..base_options
	};
	let d_im_decomp = Algo::init(Some(d_im_decomp_options))
		.add_cols(d_im)
		.decompose();
	debug!("Decomposed d_im");

	// Decompose d_ker
	let d_ker = build_d_ker(&d_im_decomp, dom_first_mapping);
	let d_ker_options = LoPhatOptions {
		clearing: false,             // Not a chain complex so no clearing
		column_height: Some(sz_cod), // Non-square matrix
		..base_options
	};
	let d_ker_decomp = Algo::init(Some(d_ker_options)).add_cols(d_ker).decompose();
	let ker_mapping = decomp_cycle_idxs(&d_im_decomp)
		.enumerate()
		.map(|(idx_in_d_ker, idx_in_d_im)| (idx_in_d_im, idx_in_d_ker))
		.collect();
	debug!("Decomposed d_ker");
	(d_im_decomp, d_ker_decomp, ker_mapping)
}

/// Reduce the anti-transpose of the quotient boundary B/A.
/// The permutation must put the `sz_domain` domain generators first. Returned
/// indices are reversed local quotient indices, not original codomain indices.
/// See [`build_d_rel`]; reduction options pass through.
fn decompose_rel<Algo: DecompositionAlgo<VecColumn, Options = LoPhatOptions>>(
	d_cod: &[VecColumn],
	dom_first_mapping: &impl Permutation,
	sz_domain: usize,
	base_options: LoPhatOptions,
) -> Algo::Decomposition {
	let d_rel = build_d_rel(d_cod, dom_first_mapping, sz_domain).collect::<Vec<_>>();
	// Chain complex so can use clearing and anti-transpose
	let d_rel_anti_transpose = anti_transpose(&d_rel);
	let decomp_d_rel = Algo::init(Some(base_options))
		.add_cols(d_rel_anti_transpose.into_iter())
		.decompose();
	debug!("Decomposed d_rel");
	decomp_d_rel
}

struct PreparedInclusion {
	d_cod: Vec<VecColumn>,
	cols_in_dom: Vec<usize>,
	dom_first_permutation: DensePermutation,
}

fn prepare_inclusion<UsizeSlice: Deref<Target = [usize]>>(
	boundary_matrix: &[UsizeSlice],
	degrees: &[usize],
	cols_in_dom: &[usize],
) -> Result<PreparedInclusion, InclusionError> {
	let sz_codomain = boundary_matrix.len();
	if degrees.len() != sz_codomain {
		return Err(InclusionError::DegreeCount {
			expected: sz_codomain,
			actual: degrees.len(),
		});
	}
	let mut cols_in_dom = cols_in_dom.to_vec();
	if !cols_in_dom.is_sorted() {
		cols_in_dom.sort_unstable();
	}
	cols_in_dom.dedup();
	if let Some(&index) = cols_in_dom.last()
		&& index >= sz_codomain
	{
		return Err(InclusionError::DomainIndexOutOfBounds {
			index,
			columns: sz_codomain,
		});
	}
	let mut in_domain = vec![false; sz_codomain];
	for &index in &cols_in_dom {
		in_domain[index] = true;
	}
	let mut d_cod = Vec::with_capacity(sz_codomain);
	for (column, support) in boundary_matrix.iter().enumerate() {
		let mut entries = support.to_vec();
		if !entries.is_sorted() {
			entries.sort_unstable();
		}
		entries.dedup();
		if let Some(&row) = entries.last()
			&& row >= column
		{
			return Err(InclusionError::NonUpperTriangular { column, row });
		}
		for &row in &entries {
			if degrees[column].checked_sub(1) != Some(degrees[row]) {
				return Err(InclusionError::InvalidBoundaryDegree {
					column,
					row,
					degree: degrees[column],
					boundary_degree: degrees[row],
				});
			}
			if in_domain[column] && !in_domain[row] {
				return Err(InclusionError::DomainNotClosed { column, row });
			}
		}
		d_cod.push(VecColumn::from((degrees[column], entries)));
	}
	let dom_first_permutation = compute_dom_first_permutation(sz_codomain, &cols_in_dom);
	Ok(PreparedInclusion {
		d_cod,
		cols_in_dom,
		dom_first_permutation,
	})
}

/// Compute six matrix decompositions for an inclusion A into B over F2.
///
/// Use this function when you need the reduced matrices as well as the
/// diagrams. Call [`DecompositionEnsemble::all_diagrams`] on the result for all
/// six diagrams in the original indices. For diagrams alone, use
/// [`crate::sixpack_from_inclusion`].
///
/// `boundary_matrix[j]` lists the nonzero rows of column j of B. `degrees`
/// gives one degree per column, and `cols_in_dom` selects the domain
/// generators. Row lists and domain indices may be in any order. Repeated
/// listings identify the same entry and are ignored, not summed modulo two.
/// Inputs are not modified. Column indices specify filtration order; no
/// filtration values are accepted.
///
/// `num_threads` limits workers per reduction; the whole call may use more
/// threads. Zero selects the count automatically. Results remain in memory. Use
/// [`all_decompositions_slow`] to store intermediate results in temporary
/// files.
///
/// # Errors
/// Returns [`InclusionError`] for mismatched degree counts, out-of-range domain
/// indices, boundary rows not less than their column, boundaries that fail to
/// lower degree by one, or a domain not closed under boundary.
///
/// # Assumptions and panics
/// The boundary must square to zero. This is not checked; violations may
/// produce incorrect results or panic. Reduction failures can also panic.
///
/// # Example
///
/// ```
/// use lophat::algorithms::LockFreeAlgorithm;
/// use phimaker::all_decompositions;
///
/// let ensemble = all_decompositions::<LockFreeAlgorithm<_>, _>(
/// 	&[vec![], vec![], vec![0, 1]],
/// 	&[0, 0, 1],
/// 	&[1],
/// 	1,
/// )?;
/// let diagrams = ensemble.all_diagrams();
/// assert!(diagrams.kernel.is_empty());
/// assert_eq!(ensemble.metadata.cols_in_dom, vec![1]);
/// # Ok::<(), phimaker::errors::InclusionError>(())
/// ```
pub fn all_decompositions<
	Algo: DecompositionAlgo<VecColumn, Options = LoPhatOptions>,
	UsizeSlice: Deref<Target = [usize]>,
>(
	boundary_matrix: &[UsizeSlice],
	degrees: &[usize],
	cols_in_dom: &[usize],
	num_threads: usize,
) -> Result<DecompositionEnsemble<VecColumn, Algo>, InclusionError>
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

	let PreparedInclusion {
		d_cod,
		cols_in_dom,
		dom_first_permutation,
	} = prepare_inclusion(boundary_matrix, degrees, cols_in_dom)?;
	let sz_codomain = d_cod.len();
	let sz_domain = cols_in_dom.len();

	let (cod_decomp, (dom_decomp, cok_decomp), (im_decomp, ker_decomp, kernel_mapping), rel_decomp) =
		thread::scope(|s| {
			let thread1 = s.spawn(|| decompose_cod::<Algo>(&d_cod, base_options));

			let thread2 = s.spawn(|| {
				decompose_dom_cok::<Algo>(
					&d_cod,
					&cols_in_dom,
					&dom_first_permutation,
					base_options,
				)
			});

			let thread3 = s.spawn(|| {
				decompose_im_ker::<Algo>(&d_cod, &dom_first_permutation, sz_codomain, base_options)
			});

			let thread4 = s.spawn(|| {
				decompose_rel::<Algo>(&d_cod, &dom_first_permutation, sz_domain, base_options)
			});

			(
				thread1.join().unwrap(),
				thread2.join().unwrap(),
				thread3.join().unwrap(),
				thread4.join().unwrap(),
			)
		});
	Ok(DecompositionEnsemble {
		d_cod: cod_decomp,
		d_dom: dom_decomp,
		d_im: im_decomp,
		d_ker: ker_decomp,
		d_cok: cok_decomp,
		d_rel: rel_decomp,
		metadata: EnsembleMetadata {
			cols_in_dom,
			dom_first_permutation,
			kernel_mapping,
			sz_domain,
			sz_codomain,
		},
		phantom: PhantomData,
	})
}

/// Serialize a value with bincode and return a temporary-file read handle.
///
/// The value is consumed. The handle starts at byte zero and keeps its data
/// alive until it is closed. Read with a matching bincode type; for
/// decompositions use [`crate::diagrams::from_file`]. Rewind before reading a
/// second time.
///
/// # Panics
/// Panics if temporary-file creation, reopening, or serialization fails.
///
/// # Example
///
/// ```
/// use phimaker::ensemble::to_file;
///
/// let file = to_file(vec![1usize, 2]);
/// let restored: Vec<usize> = bincode::deserialize_from(&file)?;
/// assert_eq!(restored, vec![1, 2]);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn to_file<Algo: Serialize>(algo: Algo) -> File {
	let mut file_write = tempfile::NamedTempFile::new().expect("Can't get temp file");
	println!("Writing to {:?}", file_write.path());
	// We reopen so that we can hold onto the file for later reading
	let file_read = file_write.reopen().expect("Can't reopen tempfile");
	{
		let mut buf = BufWriter::new(&mut file_write);
		serialize_into(&mut buf, &algo).expect("Can't serialize to file");
	}
	// Explicitly release memory
	drop(algo);
	file_read
}

/// Compute six decompositions for an inclusion, storing results in temporary
/// files.
///
/// Inputs, index conventions, error conditions, and the unchecked requirement
/// that the boundary square to zero are the same as [`all_decompositions`].
/// Empty inputs, domains, and quotients are supported. `num_threads` limits
/// workers per reduction; zero selects the count automatically.
///
/// Use this variant to reduce memory use between computation steps. It may be
/// slower, and final diagram extraction still requires memory for all six
/// results. Call [`FileEnsemble::all_diagrams`] to obtain diagrams in original
/// input indices. That call advances the file cursors; rewind each file before
/// extracting again.
///
/// # Errors
/// Returns [`InclusionError`] for structurally invalid inputs, as described in
/// [`all_decompositions`].
///
/// # Panics
/// Reduction, temporary-file, and serialization failures may panic. A boundary
/// that does not square to zero may produce incorrect results or panic.
///
/// # Example
///
/// ```
/// use lophat::algorithms::LockFreeAlgorithm;
/// use phimaker::all_decompositions_slow;
///
/// let ensemble = all_decompositions_slow::<LockFreeAlgorithm<_>, _>(&[vec![]], &[0], &[0], 1)?;
/// let diagrams = ensemble.all_diagrams();
/// assert_eq!(diagrams.domain.len(), 1);
/// assert!(diagrams.relative.is_empty());
/// # Ok::<(), phimaker::errors::InclusionError>(())
/// ```
pub fn all_decompositions_slow<
	Algo: DecompositionAlgo<VecColumn, Options = LoPhatOptions>,
	UsizeSlice: Deref<Target = [usize]>,
>(
	boundary_matrix: &[UsizeSlice],
	degrees: &[usize],
	cols_in_dom: &[usize],
	num_threads: usize,
) -> Result<FileEnsemble, InclusionError>
where
	Algo::Decomposition: Serialize + Send,
{
	let base_options = LoPhatOptions {
		maintain_v: false,   // Only turn on maintain_v on threads where we need it
		column_height: None, // Assume square unless told otherwise
		num_threads,
		min_chunk_len: 10000,
		clearing: true, // Clear whenever we can
	};

	let PreparedInclusion {
		d_cod,
		cols_in_dom,
		dom_first_permutation,
	} = prepare_inclusion(boundary_matrix, degrees, cols_in_dom)?;
	let sz_codomain = d_cod.len();
	let sz_domain = cols_in_dom.len();

	let cod_decomp = decompose_cod::<Algo>(&d_cod, base_options);
	let cod = to_file(cod_decomp);
	let (dom_decomp, cok_decomp) =
		decompose_dom_cok::<Algo>(&d_cod, &cols_in_dom, &dom_first_permutation, base_options);
	let dom = to_file(dom_decomp);
	let cok = to_file(cok_decomp);
	let (im_decomp, ker_decomp, kernel_mapping) =
		decompose_im_ker::<Algo>(&d_cod, &dom_first_permutation, sz_codomain, base_options);
	let im = to_file(im_decomp);
	let ker = to_file(ker_decomp);
	let rel_decomp = decompose_rel::<Algo>(&d_cod, &dom_first_permutation, sz_domain, base_options);
	let rel = to_file(rel_decomp);

	Ok(FileEnsemble {
		d_cod: cod,
		d_dom: dom,
		d_im: im,
		d_ker: ker,
		d_cok: cok,
		d_rel: rel,
		metadata: EnsembleMetadata {
			cols_in_dom,
			dom_first_permutation,
			kernel_mapping,
			sz_domain,
			sz_codomain,
		},
	})
}
