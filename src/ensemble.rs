//! Reduction of the six matrices, in memory or using temporary files.

use bincode::serialize_into;
use itertools::Itertools;
use log::debug;
use lophat::{
    algorithms::DecompositionAlgo,
    columns::{Column, VecColumn},
    options::LoPhatOptions,
    utils::anti_transpose,
};
use serde::Serialize;
use std::{collections::HashMap, fs::File, io::BufWriter, marker::PhantomData, ops::Deref, thread};

use crate::indexing::Permutation;
use crate::{
    builders::{
        build_d_cok, build_d_dom, build_d_im, build_d_ker, build_d_rel,
        compute_dom_first_permutation, decomp_cycle_idxs,
    },
    indexing::DensePermutation,
};

#[derive(Debug, Clone, Default)]
/// Coordinate maps shared by the six decompositions.
pub struct EnsembleMetadata {
    /// Original indices to domain-first indices, preserving order within each block.
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
/// Six in-memory decompositions with the metadata needed to extract diagrams.
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
/// Six serialized decompositions with shared metadata. Reading advances file cursors.
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
/// [`crate::diagrams::PersistenceDiagram::anti_transpose`]. Options pass through.
pub fn decompose_cod<Algo: DecompositionAlgo<VecColumn, Options = LoPhatOptions>>(
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
pub fn decompose_dom_cok<Algo: DecompositionAlgo<VecColumn, Options = LoPhatOptions>>(
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
/// The permutation must match the inclusion; see [`build_d_im`] and [`build_d_ker`].
pub fn decompose_im_ker<Algo: DecompositionAlgo<VecColumn, Options = LoPhatOptions>>(
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
    // Need to use v columns of Dim later, also no anti-transpose or clearing since D^2 != 0
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
pub fn decompose_rel<Algo: DecompositionAlgo<VecColumn, Options = LoPhatOptions>>(
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

/// Compute all six decompositions of an inclusion A into B over F2.
///
/// `boundary_matrix[j]` contains sorted, distinct nonzero row indices less than j.
/// `dimensions` has one entry per column; boundaries lower degree by one and
/// square to zero. `cols_in_dom` contains distinct in-range indices closed under
/// boundary. Its order is arbitrary and is normalized internally. Inputs are
/// assumed valid, not comprehensively validated; violations may panic or produce
/// incorrect diagrams. Column indices define filtration order.
///
/// Four scoped workers reduce codomain, domain/cokernel, image/kernel, and relative
/// matrices. `num_threads` limits each reduction, not the total worker count;
/// zero requests automatic counts. All results remain in memory. Call
/// [`DecompositionEnsemble::all_diagrams`] to restore original index coordinates.
pub fn all_decompositions<
    Algo: DecompositionAlgo<VecColumn, Options = LoPhatOptions>,
    UsizeSlice: Deref<Target = [usize]>,
>(
    boundary_matrix: &[UsizeSlice],
    dimensions: &[usize],
    cols_in_dom: &[usize],
    num_threads: usize,
) -> DecompositionEnsemble<VecColumn, Algo>
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

    let sz_codomain = boundary_matrix.len();
    let sz_domain = cols_in_dom.len();
    let cols_in_dom = cols_in_dom.iter().copied().sorted().collect_vec();
    let dom_first_permutation = compute_dom_first_permutation(sz_codomain, &cols_in_dom);
    let d_cod: Vec<VecColumn> = (0..sz_codomain)
        .map(|idx| {
            let column = boundary_matrix[idx].iter().copied().collect_vec();
            VecColumn::from((dimensions[idx], column))
        })
        .collect();

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
    DecompositionEnsemble {
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
    }
}

/// Serialize a value with bincode into a temporary file and release the value.
/// Returns a separate read handle positioned at the start. The temporary pathname
/// is removed when the writer is dropped; the returned handle keeps data alive.
/// Use [`crate::diagrams::from_file`] for LoPhat decompositions.
///
/// # Panics
/// Panics if temporary-file creation, reopening, or serialization fails.
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

/// Compute all six decompositions of an inclusion A into B over F2.
///
/// `boundary_matrix[j]` contains sorted, distinct nonzero row indices less than j.
/// `dimensions` has one entry per column; boundaries lower degree by one and
/// square to zero. `cols_in_dom` contains distinct in-range indices closed under
/// boundary. Its order is arbitrary and is normalized internally. Inputs are
/// assumed valid, not comprehensively validated; violations may panic or produce
/// incorrect diagrams. Column indices define filtration order.
///
/// Four scoped workers reduce codomain, domain/cokernel, image/kernel, and relative
/// matrices. `num_threads` limits each reduction, not the total worker count;
/// zero requests automatic counts. All results remain in memory. Call
/// [`DecompositionEnsemble::all_diagrams`] to restore original index coordinates.
/// Compute the same reductions as [`all_decompositions`] using temporary files.
///
/// Inputs, assumptions, and thread-count semantics are identical. Reduction groups
/// run sequentially and release decompositions after serialization. This is not
/// fully out-of-core: [`FileEnsemble::all_diagrams`] reloads all six at once.
///
/// # Panics
/// Panics on temporary-file or serialization errors. LoPhat 0.11.0 also panics
/// when serializing an empty decomposition, e.g. for an empty domain or quotient.
pub fn all_decompositions_slow<
    Algo: DecompositionAlgo<VecColumn, Options = LoPhatOptions>,
    UsizeSlice: Deref<Target = [usize]>,
>(
    boundary_matrix: &[UsizeSlice],
    dimensions: &[usize],
    cols_in_dom: &[usize],
    num_threads: usize,
) -> FileEnsemble
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

    let sz_codomain = boundary_matrix.len();
    let sz_domain = cols_in_dom.len();
    let cols_in_dom = cols_in_dom.iter().copied().sorted().collect_vec();
    let dom_first_permutation = compute_dom_first_permutation(sz_codomain, &cols_in_dom);

    let d_cod: Vec<VecColumn> = (0..sz_codomain)
        .map(|idx| {
            let column = boundary_matrix[idx].iter().copied().collect_vec();
            VecColumn::from((dimensions[idx], column))
        })
        .collect();
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

    FileEnsemble {
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
    }
}
