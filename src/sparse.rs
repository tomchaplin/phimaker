//! Sparse F2 arithmetic on supports of nonzero coordinates.

/// Add two sorted, duplicate-free supports, returning their symmetric
/// difference. Use for incremental reductions where the next column depends on
/// this sum.
pub(crate) fn sum_mod2(left: &[usize], right: &[usize]) -> Vec<usize> {
	let mut out = Vec::with_capacity(left.len() + right.len());
	let (mut i, mut j) = (0, 0);
	while i < left.len() && j < right.len() {
		match left[i].cmp(&right[j]) {
			std::cmp::Ordering::Less => {
				out.push(left[i]);
				i += 1;
			},
			std::cmp::Ordering::Greater => {
				out.push(right[j]);
				j += 1;
			},
			std::cmp::Ordering::Equal => {
				i += 1;
				j += 1;
			},
		}
	}
	out.extend_from_slice(&left[i..]);
	out.extend_from_slice(&right[j..]);
	out
}

/// Add several supports over F2, returning a sorted, duplicate-free support.
/// Each occurrence contributes one, including repeats within an input support;
/// input order is arbitrary, and an empty iterator gives the zero vector.
///
/// Collect all entries, sort once, then cancel equal entries by parity in
/// place. This avoids repeatedly copying a growing intermediate sum when only
/// the final result is needed. It temporarily stores all input entries, even
/// those that cancel; use [`sum_mod2`] when each intermediate result is needed.
pub(crate) fn sum_many_mod2<I, S>(supports: I) -> Vec<usize>
where
	I: IntoIterator<Item = S>,
	S: AsRef<[usize]>,
{
	let mut sum = Vec::new();
	for support in supports {
		sum.extend_from_slice(support.as_ref());
	}
	sum.sort_unstable();
	let (mut read, mut write) = (0, 0);
	while read < sum.len() {
		let entry = sum[read];
		let mut end = read + 1;
		while end < sum.len() && sum[end] == entry {
			end += 1;
		}
		if (end - read) % 2 == 1 {
			sum[write] = entry;
			write += 1;
		}
		read = end;
	}
	sum.truncate(write);
	sum
}

#[cfg(test)]
mod tests {
	use super::{sum_many_mod2, sum_mod2};

	fn support(mask: usize) -> Vec<usize> {
		(0..4).filter(|&index| mask & (1 << index) != 0).collect()
	}

	/// Check every triple of four-coordinate vectors against bitwise XOR,
	/// including zero inputs, complete cancellation, and triple occurrences.
	#[test]
	fn sparse_sums_agree_with_binary_arithmetic() {
		for first in 0..16 {
			for second in 0..16 {
				let left = support(first);
				let right = support(second);
				assert_eq!(sum_mod2(&left, &right), support(first ^ second));
				for third in 0..16 {
					assert_eq!(
						sum_many_mod2([&left, &right, &support(third)]),
						support(first ^ second ^ third),
					);
				}
			}
		}
	}

	/// Count parity rather than deduplicating, even for unsorted entries within
	/// one support; coordinate values include zero and the largest usize.
	#[test]
	fn many_supports_cancel_repeated_indices() {
		assert_eq!(
			sum_many_mod2([
				vec![usize::MAX, 0, 3, 3, 3, 1, 1],
				vec![0, 2, usize::MAX],
				vec![usize::MAX, 2, 2],
			]),
			vec![2, 3, usize::MAX],
		);
		assert_eq!(sum_many_mod2([vec![7, 7, 7, 7]]), Vec::<usize>::new());
	}

	/// Both no summands and any collection of empty summands represent zero.
	#[test]
	fn empty_sums_are_zero() {
		assert!(sum_many_mod2(std::iter::empty::<&[usize]>()).is_empty());
		assert!(sum_many_mod2([&[] as &[usize]; 3]).is_empty());
	}
}
