use super::IndexMapping;
use std::cmp::Ordering;

pub trait Column: Send + Default {
    fn pivot(&self) -> Option<usize>;
    fn add_col(&mut self, other: &Self);
    fn add_entry(&mut self, entry: usize);
    fn reorder_rows(&mut self, mapping: &impl IndexMapping);
    fn unreorder_rows(&mut self, mapping: &impl IndexMapping);
}

#[derive(Debug, Default, Clone)]
pub struct VecColumn {
    pub internal: Vec<usize>,
}

#[derive(Debug, Default, Clone)]
pub struct AnnotatedColumn<T> {
    pub col: T,
    pub in_g: bool,
}

unsafe impl Sync for VecColumn {}

impl VecColumn {
    // Returns the index where we should try to insert next entry
    fn add_entry_starting_at(&mut self, entry: usize, starting_idx: usize) -> usize {
        let mut working_idx = starting_idx;
        while let Some(value_at_idx) = self.internal.iter().nth(working_idx) {
            match value_at_idx.cmp(&entry) {
                Ordering::Less => {
                    working_idx += 1;
                    continue;
                }
                Ordering::Equal => {
                    self.internal.remove(working_idx);
                    return working_idx;
                }
                Ordering::Greater => {
                    self.internal.insert(working_idx, entry);
                    return working_idx + 1;
                }
            }
        }
        // Bigger than all idxs in col - add to end
        self.internal.push(entry);
        return self.internal.len() - 1;
    }
}

impl Column for VecColumn {
    fn pivot(&self) -> Option<usize> {
        Some(*(self.internal.iter().last()?))
    }

    fn add_entry(&mut self, entry: usize) {
        self.add_entry_starting_at(entry, 0);
    }

    fn add_col(&mut self, other: &Self) {
        let mut working_idx = 0;
        for entry in other.internal.iter() {
            working_idx = self.add_entry_starting_at(*entry, working_idx);
        }
    }

    // TODO: Reimplement so that this happens in-place?
    fn reorder_rows(&mut self, mapping: &impl IndexMapping) {
        let mut new_col: Vec<usize> = self
            .internal
            .iter()
            .filter_map(|&row_idx| mapping.map(row_idx))
            .collect();
        new_col.sort();
        self.internal = new_col;
    }

    // TODO: Reimplement so that this happens in-place?
    fn unreorder_rows(&mut self, mapping: &impl IndexMapping) {
        let mut new_col: Vec<usize> = self
            .internal
            .iter()
            .filter_map(|&row_idx| mapping.inverse_map(row_idx))
            .collect();
        new_col.sort();
        self.internal = new_col;
    }
}
