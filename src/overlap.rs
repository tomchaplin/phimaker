use disjoint_sets::UnionFindNode;
use lophat::columns::{Column, VecColumn};

use crate::indexing::AnnotatedColumn;

#[derive(Clone, Copy)]
enum ChromaticIndex {
    Monochromatic(bool),
    Colourful,
}

impl From<bool> for ChromaticIndex {
    fn from(value: bool) -> Self {
        Self::Monochromatic(value)
    }
}

struct ClusterData {
    chromatic_index: ChromaticIndex,
    size: usize,
}

/// Computes the H_0 overlap between two disjoint point clouds,
/// forming the 0-skeleton of a filtered cell complex.

pub fn compute_zero_overlap(matrix: &Vec<AnnotatedColumn<VecColumn>>) -> Vec<(usize, usize)> {
    let mut node_list = vec![];
    let mut feature_list = vec![];
    for (idx, column) in matrix.iter().enumerate() {
        // Maintain list of nodes
        // We push None to list so that node index line up with column indexes
        let new_node = if column.col.dimension() == 0 {
            let data = ClusterData {
                chromatic_index: column.in_g.into(),
                size: 1,
            };
            Some(UnionFindNode::new(data))
        } else {
            None
        };
        node_list.push(new_node);

        // If we see an edge, merge the data accordingly
        if column.col.dimension() == 1 {
            let mut entries = column.col.entries();
            let source_idx = entries
                .next()
                .expect("Edge columns should have two non-zero boundary indices");
            let target_idx = entries
                .next()
                .expect("Edge columns should have two non-zero boundary indices");
            // Implicit assumption that source_idx < target_idx here
            let (left, right) = node_list.split_at_mut(target_idx);
            let source_node = left[source_idx].as_mut().unwrap();
            let mut target_node = right[0].as_mut().unwrap();
            if source_node.equiv(&target_node) {
                // This edge creates a loop - not element of MST
                continue;
            } else {
                // A merge is hapenning, we must update feature_list accordingly
                source_node.union_with(&mut target_node, |source_data, target_data| {
                    let (new_data, new_features) = merge_clusters(source_data, target_data, idx);
                    feature_list.extend_from_slice(new_features.as_slice());
                    new_data
                });
            }
        }
    }
    feature_list
}

fn merge_clusters(
    source_data: ClusterData,
    target_data: ClusterData,
    current_idx: usize,
) -> (ClusterData, Vec<(usize, usize)>) {
    // Size of the new cluster is the sum of its union summands
    let new_size = source_data.size + target_data.size;
    match (source_data.chromatic_index, target_data.chromatic_index) {
        (ChromaticIndex::Monochromatic(s_col), ChromaticIndex::Monochromatic(t_col)) => {
            // Cluster either stays monochromatic or becomes colourful
            let new_col = if s_col == t_col {
                ChromaticIndex::Monochromatic(s_col)
            } else {
                ChromaticIndex::Colourful
            };
            // If stays monochromatic, then one feature dies
            // Weight it with the size of the smaller cluster
            // Otherwise both features die and we weight with each of the cluster sizes
            let new_pts = if s_col == t_col {
                vec![(source_data.size.min(target_data.size), current_idx)]
            } else {
                vec![
                    (source_data.size, current_idx),
                    (target_data.size, current_idx),
                ]
            };
            let new_data = ClusterData {
                chromatic_index: new_col,
                size: new_size,
            };
            (new_data, new_pts)
        }
        (ChromaticIndex::Monochromatic(_), ChromaticIndex::Colourful) => {
            let new_data = ClusterData {
                chromatic_index: ChromaticIndex::Colourful,
                size: new_size,
            };
            // The monochromatic feature dies - weight it with its cluster size
            let new_pts = vec![(source_data.size, current_idx)];
            (new_data, new_pts)
        }
        (ChromaticIndex::Colourful, ChromaticIndex::Colourful) => {
            let new_data = ClusterData {
                chromatic_index: ChromaticIndex::Colourful,
                size: new_size,
            };
            // Both clusters already quotiented out - no death
            let new_pts = vec![];
            (new_data, new_pts)
        }
        _ => merge_clusters(target_data, source_data, current_idx),
    }
}
