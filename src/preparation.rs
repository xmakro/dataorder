//! Optional diagnostics collected while preparing an order.

use crate::order::Node;

/// A report from [`Order::prepare`](crate::Order::prepare).
/// Contains metadata only, without enumerating records. Ordinary constructors
/// do not collect or retain these diagnostics.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct Preparation {
    /// Final compiled nodes in preorder. Paths refer to the simplified tree,
    /// whose children can differ from the original configuration.
    pub nodes: Vec<PreparedNode>,
    /// Quotas for every weighted node, in original configuration order.
    /// Paths refer to the original configuration, including nodes later removed
    /// by a surrounding slice or zero repetition.
    pub weighted: Vec<WeightedAllocation>,
}

/// One node of the final compiled order.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct PreparedNode {
    /// Child indices from the compiled root; an empty path identifies the root.
    pub path: Vec<usize>,
    /// Operation after simplification.
    pub kind: PreparedKind,
    /// Number of positions produced by this node; may exceed `usize` internally.
    pub len: u64,
    /// Index into [`Order::sources`](crate::Order::sources) for a source node.
    pub source_ordinal: Option<usize>,
}

/// An operation retained in the compiled tree.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum PreparedKind {
    /// No elements.
    Empty,
    /// A contiguous range of a source.
    Source,
    /// Concatenated children.
    Concat,
    /// Interleaved children, including compiled weighted mixes.
    Mix,
    /// A seeded permutation.
    Shuffle,
    /// Repeated epochs, possibly with a truncated last epoch.
    Repeat,
    /// A contiguous range of a child.
    Slice,
    /// A regular stride over a child.
    Stride,
}

/// Exact quotas assigned to one original [`Seq::Weighted`](crate::Seq::Weighted) node.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct WeightedAllocation {
    /// Child indices from the original configuration root.
    pub path: Vec<usize>,
    /// Assigned counts in original part order, including zero shares.
    /// Counts describe this node before any enclosing transformations.
    pub counts: Vec<u64>,
}

impl Preparation {
    pub(crate) fn new(root: &Node, mut weighted: Vec<WeightedAllocation>) -> Self {
        let mut nodes = Vec::new();
        let mut work = vec![(root, Vec::new())];
        while let Some((node, path)) = work.pop() {
            let kind = match node {
                Node::Empty => PreparedKind::Empty,
                Node::Source { .. } => PreparedKind::Source,
                Node::Concat { .. } => PreparedKind::Concat,
                Node::Mix { .. } => PreparedKind::Mix,
                Node::Shuffle { .. } => PreparedKind::Shuffle,
                Node::Repeat { .. } => PreparedKind::Repeat,
                Node::Slice { .. } => PreparedKind::Slice,
                Node::Stride { .. } => PreparedKind::Stride,
            };
            let source_ordinal = if let Node::Source { src, .. } = node { Some(*src as usize) } else { None };
            nodes.push(PreparedNode { path: path.clone(), kind, len: node.len(), source_ordinal });
            match node {
                Node::Concat { children, .. } | Node::Mix { children, .. } => {
                    for (i, child) in children.iter().enumerate().rev() {
                        let mut child_path = path.clone();
                        child_path.push(i);
                        work.push((child, child_path));
                    }
                }
                Node::Shuffle { child, .. } | Node::Repeat { child, .. } | Node::Slice { child, .. } | Node::Stride { child, .. } => {
                    let mut child_path = path;
                    child_path.push(0);
                    work.push((child, child_path));
                }
                Node::Empty | Node::Source { .. } => {}
            }
        }
        weighted.sort_by(|a, b| a.path.cmp(&b.path));
        Self { nodes, weighted }
    }
}
