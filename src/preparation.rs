//! Optional diagnostics collected while preparing an order.

use crate::Sampling;
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
    /// Every source in original configuration order, including sources removed by
    /// simplification. Join a compiled node's `source_ordinal` to this vector.
    pub sources: Vec<PreparedSource>,
    /// Counts and virtual-clock schedules for every original mix or weighted mix, including
    /// nodes subsequently removed by simplification. Paths are original paths.
    pub mixes: Vec<PreparedMix>,
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
    /// Parameters after folding slices, strides and other transformations.
    pub parameters: PreparedParameters,
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

/// Parameters of a compiled operation; lengths are in [`PreparedNode::len`].
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum PreparedParameters {
    /// Empty nodes and mixes have no additional positional parameters.
    None,
    /// Source indices start at this offset.
    Source {
        /// First index within the original source.
        offset: u64,
    },
    /// Starts of the children followed by the total length.
    Concat {
        /// Cumulative child boundaries.
        offsets: Vec<u64>,
    },
    /// The shuffle's local seed and folded source salt, before the order seed.
    Shuffle {
        /// Configured shuffle seed.
        seed: u64,
        /// Compiled source salt.
        salt: u64,
    },
    /// Epoch size and nesting depth used to derive epoch contexts.
    Repeat {
        /// Length of one complete epoch.
        child_len: u64,
        /// Number of enclosing repeats.
        depth: u32,
    },
    /// A contiguous child range.
    Slice {
        /// First child position.
        start: u64,
    },
    /// Child positions `offset + i * step`.
    Stride {
        /// Distance between child positions.
        step: u64,
        /// First child position.
        offset: u64,
    },
}

/// Original source metadata, read once during compilation.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct PreparedSource {
    /// Child indices from the original configuration root.
    pub path: Vec<usize>,
    /// Original source length, before any transformations.
    pub len: u64,
    /// Salt returned by the source during compilation.
    pub salt: u64,
}

/// An original mix and its independent schedules, before enclosing transformations.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct PreparedMix {
    /// Child indices from the original configuration root.
    pub path: Vec<usize>,
    /// Compiled part lengths or weighted quotas, including empty parts.
    pub counts: Vec<u64>,
    /// Schedules in original part order.
    pub sampling: Vec<Sampling>,
}

#[derive(Default)]
pub(crate) struct CompilationReport {
    pub weighted: Vec<WeightedAllocation>,
    pub sources: Vec<PreparedSource>,
    pub mixes: Vec<PreparedMix>,
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
    pub(crate) fn new(root: &Node, CompilationReport { mut weighted, sources, mut mixes }: CompilationReport) -> Self {
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
            let parameters = match node {
                Node::Empty | Node::Mix { .. } => PreparedParameters::None,
                Node::Source { offset, .. } => PreparedParameters::Source { offset: *offset },
                Node::Concat { offsets, .. } => PreparedParameters::Concat { offsets: offsets.clone() },
                Node::Shuffle { seed, salt, .. } => PreparedParameters::Shuffle { seed: *seed, salt: *salt },
                Node::Repeat { child_len, depth, .. } => PreparedParameters::Repeat { child_len: *child_len, depth: *depth },
                Node::Slice { start, .. } => PreparedParameters::Slice { start: *start },
                Node::Stride { step, offset, .. } => PreparedParameters::Stride { step: *step, offset: *offset },
            };
            nodes.push(PreparedNode { path: path.clone(), kind, len: node.len(), source_ordinal, parameters });
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
        mixes.sort_by(|a, b| a.path.cmp(&b.path));
        Self { nodes, weighted, sources, mixes }
    }
}
