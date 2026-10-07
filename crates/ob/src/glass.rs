//! `glass` — ordered set / map over integer price keys (arXiv:2506.13991).
//!
//! Faithful Rust implementation of the data structure proposed in
//! *V. Krapivensky, "glass: ordered set data structure for client-side order
//! books" (arXiv:2506.13991)*:
//!
//! * **Uncompressed 32-way trie** over 64-bit integer keys, sliced 5 bits per
//!   level: depth = ceil(64/5) = 13 nodes, **independent of the number of
//!   stored levels** (contrast: O(log n) red-black / B-trees such as
//!   `std::map` or `BTreeMap`, whose depth grows with n and whose rebalancing
//!   touches many cache lines).
//! * **Cached path** exploiting the sequential locality of market data: the
//!   path from the root to the last touched key is cached; a new lookup
//!   jumps directly to the lowest common ancestor using one XOR plus a
//!   leading-zeros count (`clz`), i.e. O(1) arithmetic, then descends only
//!   the diverging suffix. Empirically consecutive order-book updates touch
//!   prices within a few ticks, so the diverging suffix is typically 0-2
//!   levels.
//! * **Flat node arena + O(1) slot allocator** (free list threaded through a
//!   `Vec`), parent pointers for upward walks, and a 32-bit child bitmap per
//!   node so "first/last present child" is a single `trailing_zeros` /
//!   `leading_zeros`.
//! * **`adjust(key, delta)`** — the exact incremental-update primitive from
//!   the paper: add `delta` to the level's amount, deleting the level when
//!   the amount reaches zero and creating it when it did not exist (venue
//!   delta semantics).
//!
//! The paper reports 6-30x speedups over `std::map` on market-data-like
//! workloads (insert/erase/find/next) and ~2-3.8x on iteration of the best
//! 25 prices; our `bench` crate reproduces the comparison against
//! `BTreeMap` on this machine.
//!
//! This is a *map* (key -> amount). The order book keeps one instance for
//! the bid side (queried as a max-glass) and one for the ask side
//! (min-glass), exactly as the paper prescribes.

/// Sentinel index meaning "no child". Node 0 is the root and is never a
/// child, so any real child index is >= 1.
const SENTINEL: u32 = u32::MAX;

/// Bits per trie level (32-way branching).
const C: u32 = 5;
/// Number of trie levels above the leaf: 12 full 5-bit chunks + 1 final
/// 4-bit chunk = 64 bits. A leaf sits at depth 13.
const LEAF_DEPTH: u32 = 13;

#[inline]
fn chunk(key: u64, depth: u32) -> usize {
    debug_assert!(depth < LEAF_DEPTH);
    if depth == 12 {
        (key & 0xF) as usize
    } else {
        ((key >> (59 - 5 * depth)) & 0x1F) as usize
    }
}

/// Bits of `mask` strictly above position `c` (i.e. child indices > c).
#[inline]
fn bits_above(mask: u32, c: usize) -> u32 {
    if c >= 31 {
        0
    } else {
        mask & !((1u32 << (c + 1)) - 1)
    }
}

/// Bits of `mask` strictly below position `c` (i.e. child indices < c).
#[inline]
fn bits_below(mask: u32, c: usize) -> u32 {
    mask & ((1u32 << c) - 1)
}

#[derive(Clone)]
struct Node {
    /// Bit `i` set iff `children[i] != SENTINEL`.
    mask: u32,
    children: [u32; 32],
    parent: u32,
    /// Only meaningful at leaf depth.
    value: u64,
    /// Only meaningful at leaf depth (avoids key reconstruction on walks).
    key: u64,
}

impl Node {
    #[inline]
    fn new(parent: u32) -> Node {
        Node {
            mask: 0,
            children: [SENTINEL; 32],
            parent,
            value: 0,
            key: 0,
        }
    }
}

enum Descend {
    /// Leaf node index (path in `cached_path` is complete).
    Leaf(u32),
    /// No child at `depth` / `chunk`; `cached_path` holds nodes 0..=depth.
    Miss { depth: u32, chunk_idx: usize },
}

/// The glass ordered map: key = integer price ticks, value = lots at level.
pub struct Glass {
    nodes: Vec<Node>,
    /// Free-list of arena indices (paper: O(1) slot allocator).
    free: Vec<u32>,
    /// Cached path root -> last touched key; `cached_path[i]` is the node at
    /// depth `i`. Invariant: `cached_path[i]` is reached from
    /// `cached_path[i-1]` by consuming `chunk(cached_key, i-1)`.
    cached_path: Vec<u32>,
    cached_key: Option<u64>,
    /// Cached extrema (paper: eager min/max caching).
    min_key: Option<u64>,
    max_key: Option<u64>,
    size: usize,
}

impl Glass {
    /// Empty glass.
    pub fn new() -> Glass {
        Glass {
            nodes: vec![Node::new(SENTINEL)],
            free: Vec::new(),
            cached_path: vec![0],
            cached_key: None,
            min_key: None,
            max_key: None,
            size: 0,
        }
    }

    /// Empty glass with capacity for roughly `levels` distinct keys.
    pub fn with_capacity(levels: usize) -> Glass {
        // Upper bound on nodes for `levels` distinct keys with shared
        // prefixes is much smaller than levels * 13; reserve optimistically
        // but bounded.
        Glass {
            nodes: Vec::with_capacity((levels + 1).min(1 << 16) * 4 + 1),
            free: Vec::new(),
            cached_path: vec![0],
            cached_key: None,
            min_key: None,
            max_key: None,
            size: 0,
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.size
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.size == 0
    }

    #[inline]
    pub fn min_key(&self) -> Option<u64> {
        self.min_key
    }

    #[inline]
    pub fn max_key(&self) -> Option<u64> {
        self.max_key
    }

    // ------------------------------------------------------------------
    // internals
    // ------------------------------------------------------------------

    #[inline]
    fn alloc(&mut self, parent: u32) -> u32 {
        let idx = match self.free.pop() {
            Some(i) => {
                self.nodes[i as usize] = Node::new(parent);
                i
            }
            None => {
                self.nodes.push(Node::new(parent));
                (self.nodes.len() - 1) as u32
            }
        };
        debug_assert_ne!(idx, SENTINEL);
        idx
    }

    #[inline]
    fn free_node(&mut self, idx: u32) {
        self.free.push(idx);
    }

    /// Number of leading 5-bit chunks shared by the cached key and `key`,
    /// clamped to the current cached-path length - 1 (the depth we can jump
    /// to). This is the paper's cached-path jump: one XOR + clz.
    fn jump_depth(&self, key: u64) -> u32 {
        match self.cached_key {
            None => 0,
            Some(ck) => {
                let diff = ck ^ key;
                let common = if diff == 0 {
                    LEAF_DEPTH
                } else {
                    // Chunk index of the highest differing bit.
                    (63 - diff.leading_zeros()) / C
                };
                common.min(self.cached_path.len() as u32 - 1)
            }
        }
    }

    /// Descend towards `key`, updating the cached path. Returns
    /// [`Descend::Leaf`] when the key exists, otherwise the first missing
    /// position. `cached_key` is set to `key` (the cached path always
    /// corresponds to it, possibly truncated).
    fn descend(&mut self, key: u64) -> Descend {
        let start = self.jump_depth(key) as usize;
        let mut node = self.cached_path[start];
        self.cached_path.truncate(start + 1);
        self.cached_key = Some(key);
        for d in start..LEAF_DEPTH as usize {
            let c = chunk(key, d as u32);
            let child = self.nodes[node as usize].children[c];
            if child == SENTINEL {
                return Descend::Miss {
                    depth: d as u32,
                    chunk_idx: c,
                };
            }
            node = child;
            self.cached_path.push(node);
        }
        Descend::Leaf(node)
    }

    /// Follow the smallest present child down to a leaf; returns (key, value).
    fn descend_leftmost(&self, mut node: u32) -> (u64, u64) {
        while !self.is_leaf(node) {
            let c = self.nodes[node as usize].mask.trailing_zeros() as usize;
            node = self.nodes[node as usize].children[c];
            debug_assert_ne!(node, SENTINEL);
        }
        let l = &self.nodes[node as usize];
        (l.key, l.value)
    }

    /// Follow the largest present child down to a leaf; returns (key, value).
    fn descend_rightmost(&self, mut node: u32) -> (u64, u64) {
        while !self.is_leaf(node) {
            let c = (31 - self.nodes[node as usize].mask.leading_zeros()) as usize;
            node = self.nodes[node as usize].children[c];
            debug_assert_ne!(node, SENTINEL);
        }
        let l = &self.nodes[node as usize];
        (l.key, l.value)
    }

    #[inline]
    fn is_leaf(&self, node: u32) -> bool {
        // Internal nodes always have at least one child (empty ones are
        // pruned on erase); the root of an empty set is excluded explicitly.
        self.nodes[node as usize].mask == 0 && node != 0
    }

    // ------------------------------------------------------------------
    // public API (paper's operation set)
    // ------------------------------------------------------------------

    /// `insert` — set the amount at `key` (creating or replacing the level).
    pub fn insert(&mut self, key: u64, value: u64) {
        match self.descend(key) {
            Descend::Leaf(i) => {
                self.nodes[i as usize].value = value;
                // extrema unchanged
            }
            Descend::Miss { depth, chunk_idx } => {
                let mut parent = *self.cached_path.last().unwrap();
                debug_assert_eq!(self.cached_path.len() as u32, depth + 1);
                for d in depth..LEAF_DEPTH {
                    let c = if d == depth {
                        chunk_idx
                    } else {
                        chunk(key, d)
                    };
                    let new = self.alloc(parent);
                    let pn = &mut self.nodes[parent as usize];
                    pn.children[c] = new;
                    pn.mask |= 1u32 << c;
                    parent = new;
                    self.cached_path.push(new);
                }
                // `parent` is now the leaf at depth 13.
                let leaf = &mut self.nodes[parent as usize];
                leaf.key = key;
                leaf.value = value;
                self.size += 1;
                self.min_key = Some(match self.min_key {
                    Some(m) => m.min(key),
                    None => key,
                });
                self.max_key = Some(match self.max_key {
                    Some(m) => m.max(key),
                    None => key,
                });
            }
        }
    }

    /// `find` — amount at `key`, or `None`.
    pub fn find(&mut self, key: u64) -> Option<u64> {
        match self.descend(key) {
            Descend::Leaf(i) => Some(self.nodes[i as usize].value),
            Descend::Miss { .. } => None,
        }
    }

    /// `find` without cached-path mutation (cold path, e.g. diagnostics).
    pub fn find_readonly(&self, key: u64) -> Option<u64> {
        let mut node = 0u32;
        for d in 0..LEAF_DEPTH {
            let c = chunk(key, d);
            let child = self.nodes[node as usize].children[c];
            if child == SENTINEL {
                return None;
            }
            node = child;
        }
        Some(self.nodes[node as usize].value)
    }

    /// `contains` — membership test.
    pub fn contains(&mut self, key: u64) -> bool {
        self.find(key).is_some()
    }

    /// `adjust(pi, delta)` — the paper's incremental-update primitive:
    /// add `delta` (may be negative) to the amount at `key`; delete the
    /// level if the new amount is zero; create it (amount = max(delta, 0))
    /// if absent. Returns the new amount, or `None` if the key is absent
    /// and `delta <= 0`.
    pub fn adjust(&mut self, key: u64, delta: i64) -> Option<u64> {
        match self.descend(key) {
            Descend::Leaf(i) => {
                let cur = self.nodes[i as usize].value as i64;
                let new = cur + delta;
                if new <= 0 {
                    self.erase(key);
                    Some(0)
                } else {
                    self.nodes[i as usize].value = new as u64;
                    Some(new as u64)
                }
            }
            Descend::Miss { .. } => {
                if delta > 0 {
                    self.insert(key, delta as u64);
                    Some(delta as u64)
                } else {
                    None
                }
            }
        }
    }

    /// `erase` — remove the level at `key`; returns whether it existed.
    /// Prunes empty ancestors (the paper truncates the cached path to the
    /// surviving ancestor rather than invalidating it).
    pub fn erase(&mut self, key: u64) -> bool {
        match self.descend(key) {
            Descend::Miss { .. } => false,
            Descend::Leaf(leaf) => {
                let mut child = leaf;
                let mut depth = LEAF_DEPTH;
                let mut survivor_depth = 0;
                while depth > 0 {
                    let p = self.nodes[child as usize].parent;
                    let c = chunk(key, depth - 1);
                    debug_assert_eq!(self.nodes[p as usize].children[c], child);
                    self.nodes[p as usize].children[c] = SENTINEL;
                    self.nodes[p as usize].mask &= !(1u32 << c);
                    self.free_node(child);
                    if self.nodes[p as usize].mask != 0 || p == 0 {
                        survivor_depth = depth;
                        break;
                    }
                    child = p;
                    depth -= 1;
                }
                if depth == 0 {
                    // Emptied all the way to (but excluding) the root.
                    self.cached_path.truncate(1);
                } else {
                    self.cached_path.truncate(survivor_depth as usize);
                }
                self.size -= 1;
                self.cached_key = Some(key);
                if self.size == 0 {
                    self.min_key = None;
                    self.max_key = None;
                } else {
                    if self.min_key == Some(key) {
                        self.min_key = key
                            .checked_add(1)
                            .and_then(|k| self.next_ge_readonly(k).map(|(k, _)| k));
                    }
                    if self.max_key == Some(key) {
                        self.max_key = key
                            .checked_sub(1)
                            .and_then(|k| self.prev_le_readonly(k).map(|(k, _)| k));
                    }
                }
                true
            }
        }
    }

    /// `min` — smallest key and its amount.
    pub fn min(&self) -> Option<(u64, u64)> {
        self.min_key.map(|k| {
            let v = self.find_readonly(k).unwrap_or(0);
            (k, v)
        })
    }

    /// `max` — largest key and its amount.
    pub fn max(&self) -> Option<(u64, u64)> {
        self.max_key.map(|k| {
            let v = self.find_readonly(k).unwrap_or(0);
            (k, v)
        })
    }

    /// `next(k)` — smallest entry with key >= k (successor-or-equal).
    pub fn next_ge(&mut self, key: u64) -> Option<(u64, u64)> {
        let miss = match self.descend(key) {
            Descend::Leaf(i) => {
                let n = &self.nodes[i as usize];
                return Some((n.key, n.value));
            }
            Descend::Miss { depth, chunk_idx } => (depth, chunk_idx),
        };
        self.next_after_miss(key, miss.0, miss.1)
    }

    fn next_ge_readonly(&self, key: u64) -> Option<(u64, u64)> {
        let mut node = 0u32;
        let mut stack: Vec<(u32, usize)> = Vec::with_capacity(LEAF_DEPTH as usize);
        for d in 0..LEAF_DEPTH {
            let c = chunk(key, d);
            let child = self.nodes[node as usize].children[c];
            if child == SENTINEL {
                // right sibling inside the miss node
                let above = bits_above(self.nodes[node as usize].mask, c);
                if above != 0 {
                    let c2 = above.trailing_zeros() as usize;
                    let ch = self.nodes[node as usize].children[c2];
                    return Some(self.descend_leftmost(ch));
                }
                return self.next_from_stack(stack);
            }
            stack.push((node, c));
            node = child;
        }
        let n = &self.nodes[node as usize];
        Some((n.key, n.value))
    }

    /// Shared successor walk: try right-sibling at the miss node, then walk
    /// up the (already materialized) cached path.
    fn next_after_miss(
        &mut self,
        _key: u64,
        miss_depth: u32,
        miss_chunk: usize,
    ) -> Option<(u64, u64)> {
        // (a) right sibling inside the miss node
        let miss_node = self.cached_path[miss_depth as usize];
        let above = bits_above(self.nodes[miss_node as usize].mask, miss_chunk);
        if above != 0 {
            let c2 = above.trailing_zeros() as usize;
            let child = self.nodes[miss_node as usize].children[c2];
            return Some(self.descend_leftmost(child));
        }
        // (b) walk up: deepest ancestor whose path chunk has a right sibling
        for j in (0..miss_depth as usize).rev() {
            let c_j = {
                // recompute the chunk taken at depth j for `_key`
                let d = j as u32;
                if d == 12 {
                    (_key & 0xF) as usize
                } else {
                    ((_key >> (59 - 5 * d)) & 0x1F) as usize
                }
            };
            let anc = self.cached_path[j];
            let above = bits_above(self.nodes[anc as usize].mask, c_j);
            if above != 0 {
                let c2 = above.trailing_zeros() as usize;
                let child = self.nodes[anc as usize].children[c2];
                return Some(self.descend_leftmost(child));
            }
        }
        None
    }

    fn next_from_stack(&self, stack: Vec<(u32, usize)>) -> Option<(u64, u64)> {
        for (node, c) in stack.into_iter().rev() {
            let above = bits_above(self.nodes[node as usize].mask, c);
            if above != 0 {
                let c2 = above.trailing_zeros() as usize;
                let child = self.nodes[node as usize].children[c2];
                return Some(self.descend_leftmost(child));
            }
        }
        None
    }

    /// `prev(k)` — largest entry with key <= k (predecessor-or-equal).
    pub fn prev_le(&mut self, key: u64) -> Option<(u64, u64)> {
        let miss = match self.descend(key) {
            Descend::Leaf(i) => {
                let n = &self.nodes[i as usize];
                return Some((n.key, n.value));
            }
            Descend::Miss { depth, chunk_idx } => (depth, chunk_idx),
        };
        let (miss_depth, miss_chunk) = miss;
        // (a) left sibling inside the miss node
        let miss_node = self.cached_path[miss_depth as usize];
        let below = bits_below(self.nodes[miss_node as usize].mask, miss_chunk);
        if below != 0 {
            let c2 = 31 - below.leading_zeros();
            let child = self.nodes[miss_node as usize].children[c2 as usize];
            return Some(self.descend_rightmost(child));
        }
        // (b) walk up
        for j in (0..miss_depth as usize).rev() {
            let c_j = {
                let d = j as u32;
                if d == 12 {
                    (key & 0xF) as usize
                } else {
                    ((key >> (59 - 5 * d)) & 0x1F) as usize
                }
            };
            let anc = self.cached_path[j];
            let below = bits_below(self.nodes[anc as usize].mask, c_j);
            if below != 0 {
                let c2 = 31 - below.leading_zeros();
                let child = self.nodes[anc as usize].children[c2 as usize];
                return Some(self.descend_rightmost(child));
            }
        }
        None
    }

    fn prev_le_readonly(&self, key: u64) -> Option<(u64, u64)> {
        let mut node = 0u32;
        let mut stack: Vec<(u32, usize)> = Vec::with_capacity(LEAF_DEPTH as usize);
        for d in 0..LEAF_DEPTH {
            let c = chunk(key, d);
            let child = self.nodes[node as usize].children[c];
            if child == SENTINEL {
                // left sibling inside the miss node
                let below = bits_below(self.nodes[node as usize].mask, c);
                if below != 0 {
                    let c2 = 31 - below.leading_zeros();
                    let ch = self.nodes[node as usize].children[c2 as usize];
                    return Some(self.descend_rightmost(ch));
                }
                return self.prev_from_stack(stack);
            }
            stack.push((node, c));
            node = child;
        }
        let n = &self.nodes[node as usize];
        Some((n.key, n.value))
    }

    fn prev_from_stack(&self, stack: Vec<(u32, usize)>) -> Option<(u64, u64)> {
        for (node, c) in stack.into_iter().rev() {
            let below = bits_below(self.nodes[node as usize].mask, c);
            if below != 0 {
                let c2 = 31 - below.leading_zeros();
                let child = self.nodes[node as usize].children[c2 as usize];
                return Some(self.descend_rightmost(child));
            }
        }
        None
    }

    /// The `n` cheapest (lowest) entries, ascending — the ask ladder.
    pub fn best_ascending(&mut self, n: usize) -> Vec<(u64, u64)> {
        let mut out = Vec::with_capacity(n);
        let mut probe = match self.min_key {
            Some(k) => k,
            None => return out,
        };
        while out.len() < n {
            match self.next_ge(probe) {
                Some((k, v)) => {
                    out.push((k, v));
                    probe = match k.checked_add(1) {
                        Some(p) => p,
                        None => break,
                    };
                }
                None => break,
            }
        }
        out
    }

    /// The `n` highest entries, descending — the bid ladder.
    pub fn best_descending(&mut self, n: usize) -> Vec<(u64, u64)> {
        let mut out = Vec::with_capacity(n);
        let mut probe = match self.max_key {
            Some(k) => k,
            None => return out,
        };
        while out.len() < n {
            match self.prev_le(probe) {
                Some((k, v)) => {
                    out.push((k, v));
                    probe = match k.checked_sub(1) {
                        Some(p) => p,
                        None => break,
                    };
                }
                None => break,
            }
        }
        out
    }

    /// Total amount stored at keys in `[lo, hi]` (inclusive).
    pub fn sum_between(&mut self, lo: u64, hi: u64) -> u64 {
        let mut total = 0u64;
        let mut probe = lo;
        while let Some((k, v)) = self.next_ge(probe) {
            if k > hi {
                break;
            }
            total = total.saturating_add(v);
            probe = match k.checked_add(1) {
                Some(p) => p,
                None => break,
            };
        }
        total
    }

    /// Drain all entries (used when applying a snapshot reset).
    pub fn clear(&mut self) {
        self.nodes.clear();
        self.nodes.push(Node::new(SENTINEL));
        self.free.clear();
        self.cached_path = vec![0];
        self.cached_key = None;
        self.min_key = None;
        self.max_key = None;
        self.size = 0;
    }
}

impl Default for Glass {
    fn default() -> Self {
        Glass::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gap_seq(n: u64) -> Vec<u64> {
        // pseudo-random-ish clustered keys in a small range, like price ticks
        let mut x = 0x9E3779B97F4A7C15u64;
        (0..n)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                1000 + (x % 500)
            })
            .collect()
    }

    #[test]
    fn basic_insert_find_erase() {
        let mut g = Glass::new();
        assert_eq!(g.find(42), None);
        g.insert(42, 10);
        assert_eq!(g.find(42), Some(10));
        g.insert(43, 5);
        g.insert(7, 3);
        assert_eq!(g.min(), Some((7, 3)));
        assert_eq!(g.max(), Some((43, 5)));
        assert!(g.erase(42));
        assert_eq!(g.find(42), None);
        assert_eq!(g.len(), 2);
        assert!(!g.erase(42));
    }

    #[test]
    fn adjust_semantics() {
        let mut g = Glass::new();
        assert_eq!(g.adjust(100, 7), Some(7));
        assert_eq!(g.adjust(100, 3), Some(10));
        assert_eq!(g.adjust(100, -4), Some(6));
        assert_eq!(g.adjust(100, -6), Some(0)); // deleted
        assert_eq!(g.find(100), None);
        assert_eq!(g.adjust(100, -5), None); // absent, negative
    }

    #[test]
    fn next_prev() {
        let mut g = Glass::new();
        for k in [5u64, 10, 20, 30] {
            g.insert(k, k * 2);
        }
        assert_eq!(g.next_ge(0), Some((5, 10)));
        assert_eq!(g.next_ge(5), Some((5, 10)));
        assert_eq!(g.next_ge(6), Some((10, 20)));
        assert_eq!(g.next_ge(31), None);
        assert_eq!(g.prev_le(100), Some((30, 60)));
        assert_eq!(g.prev_le(30), Some((30, 60)));
        assert_eq!(g.prev_le(29), Some((20, 40)));
        assert_eq!(g.prev_le(4), None);
    }

    #[test]
    fn ladders_and_sums() {
        let mut g = Glass::new();
        let keys = gap_seq(200);
        for &k in &keys {
            g.insert(k, 1);
        }
        let asc = g.best_ascending(50);
        let desc = g.best_descending(50);
        assert_eq!(asc.len(), 50);
        assert_eq!(desc.len(), 50);
        let mut sorted = keys.clone();
        sorted.sort();
        sorted.dedup();
        for (i, &(k, _)) in asc.iter().enumerate() {
            assert_eq!(k, sorted[i]);
        }
        for (i, &(k, _)) in desc.iter().enumerate() {
            assert_eq!(k, sorted[sorted.len() - 1 - i]);
        }
        if let Some(lo) = sorted.first() {
            let hi = sorted[sorted.len() / 2];
            assert_eq!(g.sum_between(*lo, hi) as usize, sorted.len() / 2 + 1);
        }
    }

    #[test]
    fn erase_updates_extrema() {
        let mut g = Glass::new();
        g.insert(10, 1);
        g.insert(20, 1);
        g.insert(30, 1);
        assert!(g.erase(10));
        assert_eq!(g.min(), Some((20, 1)));
        assert!(g.erase(30));
        assert_eq!(g.max(), Some((20, 1)));
        assert!(g.erase(20));
        assert_eq!(g.min(), None);
        assert_eq!(g.max(), None);
        assert!(g.is_empty());
        // Re-insert after full clear-by-erase
        g.insert(1, 9);
        assert_eq!(g.find(1), Some(9));
    }

    #[test]
    fn cached_path_survives_truncation() {
        // Erase a key whose path is cached, then immediately look up a
        // nearby key — the truncated cached path must stay consistent.
        let mut g = Glass::new();
        for k in 0u64..64 {
            g.insert(k * 4, 1);
        }
        assert!(g.erase(124));
        assert_eq!(g.find(124), None);
        assert_eq!(g.find(120), Some(1));
        assert_eq!(g.next_ge(121), Some((128, 1))); // 124 erased, next is 128
    }
}
