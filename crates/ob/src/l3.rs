//! L3 (individual-order) book with per-level intrusive FIFO queues.
//!
//! The L2 [`crate::glass::Glass`] answers "how much rests at each price";
//! this module answers "which orders rest there, in what queue position" —
//! required by execution-probability models (Cont–de Larrard style) and by
//! self-trade-prevention logic.
//!
//! Design (production-grade, O(1) hot ops):
//! * **Slab arena** of orders (`Vec` + free list) indexed by dense `u32`
//!   handles; `HashMap` only maps venue order-id -> handle.
//! * Each active order is linked into its price level's **intrusive
//!   doubly-linked FIFO queue** via `prev`/`next` handle fields, so cancel
//!   is O(1) (paper-style tombstones are unnecessary — we can unlink).
//! * Levels are held in a `HashMap<price, LevelHead>`; the L2 glass stays
//!   the source of truth for price ordering.

use crate::types::Side;
use std::collections::HashMap;

const NIL: u32 = u32::MAX;

#[derive(Clone, Debug)]
struct SlabOrder {
    /// Venue-assigned order id (engine-monotonic, like the venue's OrderId).
    pub id: u64,
    pub side: Side,
    pub price_ticks: u64,
    pub open_lots: u64,
    pub ts_ns: i64,
    /// Intrusive queue links (slab handles).
    prev: u32,
    next: u32,
    /// Price of the level this order is linked into (for O(1) unlink bookkeeping).
    level_price: u64,
}

#[derive(Clone, Debug)]
struct LevelHead {
    head: u32, // NIL when empty
    tail: u32,
    total_lots: u64,
    count: u32,
}

// CRITICAL: the default must be NIL links, NOT 0 — handle 0 is a valid
// slab slot. The derived Default (all zeros) silently linked fresh
// levels to the first order ever allocated, corrupting the intrusive
// queues; caught by the multi-level quoting stress test
// (engine/tests/l3_invariant.rs).
impl Default for LevelHead {
    fn default() -> Self {
        LevelHead {
            head: NIL,
            tail: NIL,
            total_lots: 0,
            count: 0,
        }
    }
}

/// A resting order as seen by the market maker.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct L3Order {
    pub id: u64,
    pub side: Side,
    pub price_ticks: u64,
    pub open_lots: u64,
    pub ts_ns: i64,
}

/// One execution against a resting order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Fill {
    pub order_id: u64,
    pub side: Side,
    pub price_ticks: u64,
    pub filled_lots: u64,
    /// Remaining lots on the order after this fill (0 => fully filled).
    pub remaining_lots: u64,
}

/// L3 book for one instrument.
pub struct Level3 {
    slab: Vec<SlabOrder>,
    free: Vec<u32>,
    by_id: HashMap<u64, u32>,
    levels: HashMap<u64, LevelHead>,
}

impl Level3 {
    pub fn new() -> Level3 {
        Level3 {
            slab: Vec::new(),
            free: Vec::new(),
            by_id: HashMap::new(),
            levels: HashMap::new(),
        }
    }

    pub fn order_count(&self) -> usize {
        self.by_id.len()
    }

    pub fn level_count(&self) -> usize {
        self.levels.len()
    }

    /// Total open lots resting at `price_ticks`.
    pub fn level_lots(&self, price_ticks: u64) -> u64 {
        self.levels.get(&price_ticks).map_or(0, |l| l.total_lots)
    }

    /// Resting order by venue id.
    pub fn get(&self, id: u64) -> Option<L3Order> {
        self.by_id
            .get(&id)
            .map(|&h| (&self.slab[h as usize]).into())
    }

    /// Add a new resting order (FIFO append at its level).
    pub fn add(&mut self, id: u64, side: Side, price_ticks: u64, lots: u64, ts_ns: i64) {
        debug_assert!(lots > 0);
        debug_assert!(!self.by_id.contains_key(&id), "duplicate order id");
        let handle = self.alloc();
        self.slab[handle as usize] = SlabOrder {
            id,
            side,
            price_ticks,
            open_lots: lots,
            ts_ns,
            prev: NIL,
            next: NIL,
            level_price: price_ticks,
        };
        let lvl = self.levels.entry(price_ticks).or_default();
        if lvl.tail != NIL {
            let t = lvl.tail as usize;
            self.slab[t].next = handle;
            self.slab[handle as usize].prev = lvl.tail;
            lvl.tail = handle;
        } else {
            lvl.head = handle;
            lvl.tail = handle;
        }
        lvl.total_lots += lots;
        lvl.count += 1;
        self.by_id.insert(id, handle);
    }

    /// Reduce a specific order by `lots` (our-order partial fill); removes
    /// it when exhausted. Returns the new open lots, or None if unknown.
    pub fn reduce_order(&mut self, id: u64, lots: u64) -> Option<u64> {
        let &h = self.by_id.get(&id)?;
        let price = self.slab[h as usize].level_price;
        let take = lots.min(self.slab[h as usize].open_lots);
        self.slab[h as usize].open_lots -= take;
        let remaining = self.slab[h as usize].open_lots;
        if let Some(lvl) = self.levels.get_mut(&price) {
            lvl.total_lots = lvl.total_lots.saturating_sub(take);
        }
        if remaining == 0 {
            self.by_id.remove(&id);
            self.unlink(h);
            if let Some(lvl) = self.levels.get_mut(&price) {
                lvl.count -= 1;
                if lvl.head == NIL {
                    self.levels.remove(&price);
                }
            }
            self.free.push(h);
        }
        Some(remaining)
    }

    /// Cancel an order; returns the removed order if it existed.
    pub fn cancel(&mut self, id: u64) -> Option<L3Order> {
        let handle = self.by_id.remove(&id)?;
        let price = self.slab[handle as usize].level_price;
        let order: L3Order = (&self.slab[handle as usize]).into();
        self.unlink(handle);
        let lvl = self.levels.get_mut(&price).unwrap();
        lvl.total_lots = lvl.total_lots.saturating_sub(order.open_lots);
        lvl.count -= 1;
        if lvl.head == NIL {
            self.levels.remove(&price);
        }
        self.free.push(handle);
        Some(order)
    }

    /// Consume up to `lots` from the front of the FIFO queue at
    /// `price_ticks` (a market order sweeping this level). Returns the
    /// fills in execution order. Orders fully consumed are removed.
    pub fn execute_fifo(&mut self, price_ticks: u64, lots: u64) -> Vec<Fill> {
        let mut fills = Vec::new();
        let mut remaining = lots;
        while remaining > 0 {
            let head = match self.levels.get(&price_ticks) {
                Some(l) if l.head != NIL => l.head,
                _ => break,
            };
            let h = head as usize;
            let order_id = self.slab[h].id;
            let side = self.slab[h].side;
            let open = self.slab[h].open_lots;
            let take = open.min(remaining);
            self.slab[h].open_lots -= take;
            remaining -= take;
            if let Some(lvl) = self.levels.get_mut(&price_ticks) {
                lvl.total_lots -= take;
            }
            let fully = self.slab[h].open_lots == 0;
            fills.push(Fill {
                order_id,
                side,
                price_ticks,
                filled_lots: take,
                remaining_lots: self.slab[h].open_lots,
            });
            if fully {
                self.by_id.remove(&order_id);
                self.unlink(head);
                if let Some(lvl) = self.levels.get_mut(&price_ticks) {
                    lvl.count -= 1;
                    if lvl.head == NIL {
                        self.levels.remove(&price_ticks);
                        break;
                    }
                }
            }
        }
        fills
    }

    /// Debug invariant check: per-level `count`/`total_lots` must match
    /// the linked-list walk; every listed order must be in `by_id` and
    /// vice versa. Returns the first violation found.
    pub fn invariant_violation(&self) -> Option<String> {
        let mut listed_ids = 0usize;
        for (&price, lvl) in self.levels.iter() {
            let mut n = 0usize;
            let mut tot = 0u64;
            let mut cur = lvl.head;
            let mut guard = 0usize;
            while cur != NIL {
                guard += 1;
                if guard > self.slab.len() + 2 {
                    return Some(format!("level {price}: list cycle"));
                }
                let o = &self.slab[cur as usize];
                if o.price_ticks != price || o.level_price != price {
                    return Some(format!(
                        "level {price}: order id {} wrong price {} (level_price {})",
                        o.id, o.price_ticks, o.level_price
                    ));
                }
                if o.open_lots == 0 {
                    return Some(format!("level {price}: zombie order {} (0 lots)", o.id));
                }
                n += 1;
                tot += o.open_lots;
                cur = o.next;
            }
            if lvl.count as usize != n {
                return Some(format!("level {price}: count {} vs list {}", lvl.count, n));
            }
            if lvl.total_lots != tot {
                return Some(format!("level {price}: total_lots {} vs list {}", lvl.total_lots, tot));
            }
            listed_ids += n;
        }
        if listed_ids != self.by_id.len() {
            return Some(format!("by_id {} vs listed {}", self.by_id.len(), listed_ids));
        }
        None
    }

    /// Number of *orders* ahead of `id` in its level's queue (0 = at the
    /// front). `None` if the order is unknown.
    pub fn queue_ahead_orders(&self, id: u64) -> Option<u32> {
        let &h = self.by_id.get(&id)?;
        let mut cur = self.slab[h as usize].prev;
        let mut n = 0u32;
        while cur != NIL {
            n += 1;
            cur = self.slab[cur as usize].prev;
        }
        Some(n)
    }

    /// Total *lots* resting ahead of `id` in its level's queue — the queue
    /// depth that must be consumed before `id` starts filling.
    pub fn queue_ahead_lots(&self, id: u64) -> Option<u64> {
        let &h = self.by_id.get(&id)?;
        let mut cur = self.slab[h as usize].prev;
        let mut lots = 0u64;
        while cur != NIL {
            lots += self.slab[cur as usize].open_lots;
            cur = self.slab[cur as usize].prev;
        }
        Some(lots)
    }

    /// Orders resting at a level, front-of-queue first.
    pub fn level_orders(&self, price_ticks: u64, max: usize) -> Vec<L3Order> {
        let mut out = Vec::new();
        if let Some(lvl) = self.levels.get(&price_ticks) {
            let mut cur = lvl.head;
            while cur != NIL && out.len() < max {
                out.push((&self.slab[cur as usize]).into());
                cur = self.slab[cur as usize].next;
            }
        }
        out
    }

    /// Drop every order (snapshot reset).
    pub fn clear(&mut self) {
        self.slab.clear();
        self.free.clear();
        self.by_id.clear();
        self.levels.clear();
    }

    // ------------------------------------------------------------------
    // internals
    // ------------------------------------------------------------------

    #[inline]
    fn alloc(&mut self) -> u32 {
        match self.free.pop() {
            Some(h) => h,
            None => {
                self.slab.push(SlabOrder {
                    id: 0,
                    side: Side::Bid,
                    price_ticks: 0,
                    open_lots: 0,
                    ts_ns: 0,
                    prev: NIL,
                    next: NIL,
                    level_price: 0,
                });
                (self.slab.len() - 1) as u32
            }
        }
    }

    #[inline]
    fn unlink(&mut self, handle: u32) {
        let (prev, next, price) = {
            let o = &self.slab[handle as usize];
            (o.prev, o.next, o.level_price)
        };
        if let Some(lvl) = self.levels.get_mut(&price) {
            if lvl.head == handle {
                lvl.head = next;
            }
            if lvl.tail == handle {
                lvl.tail = prev;
            }
        }
        if prev != NIL {
            self.slab[prev as usize].next = next;
        }
        if next != NIL {
            self.slab[next as usize].prev = prev;
        }
        self.slab[handle as usize].prev = NIL;
        self.slab[handle as usize].next = NIL;
    }
}

impl From<&SlabOrder> for L3Order {
    fn from(o: &SlabOrder) -> L3Order {
        L3Order {
            id: o.id,
            side: o.side,
            price_ticks: o.price_ticks,
            open_lots: o.open_lots,
            ts_ns: o.ts_ns,
        }
    }
}

impl Default for Level3 {
    fn default() -> Self {
        Level3::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fifo_and_partial_fills() {
        let mut b = Level3::new();
        b.add(1, Side::Bid, 100, 5, 1);
        b.add(2, Side::Bid, 100, 3, 2);
        b.add(3, Side::Bid, 100, 2, 3);
        assert_eq!(b.level_lots(100), 10);
        let fills = b.execute_fifo(100, 7);
        assert_eq!(
            fills,
            vec![
                Fill {
                    order_id: 1,
                    side: Side::Bid,
                    price_ticks: 100,
                    filled_lots: 5,
                    remaining_lots: 0
                },
                Fill {
                    order_id: 2,
                    side: Side::Bid,
                    price_ticks: 100,
                    filled_lots: 2,
                    remaining_lots: 1
                },
            ]
        );
        assert_eq!(b.get(2).unwrap().open_lots, 1);
        assert_eq!(b.get(3).unwrap().open_lots, 2);
        assert_eq!(b.level_lots(100), 3);
    }

    #[test]
    fn cancel_is_o1_and_keeps_queue() {
        let mut b = Level3::new();
        for i in 0..5u64 {
            b.add(i, Side::Ask, 200, 4, i as i64);
        }
        assert_eq!(b.queue_ahead_orders(3), Some(3));
        assert_eq!(b.queue_ahead_lots(3), Some(12));
        assert!(b.cancel(1).is_some());
        assert_eq!(b.queue_ahead_orders(3), Some(2));
        assert_eq!(b.queue_ahead_lots(3), Some(8));
        assert_eq!(b.level_orders(200, 10).len(), 4);
        // front is now order 0
        assert_eq!(b.level_orders(200, 10)[0].id, 0);
        // cancel tail then head
        assert!(b.cancel(4).is_some());
        assert!(b.cancel(0).is_some());
        let ids: Vec<u64> = b.level_orders(200, 10).iter().map(|o| o.id).collect();
        assert_eq!(ids, vec![2, 3]);
    }

    #[test]
    fn level_disappears_when_empty() {
        let mut b = Level3::new();
        b.add(9, Side::Bid, 55, 1, 0);
        let fills = b.execute_fifo(55, 1);
        assert_eq!(fills.len(), 1);
        assert_eq!(b.level_count(), 0);
        assert_eq!(b.order_count(), 0);
        assert_eq!(b.level_lots(55), 0);
    }
}
