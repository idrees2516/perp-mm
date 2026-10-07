//! Property tests: `Glass` vs `BTreeMap` oracle on market-data-like
//! clustered random workloads, and `OrderBook` L2/L3 consistency.

use ob::glass::Glass;
use ob::types::Side;
use ob::{BookEvent, OrderBook};
use std::collections::BTreeMap;

/// Deterministic xorshift64* — no external RNG dependency.
pub struct XorShift(u64);

impl XorShift {
    pub fn new(seed: u64) -> XorShift {
        XorShift(seed | 1)
    }
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }
    pub fn below(&mut self, n: u64) -> u64 {
        self.next_u64() % n
    }
}

/// Clustered random prices around a slowly drifting mid — mimics the
/// sequential locality the glass cached path exploits.
struct ClusteredWalk {
    mid: u64,
    rng: XorShift,
}

impl ClusteredWalk {
    fn next_price(&mut self) -> u64 {
        // 70% of updates within +/- 8 ticks of mid, else jump.
        let r = self.rng.below(100);
        let d = if r < 70 {
            self.rng.below(9)
        } else {
            self.rng.below(200)
        };
        if self.rng.below(2) == 0 {
            self.mid + d
        } else {
            self.mid.saturating_sub(d).max(1)
        }
    }
}

#[test]
fn glass_matches_btreemap_oracle() {
    for seed in [1u64, 0xDEADBEEF, 42] {
        let mut g = Glass::new();
        let mut oracle: BTreeMap<u64, u64> = BTreeMap::new();
        let mut walk = ClusteredWalk {
            mid: 100_000,
            rng: XorShift::new(seed),
        };
        for step in 0..6000u64 {
            let p = walk.next_price();
            let op = walk.rng.below(100);
            if op < 45 {
                // insert / set
                let v = 1 + walk.rng.below(1000);
                g.insert(p, v);
                oracle.insert(p, v);
            } else if op < 80 {
                // adjust
                let d = (walk.rng.below(200) as i64) - 100;
                let got = g.adjust(p, d);
                let cur = oracle.get(&p).copied();
                let expect = match cur {
                    Some(c) => {
                        let n = c as i64 + d;
                        if n <= 0 {
                            oracle.remove(&p);
                            Some(0)
                        } else {
                            oracle.insert(p, n as u64);
                            Some(n as u64)
                        }
                    }
                    None => {
                        if d > 0 {
                            oracle.insert(p, d as u64);
                            Some(d as u64)
                        } else {
                            None
                        }
                    }
                };
                assert_eq!(got, expect, "adjust mismatch step={} seed={}", step, seed);
            } else {
                // erase
                let got = g.erase(p);
                let expect = oracle.remove(&p).is_some();
                assert_eq!(got, expect, "erase mismatch step={}", step);
            }

            if step % 97 == 0 {
                // full order consistency
                let all: Vec<(u64, u64)> = {
                    let mut v = Vec::new();
                    let mut probe = 0u64;
                    while let Some((k, val)) = g.next_ge(probe) {
                        v.push((k, val));
                        probe = k + 1;
                    }
                    v
                };
                let want: Vec<(u64, u64)> = oracle.iter().map(|(k, v)| (*k, *v)).collect();
                assert_eq!(all, want, "iteration mismatch step={}", step);

                // extrema
                assert_eq!(g.min_key(), oracle.keys().next().copied());
                assert_eq!(g.max_key(), oracle.keys().next_back().copied());
                assert_eq!(g.len(), oracle.len());

                // successor/predecessor probes
                for _ in 0..32 {
                    let q = walk.rng.below(200_000);
                    let succ = g.next_ge(q);
                    let want_succ = oracle
                        .range(q..)
                        .next()
                        .map(|(k, v)| (*k, *v));
                    assert_eq!(succ, want_succ, "next_ge({}) step={}", q, step);
                    let pred = g.prev_le(q);
                    let want_pred = oracle
                        .range(..=q)
                        .next_back()
                        .map(|(k, v)| (*k, *v));
                    assert_eq!(pred, want_pred, "prev_le({}) step={}", q, step);
                }
            }
        }
    }
}

#[test]
fn glass_cached_path_hot_path_consistency() {
    // Hammer near-identical consecutive keys (cached path length ~1).
    let mut g = Glass::new();
    let mut oracle: BTreeMap<u64, u64> = BTreeMap::new();
    let mut rng = XorShift::new(7);
    let mut mid = 500_000u64;
    for _ in 0..4000 {
        let drift = (rng.below(5) as i64) - 2; // -2..=2
        mid = (((mid as i64) + drift).max(1)) as u64;
        let p = mid + rng.below(3);
        let v = rng.below(50) + 1;
        g.insert(p, v);
        oracle.insert(p, v);
    }
    let all: Vec<(u64, u64)> = {
        let mut v = Vec::new();
        let mut probe = 0u64;
        while let Some((k, val)) = g.next_ge(probe) {
            v.push((k, val));
            probe = k + 1;
        }
        v
    };
    let want: Vec<(u64, u64)> = oracle.iter().map(|(k, v)| (*k, *v)).collect();
    assert_eq!(all, want);
}

#[test]
fn orderbook_l2_l3_consistency() {
    let mut book = OrderBook::new();
    let mut rng = XorShift::new(99);
    let mut live: Vec<(u64, Side, u64, u64)> = Vec::new(); // id, side, price, lots
    let mut next_id = 1u64;
    for _ in 0..3000 {
        let op = rng.below(100);
        if op < 55 || live.is_empty() {
            let side = if rng.below(2) == 0 { Side::Bid } else { Side::Ask };
            let price = if side == Side::Bid {
                100 - 1 - rng.below(10)
            } else {
                101 + rng.below(10)
            };
            let lots = 1 + rng.below(30);
            book.apply(BookEvent::NewOrder {
                id: next_id,
                side,
                price_ticks: price,
                lots,
                ts_ns: next_id as i64,
            });
            live.push((next_id, side, price, lots));
            next_id += 1;
        } else if op < 85 {
            // cancel random live order
            let idx = rng.below(live.len() as u64) as usize;
            let (id, ..) = live.swap_remove(idx);
            book.apply(BookEvent::CancelOrder { id });
        } else {
            // sweep a few lots
            let aggressor = if rng.below(2) == 0 { Side::Bid } else { Side::Ask };
            let (limit, lots) = if aggressor == Side::Bid {
                (110u64, 1 + rng.below(40))
            } else {
                (90u64, 1 + rng.below(40))
            };
            let fills = book.apply(BookEvent::Sweep {
                limit_ticks: limit,
                lots,
                aggressor,
                ts_ns: 0,
            });
            // remove fully-filled orders from `live`
            let filled: std::collections::HashSet<u64> = fills
                .iter()
                .filter(|f| f.remaining_lots == 0)
                .map(|f| f.order_id)
                .collect();
            if !filled.is_empty() {
                live.retain(|(id, _, _, _)| !filled.contains(id));
            }
            // decrement partially filled orders
            for f in fills.iter().filter(|f| f.remaining_lots > 0) {
                if let Some(rec) = live.iter_mut().find(|(id, _, _, _)| *id == f.order_id) {
                    rec.3 = f.remaining_lots;
                }
            }
        }
        if rng.below(37) == 0 {
            // L2 totals must equal the live-order sums (bids and asks rest
            // at disjoint prices by construction: bids <= 100, asks >= 101).
            let stats = book.stats();
            let live_bid: u64 = live
                .iter()
                .filter(|(_, s, _, _)| *s == Side::Bid)
                .map(|(_, _, _, l)| *l)
                .sum();
            let live_ask: u64 = live
                .iter()
                .filter(|(_, s, _, _)| *s == Side::Ask)
                .map(|(_, _, _, l)| *l)
                .sum();
            assert_eq!(stats.bid_lots, live_bid);
            assert_eq!(stats.ask_lots, live_ask);
            assert_eq!(stats.l3_orders, live.len());
            assert!(!stats.crossed);
        }
    }
}
