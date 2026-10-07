//! Multi-level (ladder) quoting.
//!
//! Two strands are combined:
//! - **Pricing ladders** (Barzykin–Bergault–Guéant, arXiv:2112.02269,
//!   "Market making by an FX dealer: tiers, pricing ladders and hedging
//!   rates for optimal risk control"): size-tiered quotes — larger
//!   sizes transact at wider distances.
//! - **The fill-probability / post-fill-returns trade-off**
//!   ("The Market Maker's Dilemma", arXiv 2025): deeper quotes have
//!   lower fill intensity but fill predominantly on flow that is more
//!   adversely selected; modelled here as an adverse fraction
//!   `alpha ∈ [0,1)` of the distance lost on each deep fill.
//!
//! The policy places `K` levels per side at geometrically growing
//! distances around the GLFT optimal base, with sizes growing by tier
//! and tapering with inventory. The `effective_edge` computation gives
//! the net (post-adverse-selection) expected edge rate of the ladder,
//! which the strategy layer compares against the single-level quote to
//! pick the number of live levels; the engine-level Monte Carlo
//! (examples/study2.rs) validates the full loop with real sweeps.

use crate::glft::GlftAsymptotic;

/// Ladder policy parameters.
#[derive(Clone, Debug)]
pub struct LadderPolicy {
    /// Number of levels per side.
    pub levels: usize,
    /// Distance multiplier growth per tier (level i distance =
    /// base · (1 + tier_mult · i)).
    pub tier_mult: f64,
    /// Size growth per tier (level i size = base_size · (1 + size_mult·i)).
    pub size_mult: f64,
    /// Base size in lots (level 0).
    pub base_size: f64,
    /// Adverse fraction: expected mid move against a fill, as a fraction
    /// of that level's distance.
    pub adverse_alpha: f64,
}

impl Default for LadderPolicy {
    fn default() -> Self {
        LadderPolicy {
            levels: 3,
            tier_mult: 0.7,
            size_mult: 0.5,
            base_size: 4.0,
            adverse_alpha: 0.25,
        }
    }
}

impl LadderPolicy {
    /// Distance multipliers and sizes for the tiers:
    /// `(m_0..m_{K-1}, s_0..s_{K-1})`.
    pub fn tiers(&self) -> (Vec<f64>, Vec<f64>) {
        let k = self.levels.max(1);
        let mut ms = Vec::with_capacity(k);
        let mut ss = Vec::with_capacity(k);
        for i in 0..k {
            ms.push(1.0 + self.tier_mult * i as f64);
            ss.push(self.base_size * (1.0 + self.size_mult * i as f64));
        }
        (ms, ss)
    }

    /// Inventory taper for one side: `1` at flat inventory, `0` at the
    /// bound. Long inventory tapers the BID side and boosts the ask;
    /// short inventory mirrors.
    pub fn taper(&self, inventory: i64, q_max: i64, is_bid: bool) -> f64 {
        let qm = q_max.max(1) as f64;
        let q = inventory as f64;
        let frac = if is_bid { 1.0 - q / qm } else { 1.0 + q / qm };
        frac.clamp(0.0, 1.6)
    }

    /// Per-level quote distances (price units) around the mid at
    /// inventory `q`, using the GLFT base distances as level-0.
    pub fn distances(&self, glft: &GlftAsymptotic, q: i64) -> (Vec<f64>, Vec<f64>) {
        let (ms, _) = self.tiers();
        let base_b = glft.delta_bid(q).max(0.0);
        let base_a = glft.delta_ask(q).max(0.0);
        let bid: Vec<f64> = ms.iter().map(|&m| base_b * m).collect();
        let ask: Vec<f64> = ms.iter().map(|&m| base_a * m).collect();
        (bid, ask)
    }

    /// Net expected edge rate of the ladder (price·lots/sec) under
    /// intensity `a·e^{−κ·δ}`:
    /// per level `i`, a fill nets
    /// `s_i·d_i·(1 − α·m_i) − γσ²·s_i²·hold/2` — the first factor is the
    /// post-adverse-selection edge (deep tiers are increasingly toxic:
    /// a fill at `d_i` means the mid already traded through `d_i`, and
    /// the post-fill drift scales with the full distance); the second
    /// is the running inventory cost of a chunky fill of size `s_i`
    /// held for `hold` seconds. Levels with negative net edge are not
    /// placed (contributions are clamped at zero).
    pub fn edge_rate(
        &self,
        base_half: f64,
        kappa: f64,
        a: f64,
        gamma: f64,
        sigma: f64,
        hold_secs: f64,
    ) -> f64 {
        let (ms, ss) = self.tiers();
        let mut rate = 0.0;
        for (&m, &s) in ms.iter().zip(ss.iter()) {
            let d = base_half * m;
            let adverse = self.adverse_alpha * m;
            let risk = 0.5 * gamma * sigma * sigma * s * s * hold_secs;
            let net = s * d * (1.0 - adverse) - risk;
            if net <= 0.0 {
                continue;
            }
            rate += a * (-kappa * d).exp() * net;
        }
        rate
    }

    /// Single-level edge rate with the same total size at the base
    /// distance (the GLFT benchmark): one chunky fill of the full size.
    pub fn single_edge_rate(
        &self,
        base_half: f64,
        kappa: f64,
        a: f64,
        gamma: f64,
        sigma: f64,
        hold_secs: f64,
    ) -> f64 {
        let (_, ss) = self.tiers();
        let total: f64 = ss.iter().sum();
        let adverse = self.adverse_alpha;
        let risk = 0.5 * gamma * sigma * sigma * total * total * hold_secs;
        let net = total * base_half * (1.0 - adverse) - risk;
        if net <= 0.0 {
            return 0.0;
        }
        a * (-kappa * base_half).exp() * net
    }

    /// Choose the number of levels that maximizes the net edge rate
    /// (search over 1..=max_levels), holding the policy's tier shape.
    pub fn optimal_levels(
        &self,
        base_half: f64,
        kappa: f64,
        a: f64,
        gamma: f64,
        sigma: f64,
        hold_secs: f64,
        max_levels: usize,
    ) -> usize {
        let mut best = 1;
        let mut best_rate = f64::NEG_INFINITY;
        for k in 1..=max_levels.max(1) {
            let cand = LadderPolicy { levels: k, ..self.clone() };
            let r = cand.edge_rate(base_half, kappa, a, gamma, sigma, hold_secs);
            if r > best_rate {
                best_rate = r;
                best = k;
            }
        }
        best
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn glft() -> GlftAsymptotic {
        GlftAsymptotic::new(0.1, 2.0, 1.5, 1.5)
    }

    #[test]
    fn tiers_are_monotone() {
        let p = LadderPolicy::default();
        let (ms, ss) = p.tiers();
        for w in ms.windows(2) {
            assert!(w[1] > w[0]);
        }
        for w in ss.windows(2) {
            assert!(w[1] > w[0]);
        }
        assert_eq!(ms.len(), p.levels);
    }

    #[test]
    fn distances_grow_with_inventory_on_the_risky_side() {
        let p = LadderPolicy::default();
        let g = glft();
        let (b0, _) = p.distances(&g, 0);
        let (b5, a5) = p.distances(&g, 5);
        // long inventory: bid distances grow, ask distances shrink
        assert!(b5[0] > b0[0]);
        assert!(a5[0] < p.distances(&g, 0).1[0]);
        // every level i is wider than level i-1
        for w in b5.windows(2) {
            assert!(w[1] > w[0]);
        }
    }

    #[test]
    fn taper_gates_inventory() {
        let p = LadderPolicy::default();
        assert!((p.taper(0, 20, true) - 1.0).abs() < 1e-12);
        assert!(p.taper(20, 20, true) <= 0.0);          // at the bound: no bid
        assert!(p.taper(20, 20, false) > 1.0);          // ask boosted
        assert!(p.taper(-20, 20, true) > 1.0);          // bid boosted when short
        assert!(p.taper(-20, 20, false) <= 0.0);
    }

    #[test]
    fn heavy_adverse_selection_prefers_fewer_levels() {
        // With alpha = 0 the ladder's extra intensity outweighs the
        // chunky-fill risk (more levels); with alpha large the deep
        // tiers are toxic (fewer levels).
        let (base, kappa, a) = (0.6f64, 1.5f64, 1.5f64);
        let (gamma, sigma, hold) = (0.1f64, 0.5f64, 1.0f64);
        let clean = LadderPolicy { adverse_alpha: 0.0, ..Default::default() };
        let toxic = LadderPolicy { adverse_alpha: 0.8, ..Default::default() };
        assert!(clean.optimal_levels(base, kappa, a, gamma, sigma, hold, 5) > 1, "clean regime should ladder");
        assert_eq!(toxic.optimal_levels(base, kappa, a, gamma, sigma, hold, 5), 1, "toxic regime should not");
    }

    #[test]
    fn edge_rate_matches_manual_sum() {
        let p = LadderPolicy::default();
        let (base, kappa, a) = (0.5f64, 1.5f64, 1.5f64);
        let (gamma, sigma, hold) = (0.1f64, 0.5f64, 1.0f64);
        let (ms, ss) = p.tiers();
        let mut manual = 0.0;
        for (&m, &s) in ms.iter().zip(ss.iter()) {
            let d = base * m;
            let adverse = p.adverse_alpha * m;
            let risk = 0.5 * gamma * sigma * sigma * s * s * hold;
            let net = s * d * (1.0 - adverse) - risk;
            if net > 0.0 {
                manual += a * (-kappa * d).exp() * net;
            }
        }
        assert!((p.edge_rate(base, kappa, a, gamma, sigma, hold) - manual).abs() < 1e-12);
    }

    #[test]
    fn ladder_beats_single_when_clean_but_not_when_toxic() {
        // In a clean regime (no deep-tier toxicity), splitting one chunky
        // quote into tiers raises the net edge rate (smaller fill chunks,
        // extra intensity). In a toxic regime the ladder loses its edge.
        let (base, kappa, a) = (0.6f64, 1.5f64, 1.5f64);
        let (gamma, sigma, hold) = (0.1f64, 0.5f64, 1.0f64);
        let clean = LadderPolicy { adverse_alpha: 0.0, ..Default::default() };
        assert!(clean.edge_rate(base, kappa, a, gamma, sigma, hold) > clean.single_edge_rate(base, kappa, a, gamma, sigma, hold));
        let toxic = LadderPolicy { adverse_alpha: 0.9, ..Default::default() };
        // At alpha = 0.9 even level 0 barely pays; the ladder's deep
        // tiers add nothing — both collapse toward the single quote.
        assert!(toxic.edge_rate(base, kappa, a, gamma, sigma, hold) <= clean.edge_rate(base, kappa, a, gamma, sigma, hold));
    }
}
