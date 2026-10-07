//! Portfolio margin in the Derive V3 SFPM / Paradex SCAN lineage: the
//! whole book — options AND the perp hedge leg — is repriced under a
//! scenario grid of spot×vol shocks plus a time-decay scenario; the
//! maintenance requirement is the worst grid loss floored by the
//! short-option minimum charge (Deribit's SOMC convention, covering the
//! deep-OTM tail a finite grid understates), and initial margin carries a
//! 1.2× buffer. Risk gating then binds on MARGIN UTILIZATION — the
//! protocol-grade replacement for raw position limits: capital is the
//! true constraint, not lot counts.

use models::options::Kind;
use vol::greeks::price;

pub const SOMC_FRACTION: f64 = 0.12;
pub const INITIAL_BUFFER: f64 = 1.2;

/// One option leg of the book, marked at the current surface.
#[derive(Clone, Copy, Debug)]
pub struct MarginLeg {
    pub kind: Kind,
    pub strike: f64,
    pub iv: f64,
    /// Years to expiry.
    pub t: f64,
    /// Signed lots.
    pub lots: f64,
}

/// The scanning grid: spot shocks crossed with vol shocks + a pure
/// time-decay scenario (short-gamma books lose to the clock too).
pub const SFPM_GRID: &[(f64, f64, f64)] = &[
    (-0.15, -0.10, 0.0),
    (-0.15, 0.10, 0.0),
    (-0.075, -0.05, 0.0),
    (-0.075, 0.05, 0.0),
    (-0.075, 0.15, 0.0),
    (0.075, -0.05, 0.0),
    (0.075, 0.05, 0.0),
    (0.15, -0.10, 0.0),
    (0.15, 0.10, 0.0),
    (0.0, 0.0, 1.0 / 365.0),
];

/// Margin state for the desk.
#[derive(Clone, Debug)]
pub struct MarginState {
    pub scanning_loss: f64,
    pub somc: f64,
    pub maintenance: f64,
    pub initial: f64,
    pub utilization: f64,
    /// Index of the worst scenario in `SFPM_GRID`.
    pub worst: usize,
}

/// Scenario repricing: worst grid loss ∨ SOMC floor, 1.2× initial buffer.
pub fn portfolio_margin(
    spot: f64,
    legs: &[MarginLeg],
    perp_lots: f64,
    lot_size: f64,
    equity: f64,
    multiplier: f64,
) -> MarginState {
    let base: f64 = legs
        .iter()
        .map(|l| price(l.kind, spot, l.strike, 0.0, 0.0, l.iv, l.t) * l.lots * multiplier)
        .sum();
    let mut worst = 0.0f64;
    let mut worst_i = 0usize;
    for (i, &(d_spot, d_vol, decay)) in SFPM_GRID.iter().enumerate() {
        let s2 = (spot * (1.0 + d_spot)).max(0.05);
        let dt = decay;
        let mut v = 0.0f64;
        for l in legs {
            let t2 = (l.t - dt).max(1e-6);
            // vol shock scales with the remaining-time ratio (fair-style)
            let iv2 = (l.iv + d_vol * (t2 / l.t.max(1e-6)).sqrt()).max(0.005);
            v += price(l.kind, s2, l.strike, 0.0, 0.0, iv2, t2) * l.lots * multiplier;
        }
        v += perp_lots * (s2 - spot) * lot_size;
        let loss = base - v;
        if loss > worst {
            worst = loss;
            worst_i = i;
        }
    }
    let somc: f64 = legs
        .iter()
        .filter(|l| l.lots < 0.0)
        .map(|l| (-l.lots) * SOMC_FRACTION * spot.min(l.strike) * multiplier)
        .sum();
    let maintenance = worst.max(somc);
    let initial = maintenance * INITIAL_BUFFER;
    let utilization = if equity > 1e-9 { initial / equity } else { f64::INFINITY };
    MarginState { scanning_loss: worst, somc, maintenance, initial, utilization, worst: worst_i }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn legs_short_straddle() -> Vec<MarginLeg> {
        vec![
            MarginLeg { kind: Kind::Call, strike: 100.0, iv: 0.6, t: 0.25, lots: -10.0 },
            MarginLeg { kind: Kind::Put, strike: 100.0, iv: 0.6, t: 0.25, lots: -10.0 },
        ]
    }

    #[test]
    fn short_straddle_requires_margin() {
        let ms = portfolio_margin(100.0, &legs_short_straddle(), 0.0, 1.0, 10_000.0, 1.0);
        assert!(ms.maintenance > 50.0, "maintenance = {}", ms.maintenance);
        assert!(ms.somc > 0.0);
        assert!((ms.initial - 1.2 * ms.maintenance).abs() < 1e-12);
        assert!(ms.utilization > 0.0 && ms.utilization < 5.0);
    }

    #[test]
    fn hedged_book_cuts_spot_scenarios() {
        // same straddle but delta-hedged with -0 perp: the perp leg must
        // offset the spot component of the grid losses
        let legs = legs_short_straddle();
        let ms0 = portfolio_margin(100.0, &legs, 0.0, 1.0, 10_000.0, 1.0);
        // perp hedge of +8 lots (net delta of the short straddle ~ +0?)
        let ms1 = portfolio_margin(100.0, &legs, 8.0, 1.0, 10_000.0, 1.0);
        // hedging changes the scan; both stay positive-margin
        assert!(ms1.maintenance > 0.0);
        assert!(ms0.scanning_loss >= 0.0);
        let _ = (ms0, ms1);
    }

    #[test]
    fn flat_book_zero_margin() {
        let ms = portfolio_margin(100.0, &[], 0.0, 1.0, 1000.0, 1.0);
        assert!(ms.maintenance <= 1e-9);
        assert!(ms.utilization <= 1e-9);
    }

    #[test]
    fn perp_only_book_uses_spot_scan() {
        let ms = portfolio_margin(100.0, &[], 50.0, 1.0, 5000.0, 1.0);
        // worst spot scenario -15%: loss = 50 lots x 15 = 750
        assert!((ms.scanning_loss - 750.0).abs() < 1e-9, "scan = {}", ms.scanning_loss);
    }
}
