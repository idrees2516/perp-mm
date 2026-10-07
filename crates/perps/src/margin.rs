//! Isolated margin accounting (per-position; the venue's SFPM portfolio
//! margin lives in the engine crate where options repricing is
//! available).

/// An isolated-margin account for one position.
#[derive(Clone, Debug)]
pub struct IsolatedMargin {
    /// Collateral (quote).
    pub collateral: f64,
    /// Signed quantity (lots).
    pub qty: i64,
    /// Entry price (quote per lot).
    pub entry_price: f64,
    /// Maintenance-margin rate (e.g. 375bps = 0.0375).
    pub mm_rate: f64,
}

/// Snapshot of an isolated account's margin state at a mark.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MarginState {
    pub equity: f64,
    pub maintenance: f64,
    pub liquidatable: bool,
    /// `max(0, maintenance - equity)` — the deficit to socialize.
    pub deficit: f64,
    /// Margin ratio `equity / maintenance` (inf when maintenance is 0).
    pub margin_ratio: f64,
}

pub fn isolated_account(collateral: f64, qty: i64, entry: f64, mm_rate: f64) -> IsolatedMargin {
    IsolatedMargin {
        collateral,
        qty,
        entry_price: entry,
        mm_rate,
    }
}

impl IsolatedMargin {
    pub fn state_at(&self, mark: f64) -> MarginState {
        let equity = self.collateral + (mark - self.entry_price) * self.qty as f64;
        let maintenance = self.mm_rate * mark * self.qty.abs() as f64;
        let liquidatable = equity < maintenance;
        let deficit = (maintenance - equity).max(0.0);
        let margin_ratio = if maintenance > 0.0 {
            equity / maintenance
        } else {
            f64::INFINITY
        };
        MarginState {
            equity,
            maintenance,
            liquidatable,
            deficit,
            margin_ratio,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn liquidation_trigger_and_deficit() {
        // long 10 @ 100, collateral 60, mm 3.75%
        let acct = isolated_account(60.0, 10, 100.0, 0.0375);
        let ok = acct.state_at(100.0);
        assert!(!ok.liquidatable);
        assert_eq!(ok.deficit, 0.0);
        // price falls to 94: equity = 60 - 60 = 0 < mm = 0.0375*94*10
        let liq = acct.state_at(94.0);
        assert!(liq.liquidatable);
        assert!((liq.deficit - 35.25).abs() < 1e-9);
        // exact bankruptcy price: equity = 0 at mark = 94
        let bk = acct.state_at(94.0);
        assert!(bk.equity.abs() < 1e-9);
        // price down 10%: deeply negative equity
        let deep = acct.state_at(90.0);
        assert!(deep.equity < -35.0 && deep.liquidatable);
    }

    #[test]
    fn short_side_symmetry() {
        // short 10 @ 100, collateral 60: equity rises as price falls
        let acct = isolated_account(60.0, -10, 100.0, 0.0375);
        let up = acct.state_at(94.0);
        assert!(up.equity > 60.0 && !up.liquidatable);
        let dn = acct.state_at(107.0);
        assert!(dn.equity < 0.0 && dn.liquidatable);
    }
}
