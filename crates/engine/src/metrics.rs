//! PnL attribution, summary statistics, and CSV output.

/// Per-run metrics.
#[derive(Clone, Debug, Default)]
pub struct RunMetrics {
    pub strategy: &'static str,
    /// Final mark equity (quote units).
    pub equity: f64,
    /// Components.
    pub spread_capture: f64,
    pub adverse_cost: f64,
    pub fees_paid: f64,
    pub funding_paid: f64,
    /// Inventory path stats.
    pub inventory_final: i64,
    pub inventory_abs_mean: f64,
    pub inventory_abs_max: i64,
    /// Fills.
    pub our_fills: u64,
    /// Aggressive (taker) executions — hedges and manual takes.
    pub taker_fills: u64,
    /// Hedge cost (slippage vs mid + taker fees).
    pub hedge_cost: f64,
    /// Option book mark-to-market at the end (SSVI surface).
    pub option_mark: f64,
    /// Drawdown.
    pub max_drawdown: f64,
    /// Per-step equity series (subsampled).
    pub equity_path: Vec<f64>,
}

impl RunMetrics {
    /// Annualized-free Sharpe over the run (per-step, scaled to
    /// per-hour).
    pub fn sharpe_per_hour(&self) -> f64 {
        if self.equity_path.len() < 8 {
            return 0.0;
        }
        let rets: Vec<f64> = self
            .equity_path
            .windows(2)
            .map(|w| w[1] - w[0])
            .collect();
        let n = rets.len() as f64;
        let mean = rets.iter().sum::<f64>() / n;
        let var = rets.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / n.max(1.0);
        if var <= 0.0 {
            return 0.0;
        }
        mean / var.sqrt() * (3600.0f64).sqrt() * (60.0f64).sqrt()
    }

    /// CSV header line.
    pub fn csv_header() -> String {
        "strategy,equity,spread_capture,adverse_cost,fees,funding,inv_final,inv_abs_mean,inv_abs_max,fills,taker_fills,hedge_cost,option_mark,max_drawdown,sharpe_per_hour\n".into()
    }

    /// CSV row.
    pub fn csv_row(&self) -> String {
        format!(
            "{},{:.4},{:.4},{:.4},{:.4},{:.4},{},{:.4},{},{},{},{:.4},{:.4},{:.4},{:.4}\n",
            self.strategy,
            self.equity,
            self.spread_capture,
            self.adverse_cost,
            self.fees_paid,
            self.funding_paid,
            self.inventory_final,
            self.inventory_abs_mean,
            self.inventory_abs_max,
            self.our_fills,
            self.taker_fills,
            self.hedge_cost,
            self.option_mark,
            self.max_drawdown,
            self.sharpe_per_hour()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn csv_format() {
        let m = RunMetrics {
            strategy: "test",
            equity: 12.5,
            spread_capture: 30.0,
            adverse_cost: 10.0,
            fees_paid: 2.0,
            funding_paid: 1.0,
            inventory_final: 3,
            inventory_abs_mean: 1.2,
            inventory_abs_max: 7,
            our_fills: 42,
            taker_fills: 3,
            hedge_cost: 1.5,
            option_mark: 0.0,
            max_drawdown: 5.0,
            equity_path: vec![0.0; 10],
        };
        let row = m.csv_row();
        assert!(row.starts_with("test,12.5000,"));
        assert!(row.trim_end().split(',').count() == 15);
    }
}
