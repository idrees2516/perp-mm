//! Solana-style compute-unit (CU) metering.
//!
//! Every hot-path operation class has a fixed CU cost (calibrated to the
//! measured ns costs on this machine; the ratios mirror the spirit of
//! Solana's per-instruction compute budget: cheap arithmetic ~1 CU,
//! memory/lookup ~tens, full reprice ~hundreds). Each feed event runs
//! under a [`Budget`]; exceeding it sets [`Budget::degraded`], and the
//! engine responds by skipping cold-path work (slow-cadence estimators,
//! optional analytics) — never by dropping the book update itself.

/// Operation classes with fixed CU costs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum OpClass {
    BookLevelUpdate,
    BookNewOrder,
    BookCancel,
    BestQuoteLookup,
    LadderScan5,
    CodecDecode,
    CodecEncode,
    EstimatorTick,
    EstimatorSlowCadence,
    QuoteCompute,
    RiskFilter,
    HjbInterp,
    SpreadEstimate,
    RoughVolRefresh,
}

impl OpClass {
    /// CU cost per operation (calibrated constants).
    pub fn cost(self) -> u64 {
        match self {
            OpClass::BookLevelUpdate => 3,
            OpClass::BookNewOrder => 5,
            OpClass::BookCancel => 5,
            OpClass::BestQuoteLookup => 1,
            OpClass::LadderScan5 => 12,
            OpClass::CodecDecode => 8,
            OpClass::CodecEncode => 9,
            OpClass::EstimatorTick => 14,
            OpClass::EstimatorSlowCadence => 120,
            OpClass::QuoteCompute => 25,
            OpClass::RiskFilter => 4,
            OpClass::HjbInterp => 6,
            OpClass::SpreadEstimate => 60,
            OpClass::RoughVolRefresh => 200,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            OpClass::BookLevelUpdate => "book.level_update",
            OpClass::BookNewOrder => "book.new_order",
            OpClass::BookCancel => "book.cancel",
            OpClass::BestQuoteLookup => "book.best_lookup",
            OpClass::LadderScan5 => "book.ladder_scan5",
            OpClass::CodecDecode => "codec.decode",
            OpClass::CodecEncode => "codec.encode",
            OpClass::EstimatorTick => "estimator.tick",
            OpClass::EstimatorSlowCadence => "estimator.slow",
            OpClass::QuoteCompute => "quote.compute",
            OpClass::RiskFilter => "risk.filter",
            OpClass::HjbInterp => "hjb.interp",
            OpClass::SpreadEstimate => "estimator.spread",
            OpClass::RoughVolRefresh => "estimator.rough",
        }
    }
}

/// A per-event compute budget (Solana-style cap + degradation flag).
#[derive(Clone, Debug)]
pub struct Budget {
    pub limit: u64,
    pub consumed: u64,
    pub degraded: bool,
    /// Total events metered (for the report).
    pub events: u64,
    /// Histogram of consumed-per-event (index = consumed/16 bucket).
    pub hist: Vec<u64>,
}

impl Budget {
    /// New budget with a per-event limit (Solana's default transaction
    /// cap is 1.2M CU; our feed events get a much tighter one).
    pub fn new(limit: u64) -> Budget {
        Budget {
            limit,
            consumed: 0,
            degraded: false,
            events: 0,
            hist: vec![0; 16],
        }
    }

    /// Begin a new event (resets consumed/degraded).
    pub fn begin_event(&mut self) {
        self.consumed = 0;
        self.degraded = false;
    }

    /// Charge for an operation. Returns false if the budget was already
    /// exceeded (the caller should skip optional work).
    pub fn charge(&mut self, class: OpClass) -> bool {
        self.consumed += class.cost();
        if self.consumed > self.limit {
            self.degraded = true;
            false
        } else {
            true
        }
    }

    /// End the event (records the histogram bucket).
    pub fn end_event(&mut self) {
        self.events += 1;
        let bucket = (self.consumed / 16).min(self.hist.len() as u64 - 1) as usize;
        self.hist[bucket] += 1;
    }

    /// Remaining CU in this event.
    pub fn remaining(&self) -> u64 {
        self.limit.saturating_sub(self.consumed)
    }

    /// Charge one operation (convenience).
    pub fn op(&mut self, class: OpClass) -> bool {
        self.charge(class)
    }
}

/// A scoped meter: charges op classes to a shared budget.
pub struct ComputeMeter<'a> {
    pub budget: &'a mut Budget,
}

impl<'a> ComputeMeter<'a> {
    pub fn new(budget: &'a mut Budget) -> ComputeMeter<'a> {
        ComputeMeter { budget }
    }

    /// Charge one operation; returns false when over budget (caller
    /// should degrade gracefully).
    pub fn op(&mut self, class: OpClass) -> bool {
        self.budget.charge(class)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn budget_accounting_and_degradation() {
        let mut b = Budget::new(30);
        b.begin_event();
        assert!(b.op(OpClass::BookLevelUpdate)); // 3
        assert!(b.op(OpClass::CodecDecode)); // 8 -> 11
        assert!(b.op(OpClass::EstimatorTick)); // 14 -> 25
        assert!(!b.op(OpClass::QuoteCompute)); // 25+25 > 30 -> degraded
        assert!(b.degraded);
        assert_eq!(b.consumed, 3 + 8 + 14 + 25);
        b.end_event();
        assert_eq!(b.events, 1);
        assert_eq!(b.hist[3], 1); // 50 CU -> bucket 50/16 = 3
    }

    #[test]
    fn costs_are_ordered_sensibly() {
        assert!(OpClass::BestQuoteLookup.cost() < OpClass::BookLevelUpdate.cost());
        assert!(OpClass::BookLevelUpdate.cost() < OpClass::QuoteCompute.cost());
        assert!(OpClass::QuoteCompute.cost() < OpClass::RoughVolRefresh.cost());
    }
}
