//! Shared primitive types for the order-book crates.
//!
//! Conventions (matching the perp-options-clob venue engine):
//! * prices are `u64` **ticks** (multiply by `tick_size_quote_minor` for
//!   quote-minor units),
//! * quantities are `u64` **lots** (multiply by `lot_size_base_minor` for
//!   base-minor units),
//! * money (`u128` quote-minor) never appears inside this crate — the book
//!   is a pure integer tick/lot structure.

/// Side of the book / of an order.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Side {
    /// Buy / bid side.
    Bid,
    /// Sell / ask side.
    Ask,
}

impl Side {
    #[inline]
    pub fn opposite(self) -> Side {
        match self {
            Side::Bid => Side::Ask,
            Side::Ask => Side::Bid,
        }
    }

    /// +1 for Bid, -1 for Ask (sign convention used throughout the engine:
    /// inventory signed positive = long).
    #[inline]
    pub fn sign(self) -> i64 {
        match self {
            Side::Bid => 1,
            Side::Ask => -1,
        }
    }
}

/// Iterator over the best `n` levels of one side, produced by
/// [`crate::book::OrderBook`]. Bid side yields the highest price first,
/// ask side the lowest price first.
pub struct TickIter {
    pub(crate) levels: Vec<(u64, u64)>,
    pub(crate) side: Side,
}

impl TickIter {
    pub fn side(&self) -> Side {
        self.side
    }
}

impl Iterator for TickIter {
    type Item = (u64, u64);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.levels.is_empty() {
            return None;
        }
        Some(self.levels.remove(0))
    }
}

impl ExactSizeIterator for TickIter {
    fn len(&self) -> usize {
        self.levels.len()
    }
}
