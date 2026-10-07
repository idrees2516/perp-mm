//! Composite Liquidity Factor (arXiv:2507.05749v1,
//! "High Frequency Quoting Under Liquidity Constraints").
//!
//! Direction-sensitive log-slope of the top-of-book ladder:
//! ```text
//! bid side: CLF^b_i = log(p^b_1 / p^b_{i+1}) / log(sum_{j<=i} q^b_j)
//! ask side: CLF^a_i = log(p^a_{i+1} / p^a_1) / log(sum_{j<=i} q^a_j)
//! ```
//! A **lower CLF = flatter depth profile = better liquidity** (smaller
//! expected slippage when the leg is used as the quoting reference). The
//! paper uses it to choose which leg of a two-contract roll to quote
//! passively; we also expose it as a liquidity gate for the risk engine.
//!
//! Input ladders are (price_ticks, lots) best-first: bids descending,
//! asks ascending.

/// Average CLF over the first `nu` levels of a best-first ladder.
/// Returns `None` when the ladder is too shallow to measure.
pub fn clf_score(ladder: &[(u64, u64)], nu: usize) -> Option<f64> {
    if ladder.len() < 2 {
        return None;
    }
    let nu = nu.min(ladder.len());
    let p0 = ladder[0].0 as f64;
    if p0 <= 0.0 {
        return None;
    }
    let mut vals = Vec::new();
    let mut cum_q = 0.0f64;
    for i in 0..nu.saturating_sub(1) {
        let (_pi, qi) = ladder[i];
        let p_next = ladder[i + 1].0;
        cum_q += qi as f64;
        if cum_q <= 1.0 {
            continue;
        }
        // bid ladders descend, ask ladders ascend: distance from the best
        let dist = if p_next < ladder[i].0 {
            (ladder[0].0 as f64 / p_next as f64).ln()
        } else if p_next > ladder[i].0 {
            (p_next as f64 / ladder[0].0 as f64).ln()
        } else {
            continue;
        };
        if dist > 0.0 {
            vals.push(dist / cum_q.ln());
        }
    }
    if vals.is_empty() {
        return None;
    }
    Some(vals.iter().sum::<f64>() / vals.len() as f64)
}

/// Choose the more liquid leg given two ladders (0 = first, 1 = second).
/// Ties break to 0. `None` if both unmeasurable.
pub fn choose_reference_leg(
    ladder_a: &[(u64, u64)],
    ladder_b: &[(u64, u64)],
    nu: usize,
) -> Option<usize> {
    let (a, b) = (clf_score(ladder_a, nu), clf_score(ladder_b, nu));
    match (a, b) {
        (Some(a), Some(b)) => Some(if a <= b { 0 } else { 1 }),
        (Some(_), None) => Some(0),
        (None, Some(_)) => Some(1),
        (None, None) => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flat_book_is_more_liquid() {
        // Two ask ladders at the same best price: one deep & flat, one thin.
        let flat: Vec<(u64, u64)> = vec![(100, 500), (101, 500), (102, 500), (103, 500)];
        let thin: Vec<(u64, u64)> = vec![(100, 5), (101, 5), (102, 5), (103, 5)];
        let clf_flat = clf_score(&flat, 4).unwrap();
        let clf_thin = clf_score(&thin, 4).unwrap();
        assert!(clf_flat < clf_thin, "{clf_flat} vs {clf_thin}");
        assert_eq!(choose_reference_leg(&flat, &thin, 4), Some(0));
        assert_eq!(choose_reference_leg(&thin, &flat, 4), Some(1));
    }

    #[test]
    fn bid_ladder_works_too() {
        let bids: Vec<(u64, u64)> = vec![(100, 100), (99, 100), (98, 100)];
        let s = clf_score(&bids, 3).unwrap();
        assert!(s > 0.0 && s.is_finite());
    }

    #[test]
    fn too_shallow_is_none() {
        let one: Vec<(u64, u64)> = vec![(100, 10)];
        assert_eq!(clf_score(&one, 3), None);
    }
}
