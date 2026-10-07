use engine::config::EngineConfig;
use engine::mm::MarketMaker;
use engine::strategy::StrategyKind;
use micro::Rng;

#[test]
fn l3_invariants_hold_under_ladder_strategy() {
    let mut cfg = EngineConfig::default();
    cfg.horizon = 1e9;
    let mut mm = MarketMaker::new(cfg, StrategyKind::MultiLevel, 7);
    let mut rng = Rng::new(7 ^ 0x9E3779B9);
    for step in 0..3000 {
        mm.step(&mut rng);
        if let Some(v) = mm.venue.book.l3().invariant_violation() {
            panic!("step {}: {}", step, v);
        }
    }
}
