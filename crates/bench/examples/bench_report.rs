//! Benchmark report: book ops, codec, estimators, quote compute, the
//! full pipeline, and CU headroom.

use bench::cu::{Budget, OpClass};
use bench::timer::{measure_ns, percentiles, NsTimer};
use engine::config::EngineConfig;
use engine::estimator::EstimatorStack;
use engine::strategy::{build, QuoteCtx, StrategyKind};
use feed::codec::{decode_frame, encode_frame, Frame, MsgType, Payload};
use micro::Rng;
use ob::{BookEvent, OrderBook, Side};

fn row(name: &str, s: &bench::timer::Stats) -> String {
    format!(
        "{:<28} {:>10.0} {:>10.0} {:>10.0} {:>10.0}",
        name, s.p50, s.p90, s.p99, s.max
    )
}

fn main() {
    println!("perp-mm benchmark report");
    println!("========================\n");
    let cfg = EngineConfig::default();
    let mut rng = Rng::new(1234);

    // ---------- 1) glass book operations ----------
    println!("[ns] operation                    p50        p90        p99        max");
    {
        let mut book = OrderBook::new();
        for i in 0..2000u64 {
            book.apply(BookEvent::NewOrder {
                id: i,
                side: if i % 2 == 0 { Side::Bid } else { Side::Ask },
                price_ticks: 200 + (i % 40),
                lots: 10,
                ts_ns: i as i64,
            });
        }
        // level update (the hot path)
        let s = measure_ns(2000, 200_000, || {
            book.apply(BookEvent::LevelDelta {
                side: Side::Bid,
                price_ticks: 210,
                delta_lots: 1,
            });
        });
        println!("{}", row("book.level_update", &s));
        // best lookup
        let s = measure_ns(2000, 200_000, || {
            std::hint::black_box(book.best_bid());
        });
        println!("{}", row("book.best_lookup", &s));
        // ladder scan
        let s = measure_ns(500, 50_000, || {
            let l: Vec<(u64, u64)> = book.ladder(Side::Bid, 5).collect();
            std::hint::black_box(&l);
        });
        println!("{}", row("book.ladder_scan5", &s));
        // new order + cancel
        let mut id = 10_000_000u64;
        let s = measure_ns(500, 50_000, || {
            id += 1;
            book.apply(BookEvent::NewOrder {
                id,
                side: Side::Bid,
                price_ticks: 205,
                lots: 5,
                ts_ns: 0,
            });
        });
        println!("{}", row("book.new_order", &s));
        let s = measure_ns(500, 50_000, || {
            book.apply(BookEvent::CancelOrder { id });
            id += 1;
            book.apply(BookEvent::NewOrder {
                id,
                side: Side::Bid,
                price_ticks: 205,
                lots: 5,
                ts_ns: 0,
            });
        });
        println!("{}", row("book.cancel+readd", &s));
    }

    // ---------- 2) codec ----------
    {
        let frame = Frame {
            msg_type: MsgType::LevelDelta,
            seq: 42,
            ts_ns: 12345,
            payload: Payload::LevelDelta {
                side: Side::Bid,
                price_ticks: 199,
                delta_lots: -3,
            },
        };
        let mut buf = [0u8; 128];
        let n = encode_frame(&frame, &mut buf).unwrap();
        let s = measure_ns(2000, 200_000, || {
            let _ = encode_frame(&frame, &mut buf);
        });
        println!("{}", row("codec.encode (39B)", &s));
        let s = measure_ns(2000, 200_000, || {
            let _ = decode_frame(&buf[..n]);
        });
        println!("{}", row("codec.decode (39B)", &s));
        // throughput
        let bytes = 39.0;
        let ns = {
            let t = NsTimer::start();
            let mut sink = 0usize;
            for _ in 0..1_000_000 {
                if let Ok((_, m)) = decode_frame(&buf[..n]) {
                    sink += m;
                }
            }
            std::hint::black_box(sink);
            t.elapsed_ns() as f64 / 1_000_000.0
        };
        println!(
            "codec.decode throughput: {:.0} MB/s ({:.2} Mframes/s)\n",
            bytes / ns * 1000.0,
            1000.0 / ns
        );
    }

    // ---------- 3) estimator + strategy ----------
    {
        let mut est = EstimatorStack::new(&cfg);
        let mut mid = 100.0f64;
        for _ in 0..500 {
            mid *= (0.02 * rng.normal() * cfg.dt.sqrt()).exp();
            est.update(mid, Some((199, 40)), Some((201, 40)), cfg.tick_size);
        }
        let mut book = OrderBook::new();
        book.apply(BookEvent::NewOrder {
            id: 1,
            side: Side::Bid,
            price_ticks: 199,
            lots: 40,
            ts_ns: 0,
        });
        book.apply(BookEvent::NewOrder {
            id: 2,
            side: Side::Ask,
            price_ticks: 201,
            lots: 40,
            ts_ns: 0,
        });
        // estimator tick
        let s = measure_ns(200, 20_000, || {
            est.update(100.0 + 0.001, Some((199, 40)), Some((201, 40)), cfg.tick_size);
        });
        println!("{}", row("estimator.tick", &s));
        // snapshot
        let state = est.snapshot(&mut book, cfg.tick_size);
        let mut strat = build(StrategyKind::HjbPolicy, &cfg);
        let bid_ladder: Vec<(u64, u64)> = vec![(199, 40), (198, 30)];
        let ask_ladder: Vec<(u64, u64)> = vec![(201, 40), (202, 30)];
        let s = measure_ns(200, 20_000, || {
            let ctx = QuoteCtx {
                state: &state,
                inventory: 3,
                time_left: 600.0,
                bid_ladder: bid_ladder.clone(),
                ask_ladder: ask_ladder.clone(),
                fee_floor_ticks: 1.0,
                funding_rate: 0.0001,
                funding_interval: 8.0 * 3600.0,
                markout_mult: 1.0,
                options: None,
            };
            std::hint::black_box(strat.quotes(&ctx));
        });
        println!("{}", row("quote.compute (hjb)", &s));

        // full pipeline: decode -> book -> estimate -> quote -> encode
        let mut frame_buf = [0u8; 128];
        let frame = Frame {
            msg_type: MsgType::LevelDelta,
            seq: 43,
            ts_ns: 1,
            payload: Payload::LevelDelta {
                side: Side::Bid,
                price_ticks: 199,
                delta_lots: 2,
            },
        };
        let n = encode_frame(&frame, &mut frame_buf).unwrap();
        let mut book2 = book;
        let s = measure_ns(200, 20_000, || {
            // decode
            let (f, _) = decode_frame(&frame_buf[..n]).unwrap();
            // book
            if let Payload::LevelDelta {
                side,
                price_ticks,
                delta_lots,
            } = f.payload
            {
                book2.apply(BookEvent::LevelDelta {
                    side,
                    price_ticks,
                    delta_lots,
                });
            }
            // estimate
            est.update(100.0, book2.best_bid(), book2.best_ask(), cfg.tick_size);
            // quote
            let st = est.snapshot(&mut book2, cfg.tick_size);
            let ctx = QuoteCtx {
                state: &st,
                inventory: 0,
                time_left: 600.0,
                bid_ladder: bid_ladder.clone(),
                ask_ladder: ask_ladder.clone(),
                fee_floor_ticks: 1.0,
                funding_rate: 0.0001,
                funding_interval: 8.0 * 3600.0,
                markout_mult: 1.0,
                options: None,
            };
            let q = strat.quotes(&ctx);
            // encode
            let out = Frame {
                msg_type: MsgType::NewOrder,
                seq: 1,
                ts_ns: 2,
                payload: Payload::NewOrder {
                    order_id: 1,
                    side: Side::Bid,
                    price_ticks: q.bid.map(|(p, _)| p).unwrap_or(199),
                    lots: 5,
                },
            };
            let mut ob = [0u8; 128];
            let _ = encode_frame(&out, &mut ob);
        });
        println!("{}", row("pipeline (decode->encode)", &s));
    }

    // ---------- 4) CU metering ----------
    {
        println!("\nCU metering (per feed event, budget 200 CU):");
        let mut budget = Budget::new(200);
        // simulate a typical event: decode + 2 book updates + estimator +
        // quote + risk
        for _ in 0..10_000 {
            budget.begin_event();
            budget.charge(OpClass::CodecDecode);
            budget.charge(OpClass::BookLevelUpdate);
            budget.charge(OpClass::BookLevelUpdate);
            budget.charge(OpClass::EstimatorTick);
            budget.charge(OpClass::BestQuoteLookup);
            if budget.charge(OpClass::QuoteCompute) {
                budget.charge(OpClass::RiskFilter);
            }
            // cold path occasionally
            if rng.below(64) == 0 && budget.remaining() > 60 {
                budget.charge(OpClass::LadderScan5);
            }
            budget.end_event();
        }
        let per_event: Vec<f64> = (0..budget.hist.len())
            .map(|i| budget.hist[i] as f64)
            .collect();
        let st = percentiles(per_event.iter().flat_map(|&c| vec![c; 1]).collect());
        let _ = st;
        let total: u64 = budget.hist.iter().sum();
        println!("  events metered: {}", budget.events);
        println!("  histogram (16-CU buckets): {:?}", budget.hist);
        let degraded_estimate: f64 = (total as f64
            - budget.hist[0] as f64 * 0.0)
            .min(0.0);
        let _ = degraded_estimate;
        // events in the top bucket vs cap
        let top: u64 = budget.hist.iter().skip(12).sum();
        println!(
            "  events finishing above 192 CU: {:.2}%",
            top as f64 / budget.events as f64 * 100.0
        );
        let typical = OpClass::CodecDecode.cost()
            + 2 * OpClass::BookLevelUpdate.cost()
            + OpClass::EstimatorTick.cost()
            + OpClass::BestQuoteLookup.cost()
            + OpClass::QuoteCompute.cost()
            + OpClass::RiskFilter.cost();
        println!(
            "  typical event: {typical} CU / {} limit -> {:.0}% headroom",
            budget.limit,
            100.0 - typical as f64 / budget.limit as f64 * 100.0
        );
    }

    println!("\nrdtsc sanity: {} cycles", bench::timer::rdtsc());
}
