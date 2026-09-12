//! Adversarial inputs from the numerical and compilation review. Run potential hangs and
//! stack aborts in a subprocess so a regression fails with a bounded diagnostic.

use dataorder::{ErrorKind, MAX_DEPTH, Order, Schedule, Seq, Source};
use std::process::Command;
use std::time::{Duration, Instant};

fn isolated(case: &str) {
    let mut child = Command::new(std::env::current_exe().unwrap())
        .args(["--exact", "isolated_case", "--nocapture"])
        .env("DATAORDER_REGRESSION_CASE", case)
        .spawn()
        .unwrap();
    let deadline = Instant::now() + Duration::from_secs(15);
    loop {
        if let Some(status) = child.try_wait().unwrap() {
            assert!(status.success(), "{case}: subprocess {status}");
            return;
        }
        if Instant::now() >= deadline {
            child.kill().unwrap();
            child.wait().unwrap();
            panic!("{case}: subprocess exceeded 15 seconds");
        }
        std::thread::sleep(Duration::from_millis(10));
    }
}

#[test]
fn finite_profiles_and_seeks_terminate() {
    isolated("profiles");
}

#[test]
fn large_inline_sources_fit_a_thread_stack() {
    isolated("stack");
}

#[test]
fn order_rejects_configurations_over_the_depth_limit() {
    isolated("depth");
}

#[test]
fn isolated_case() {
    let Ok(case) = std::env::var("DATAORDER_REGRESSION_CASE") else { return };
    match case.as_str() {
        "map" => std::thread::Builder::new()
            .stack_size(2 << 20)
            .spawn(|| {
                let deep = || (0..MAX_DEPTH - 2).fold(Seq::source("later"), |s, _| s.take(1));
                // Success, an untouched deep sibling, and an already mapped deep sibling.
                drop(deep().map_sources(|_| 1usize));
                for seq in [Seq::concat([Seq::source("missing"), deep()]), Seq::concat([deep(), Seq::source("missing")])] {
                    let mut calls = Vec::new();
                    let result = seq.try_map_sources(|name| {
                        calls.push(name);
                        if name == "missing" { Err(()) } else { Ok(1usize) }
                    });
                    assert!(result.is_err());
                    assert_eq!(calls.last(), Some(&"missing"));
                    assert!(calls.len() <= 2);
                }
                let seq = Seq::concat([Seq::source("missing"), deep()]);
                assert!(std::panic::catch_unwind(|| seq.map_sources::<usize, _>(|_| panic!("mapping failed"))).is_err());
            })
            .unwrap()
            .join()
            .unwrap(),
        "depth" => std::thread::Builder::new()
            .stack_size(2 << 20)
            .spawn(|| {
                let deep = || (0..MAX_DEPTH).fold(Seq::source(1usize), |s, _| s.take(1));
                assert_eq!(Order::new(deep()).unwrap_err().kind(), &ErrorKind::TooDeep);
                assert!(Order::new(deep().step_by(0)).is_err());
                assert_eq!(Order::new(Seq::source(3)).unwrap().len(), 3);
            })
            .unwrap()
            .join()
            .unwrap(),
        "stack" => std::thread::Builder::new()
            .stack_size(2 << 20)
            .spawn(|| {
                let chain = |depth| (1..depth).fold(Seq::source([0u8; 8192]), |s, _| s.take(8192));
                let order = Order::new(chain(MAX_DEPTH)).unwrap();
                assert_eq!(order.len(), 8192);
                assert_eq!(order.get(8191).unwrap().record_index, 8191);
                assert_eq!(Order::new(chain(MAX_DEPTH + 1)).unwrap_err().kind(), &ErrorKind::TooDeep);
            })
            .unwrap()
            .join()
            .unwrap(),
        "profiles" => {
            for full in [f64::from_bits(1), 1e-309, 1e-308] {
                let seq = Seq::mix([(Seq::source(100), Schedule::Uniform), (Seq::source(100), Schedule::ramp(0.0, full))]);
                let error = Order::new(seq).unwrap_err();
                assert!(matches!(error.kind(), ErrorKind::InvalidSchedule { .. }));
                assert_eq!(error.path(), &[1]);
            }
            for d in [1e-8, 1e-12, 1e-16, 1e-20, 1e-300] {
                let order =
                    Order::new(Seq::mix([(Seq::source(300), Schedule::Uniform), (Seq::source(100), Schedule::trapezoid(0.0, d, d, 1.0))]))
                        .unwrap();
                let all: Vec<_> = order.iter().map(|item| (*item.source, item.record_index)).collect();
                for p in 0..order.len() {
                    assert_eq!(
                        all[p],
                        {
                            let dataorder::Item { source: &s, record_index: i, .. } = order.get(p).unwrap();
                            (s, i)
                        },
                        "d={d}, p={p}"
                    );
                    let drawn = all[..p].iter().filter(|&&(s, _)| s == 100).count();
                    // In the limiting fading shape: p = 300*t + 100*(2*t-t*t).
                    let t = 2.0 * p as f64 / (500.0 + (250_000.0 - 400.0 * p as f64).sqrt());
                    assert!((drawn as f64 - 100.0 * (2.0 * t - t * t)).abs() < 2.0, "d={d}, p={p}, drawn={drawn}");
                }
            }
            #[cfg(target_pointer_width = "64")]
            {
                // These profiles formerly overflowed the shared remainder builder.
                let n = (1usize << 46) - 1;
                let order =
                    Order::new(Seq::mix([(Seq::source(1), Schedule::Uniform), (Seq::source(n), Schedule::ramp(0.0, 1e-296))])).unwrap();
                let at = (n + 1) / 4;
                let window: Vec<_> = order.cursor(at - 8..at + 8).unwrap().map(|item| (*item.source, item.record_index)).collect();
                assert!(window.iter().any(|&(s, _)| s == 1));
                for (j, expected) in window.into_iter().enumerate() {
                    let dataorder::Item { source: &s, record_index: i, .. } = order.get(at - 8 + j).unwrap();
                    assert_eq!((s, i), expected);
                }
            }
            #[cfg(target_pointer_width = "64")]
            for n in [1_000_000_000_000usize, (1 << 46) - 1] {
                let order =
                    Order::new(Seq::mix([(Seq::source(n), Schedule::Uniform), (Seq::source(1), Schedule::fade(0.0, 1.0))])).unwrap();
                for start in [0, n / 4, n / 2 - 16, 3 * (n / 4), n - 16] {
                    for (p, dataorder::Item { source: &s, record_index: i, .. }) in (start..).zip(order.cursor(start..).unwrap().take(16)) {
                        assert_eq!((s, i), {
                            let dataorder::Item { source: &s, record_index: i, .. } = order.get(p).unwrap();
                            (s, i)
                        });
                        // The singleton's stagger is 3/4, whose fading quantile is 1/2.
                        if p < n / 2 - 2 {
                            assert_eq!((s, i), (n, p));
                        }
                        if p > n / 2 + 2 {
                            assert_eq!((s, i), (n, p - 1));
                        }
                    }
                }
            }
        }
        _ => panic!("unknown regression case"),
    }
}

#[derive(Clone, Debug)]
struct Named {
    salt: u64,
    len: usize,
}
impl Source for Named {
    fn len(&self) -> usize {
        self.len
    }
    fn salt(&self) -> u64 {
        self.salt
    }
}

#[test]
fn nested_slice_boundaries_preserve_configuration_salts() {
    let s = |salt| Seq::source(Named { salt, len: 10 });
    let make = |removed| Seq::concat([Seq::concat([s(removed), s(1), s(2)]).skip(1), s(3)]).skip(15);
    let tail = |removed| Seq::concat([s(1), Seq::concat([s(2), s(3), s(removed)]).take(29)]).take(25);
    for (a, b) in [(make(0), make(999)), (tail(0), tail(999))] {
        let a = Order::new(a.shuffle()).unwrap();
        let b = Order::new(b.shuffle()).unwrap();
        let mut a_items: Vec<_> = a.iter().map(|item| (item.source.salt, item.record_index)).collect();
        let mut b_items: Vec<_> = b.iter().map(|item| (item.source.salt, item.record_index)).collect();
        assert_ne!(a_items, b_items);
        a_items.sort_unstable();
        b_items.sort_unstable();
        assert_eq!(a_items, b_items);
        assert_eq!(a.sources().len(), 4);
    }
}

#[test]
fn sharding_parts_can_change_counts_without_schedule_capacity_errors() {
    let global = Seq::mix([(Seq::source(3), Schedule::Uniform), (Seq::source(1), Schedule::delayed(0.75))]);
    assert_eq!(Order::new(global.clone()).unwrap().len(), 4);
    let shard =
        Seq::mix([(Seq::source(3).skip(0).step_by(2), Schedule::Uniform), (Seq::source(1).skip(0).step_by(2), Schedule::delayed(0.75))]);
    assert_eq!(Order::new(shard).unwrap().len(), 3);
    for worker in 0..2 {
        assert_eq!(Order::new(global.clone().skip(worker).step_by(2)).unwrap().len(), 2);
    }
}

#[test]
fn empty_compaction_preserves_identity_and_error_paths() {
    let order = Order::new(Seq::mix([Seq::source(0), Seq::source(3), Seq::source(0), Seq::source(3)])).unwrap();
    assert_eq!(order.sources(), &[0, 3, 0, 3]);
    assert_eq!(order.iter().map(|item| item.source_ordinal).collect::<Vec<_>>(), [1, 3, 1, 3, 1, 3]);
    let error = Order::new(Seq::mix([
        (Seq::source(0), Schedule::Uniform),
        (Seq::source(3), Schedule::Uniform),
        (Seq::source(0), Schedule::delayed(f64::NAN)),
        (Seq::source(3), Schedule::Uniform),
    ]))
    .unwrap_err();
    assert_eq!(error.path(), &[2]);
    assert!(matches!(error.kind(), ErrorKind::InvalidSchedule { .. }));
}

#[test]
#[cfg(feature = "serde")]
fn json_preserves_float_bits_and_large_scheduled_orders() {
    let d = 0.18620199577071722;
    let seq = Seq::mix([(Seq::source(300usize), Schedule::Uniform), (Seq::source(100), Schedule::delayed(d))]);
    let back: Seq<usize> = serde_json::from_str(&serde_json::to_string(&seq).unwrap()).unwrap();
    assert_eq!(seq, back);
    #[cfg(target_pointer_width = "64")]
    {
        let n = 70_368_744_176_807;
        let seq = Seq::mix([(Seq::source(n - 100), Schedule::Uniform), (Seq::source(100), Schedule::delayed(d))]);
        let back: Seq<usize> = serde_json::from_str(&serde_json::to_string(&seq).unwrap()).unwrap();
        assert_eq!(seq, back);
        let (a, b) = (Order::new(seq).unwrap(), Order::new(back).unwrap());
        assert!(
            a.cursor(n - 100..)
                .unwrap()
                .map(|item| (item.source_ordinal, item.record_index))
                .eq(b.cursor(n - 100..).unwrap().map(|item| (item.source_ordinal, item.record_index)))
        );
    }
}

#[test]
fn mapping_stops_on_callback_errors_and_panics() {
    isolated("map");
}
