//! Adversarial inputs from the numerical and compilation review. Run potential hangs and
//! stack aborts in a subprocess so a regression fails with a bounded diagnostic.

use dataorder::{ErrorKind, MAX_DEPTH, Order, Sampling, Seq, Source};
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
fn public_validation_and_disposal_reject_deep_trees_safely() {
    isolated("dispose");
}

#[test]
fn isolated_case() {
    let Ok(case) = std::env::var("DATAORDER_REGRESSION_CASE") else { return };
    match case.as_str() {
        "map" => std::thread::Builder::new()
            .stack_size(2 << 20)
            .spawn(|| {
                let deep = || (0..200_000).fold(Seq::source("later"), |s, _| s.take(1));
                // Success, an untouched deep sibling, and an already mapped deep sibling.
                deep().map(|_| 1usize).dispose();
                for seq in [Seq::concat([Seq::source("missing"), deep()]), Seq::concat([deep(), Seq::source("missing")])] {
                    let mut calls = Vec::new();
                    let result = seq.try_map(|name| {
                        calls.push(name);
                        if name == "missing" { Err(()) } else { Ok(1usize) }
                    });
                    assert!(result.is_err());
                    assert_eq!(calls.last(), Some(&"missing"));
                    assert!(calls.len() <= 2);
                }
                let seq = Seq::concat([Seq::source("missing"), deep()]);
                assert!(std::panic::catch_unwind(|| seq.map::<usize, _>(|_| panic!("mapping failed"))).is_err());
            })
            .unwrap()
            .join()
            .unwrap(),
        "dispose" => std::thread::Builder::new()
            .stack_size(2 << 20)
            .spawn(|| {
                let deep = || (0..200_000).fold(Seq::source(1usize), |s, _| s.take(1));
                let seq = deep();
                assert_eq!(seq.check().unwrap_err().kind(), &ErrorKind::TooDeep);
                seq.dispose();
                assert_eq!(deep().validate().unwrap_err().kind(), &ErrorKind::TooDeep);
                assert!(deep().try_shard(0, 0).is_err());
                assert!(deep().try_slice(..=usize::MAX).is_err());
                assert_eq!(Seq::source(3).validate().unwrap(), Seq::source(3));
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
                assert_eq!(order.get(8191).1, 8191);
                assert_eq!(Order::new(chain(MAX_DEPTH + 1)).unwrap_err().kind(), &ErrorKind::TooDeep);
            })
            .unwrap()
            .join()
            .unwrap(),
        "profiles" => {
            for full in [f64::from_bits(1), 1e-309, 1e-308] {
                let seq = Seq::mix_with([(Seq::source(100), Sampling::Uniform), (Seq::source(100), Sampling::ramp(0.0, full))]);
                let error = Order::new(seq).unwrap_err();
                assert!(matches!(error.kind(), ErrorKind::InvalidSampling { .. }));
                assert_eq!(error.path(), &[1]);
            }
            for d in [1e-8, 1e-12, 1e-16, 1e-20, 1e-300] {
                let order = Order::new(Seq::mix_with([
                    (Seq::source(300), Sampling::Uniform),
                    (Seq::source(100), Sampling::trapezoid(0.0, d, d, 1.0)),
                ]))
                .unwrap();
                let all: Vec<_> = order.iter(..).map(|(&s, i)| (s, i)).collect();
                for p in 0..order.len() {
                    assert_eq!(
                        all[p],
                        {
                            let (&s, i) = order.get(p);
                            (s, i)
                        },
                        "d={d}, p={p}"
                    );
                    let drawn = all[..p].iter().filter(|&&(s, _)| s == 100).count();
                    let t = p as f64 / order.len() as f64;
                    assert!((drawn as f64 - 100.0 * (2.0 * t - t * t)).abs() < 2.0, "d={d}, p={p}, drawn={drawn}");
                }
            }
            #[cfg(target_pointer_width = "64")]
            {
                let seq =
                    Seq::mix_with([(Seq::source(1usize), Sampling::Uniform), (Seq::source((1 << 46) - 1), Sampling::ramp(0.0, 1e-296))]);
                let error = Order::new(seq).unwrap_err();
                assert_eq!(error.kind(), &ErrorKind::SamplingOverflow);
                assert!(error.path().is_empty());
                // Accepted overcommit tolerance can move the analytic rank by thousands
                // of positions. The old correction repeatedly guessed the same progress.
                let n = 1usize << 44;
                let order =
                    Order::new(Seq::mix_with([(Seq::source(1), Sampling::Uniform), (Seq::source(n - 1), Sampling::until(1.0 - 5e-10))]))
                        .unwrap();
                for p in n - 4000..n - 3980 {
                    assert_eq!(order.get(p), (&(n - 1), p));
                }
                assert_eq!(order.get(n - 1), (&1, 0));
            }
            #[cfg(target_pointer_width = "64")]
            for n in [1_000_000_000_000usize, (1 << 46) - 1] {
                let order =
                    Order::new(Seq::mix_with([(Seq::source(n), Sampling::Uniform), (Seq::source(1), Sampling::fading(0.0, 1.0))])).unwrap();
                for start in [0, n / 4, n / 2 - 16, 3 * (n / 4), n - 16] {
                    for (p, (&s, i)) in (start..).zip(order.iter(start..).take(16)) {
                        assert_eq!((s, i), {
                            let (&s, i) = order.get(p);
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
fn nested_slice_boundaries_remove_unreachable_salts() {
    let s = |salt| Seq::source(Named { salt, len: 10 });
    let make = |removed| Seq::concat([Seq::concat([s(removed), s(1), s(2)]).skip(1), s(3)]).skip(15);
    let tail = |removed| Seq::concat([s(1), Seq::concat([s(2), s(3), s(removed)]).take(29)]).take(25);
    for (a, b) in [(make(0), make(999)), (tail(0), tail(999))] {
        let a = Order::new(a.shuffle(1)).unwrap();
        let b = Order::new(b.shuffle(1)).unwrap();
        assert!(a.iter(..).map(|(s, i)| (s.salt, i)).eq(b.iter(..).map(|(s, i)| (s.salt, i))));
        assert_eq!(a.sources().len(), 4);
    }
}

#[test]
fn sharding_parts_requires_worker_schedule_feasibility() {
    let global = Seq::mix_with([(Seq::source(3), Sampling::Uniform), (Seq::source(1), Sampling::delayed(0.75))]);
    assert_eq!(global.check(), Ok(4));
    let shard = Seq::mix_with([(Seq::source(3).shard(2, 0), Sampling::Uniform), (Seq::source(1).shard(2, 0), Sampling::delayed(0.75))]);
    assert!(matches!(shard.check().unwrap_err().kind(), ErrorKind::Overcommitted { .. }));
    for worker in 0..2 {
        assert_eq!(global.clone().shard(2, worker).check(), Ok(2));
    }
}

#[test]
fn empty_compaction_preserves_identity_and_error_paths() {
    let order = Order::new(Seq::mix([Seq::source(0), Seq::source(3), Seq::source(0), Seq::source(3)])).unwrap();
    assert_eq!(order.sources(), &[0, 3, 0, 3]);
    assert_eq!(order.iter(..).map(|(s, _)| order.source_index(s)).collect::<Vec<_>>(), [1, 3, 1, 3, 1, 3]);
    let error = Seq::mix_with([
        (Seq::source(0), Sampling::Uniform),
        (Seq::source(3), Sampling::Uniform),
        (Seq::source(0), Sampling::delayed(f64::NAN)),
        (Seq::source(3), Sampling::Uniform),
    ])
    .check()
    .unwrap_err();
    assert_eq!(error.path(), &[2]);
    assert!(matches!(error.kind(), ErrorKind::InvalidSampling { .. }));
}

#[test]
#[cfg(feature = "serde")]
fn json_preserves_float_bits_and_large_weighted_orders() {
    let d = 0.18620199577071722;
    let seq = Seq::mix_with([(Seq::source(300usize), Sampling::Uniform), (Seq::source(100), Sampling::delayed(d))]);
    let back: Seq<usize> = serde_json::from_str(&serde_json::to_string(&seq).unwrap()).unwrap();
    assert_eq!(seq, back);
    #[cfg(target_pointer_width = "64")]
    {
        let n = 70_368_744_176_807;
        let seq = Seq::weighted(n, [(Seq::source(n), d), (Seq::source(n), 1.0 - d)]);
        let back: Seq<usize> = serde_json::from_str(&serde_json::to_string(&seq).unwrap()).unwrap();
        assert_eq!(seq, back);
        let (a, b) = (Order::new(seq).unwrap(), Order::new(back).unwrap());
        assert!(a.iter(n - 100..).map(|(s, i)| (a.source_index(s), i)).eq(b.iter(n - 100..).map(|(s, i)| (b.source_index(s), i))));
    }
}

fn weighted_counts(total: usize, weights: &[f64]) -> Vec<usize> {
    // Equal source values deliberately exercise identity through the public source index.
    // A zero-weight source is empty, so assigning it a share would also fail compilation.
    let order = Order::new(Seq::weighted(total, weights.iter().map(|&w| (Seq::source(usize::from(w > 0.0)), w)))).unwrap();
    assert_eq!(order.len(), total);
    let mut counts = vec![0; weights.len()];
    for (source, _) in order.iter(..) {
        counts[order.source_index(source)] += 1;
    }
    counts
}

fn integer_weight_counts(total: usize, weights: &[usize]) -> Vec<usize> {
    let sum: usize = weights.iter().sum();
    let mut counts: Vec<_> = weights.iter().map(|w| total * w / sum).collect();
    let mut remainders: Vec<_> = weights.iter().enumerate().map(|(i, w)| (i, total * w % sum)).collect();
    remainders.sort_by_key(|&(i, r)| (std::cmp::Reverse(r), i));
    let remaining = total - counts.iter().sum::<usize>();
    for &(i, _) in remainders.iter().take(remaining) {
        counts[i] += 1;
    }
    counts
}

#[test]
fn weighted_apportionment_preserves_exact_remainder_ties() {
    for (weights, expected) in [([1.0, 1.0, 7.0], [1, 0, 2]), ([1.0, 2.0, 12.0], [0, 1, 2]), ([7.0, 1.0, 1.0], [3, 0, 0])] {
        assert_eq!(weighted_counts(3, &weights), expected, "weights={weights:?}");
    }

    // A small integer oracle computes quotas and remainder comparisons without floats.
    for a in 1..=8 {
        for b in 1..=8 {
            for c in 1..=8 {
                let weights = [a, b, c];
                for total in 1..=20 {
                    assert_eq!(
                        weighted_counts(total, &weights.map(|w| w as f64)),
                        integer_weight_counts(total, &weights),
                        "weights={weights:?}, total={total}"
                    );
                }
            }
        }
    }
}

#[test]
fn weighted_apportionment_preserves_dyadic_scale() {
    for (weights, expected, high_exponent) in [([1.0, 1.0, 7.0], [1, 0, 2], 1021), ([1.0, 2.0, 12.0], [0, 1, 2], 1020)] {
        for scale in [f64::from_bits(1), f64::MIN_POSITIVE, 1.0, f64::from_bits((1023 + high_exponent) << 52)] {
            let scaled = weights.map(|w| w * scale);
            assert!(scaled.iter().all(|w| w.is_finite() && *w > 0.0));
            assert_eq!(weighted_counts(3, &scaled), expected, "weights={scaled:?}");
        }
    }
}

#[test]
fn weighted_apportionment_keeps_tiny_terms_that_break_ties() {
    // For [7s, s, s] and total 3, all remainders are exactly 1/3. Adding any of the
    // tiny positive weights below makes the big part's remainder smaller than either
    // unit part's. The first unit part must receive the last element, even when the
    // tiny term is thousands of binary places below the other weights.
    for scale in [1.0, f64::from_bits(2044 << 52)] {
        for tiny in [f64::from_bits(1), f64::MIN_POSITIVE, f64::from_bits((1023 - 128) << 52)] {
            for big in 0..4 {
                for small in (0..4).filter(|&i| i != big) {
                    let mut weights = [scale; 4];
                    weights[big] *= 7.0;
                    weights[small] = tiny;
                    let first_unit = (0..4).find(|&i| i != big && i != small).unwrap();
                    let mut expected = [0; 4];
                    expected[big] = 2;
                    expected[first_unit] = 1;
                    assert_eq!(weighted_counts(3, &weights), expected, "weights={weights:?}");
                }
            }
        }
    }
}

#[test]
fn zero_weight_parts_never_receive_an_element() {
    for weights in [[0, 7, 1, 1, 0], [1, 0, 2, 0, 12], [0, 0, 1, 0, 0]] {
        for scale in [f64::from_bits(1), 1.0, f64::from_bits(2043 << 52)] {
            let scaled = weights.map(|w| if w == 0 { -0.0 } else { w as f64 * scale });
            for total in 0..=32 {
                let counts = weighted_counts(total, &scaled);
                assert_eq!(counts, integer_weight_counts(total, &weights), "weights={scaled:?}, total={total}");
                assert!(weights.iter().zip(counts).all(|(&w, count)| w != 0 || count == 0));
            }
        }
    }
}

#[test]
fn mapping_and_error_cleanup_use_a_heap_stack() {
    isolated("map");
}
