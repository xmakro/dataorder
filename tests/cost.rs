//! The cost model, pinned by counting allocations: seeks within an initialized mix
//! reuse its buffers, and a forward seek or `nth` across many repetitions or concat
//! parts lands in the target one instead of entering every one on the way.

use dataorder::{Order, Seq};
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::hint::black_box;

struct Counting;

thread_local! {
    /// Allocations by this thread, so that the tests can run in parallel.
    static ALLOCATIONS: Cell<usize> = const { Cell::new(0) };
    static BYTES: Cell<usize> = const { Cell::new(0) };
}

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // Unavailable only while the thread is being torn down.
        let _ = ALLOCATIONS.try_with(|c| c.set(c.get() + 1));
        let _ = BYTES.try_with(|c| c.set(c.get() + layout.size()));
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static ALLOCATOR: Counting = Counting;

fn allocations(f: impl FnOnce()) -> usize {
    let before = ALLOCATIONS.get();
    f();
    ALLOCATIONS.get() - before
}

fn element(order: &Order<usize>, pos: usize) -> (usize, usize) {
    let item = order.get(pos).unwrap();
    (*item.source, item.record_index)
}

#[test]
fn unary_traversals_do_not_allocate_temporary_child_lists() {
    let (parts, depth) = (128, dataorder::MAX_DEPTH as usize - 2);
    let make = || Seq::mix((0..parts).map(|_| (0..depth).fold(Seq::source(10), |seq, _| seq.take(10))));
    let seq = make();
    let count = allocations(|| {
        let order = Order::new(seq).unwrap();
        assert_eq!(order.len(), parts * 10);
    });
    // Identity transforms fold away; allow source/child vectors without a
    // temporary singleton child list for each unary node.
    assert!(count < 1000, "compilation allocated {count} times");
    let seq = make();
    let count = allocations(|| drop(seq.map(|n| n + 1)));
    // Mapping rebuilds each transform's box and preserves the tree.
    assert!(count < parts * depth + 1000, "mapping allocated {count} times");
}

#[test]
fn seeks_within_each_concat_child_reuse_mix_buffers() {
    use dataorder::Sampling;
    let part = |k, len, scheduled| {
        Seq::mix((0..k).map(|i| {
            (Seq::source(len).shuffle(i as u64 + 1), if scheduled && i % 5 == 0 { Sampling::ramp(0.1, 0.6) } else { Sampling::Uniform })
        }))
    };
    let order = Order::new(Seq::concat([part(1000, 10, false), part(700, 20, true)]).repeat(3)).unwrap();
    let epoch = order.len() / 3;
    let mut cursor = order.iter(..).unwrap();
    for start in [0, 10_000, epoch, epoch + 10_000] {
        cursor.seek(start).unwrap();
        cursor.next();
        let checks = [start + 50, start + 1, start + 49].map(|pos| (pos, order.get(pos)));
        let count = allocations(|| {
            for (pos, expected) in checks {
                cursor.seek(pos).unwrap();
                assert_eq!(cursor.next(), expected);
            }
        });
        assert_eq!(count, 0, "seeks within a concat child allocated {count} times");
    }
}

/// A mix builds a part's cursor when the part is first drawn from (nothing to allocate for
/// a shuffled source, whose cursor sits in the mix's slot for it); after that, seeks reuse
/// everything. Drawing a thousand elements first enters every part of a balanced mix of a
/// hundred, so the count below is of seeking alone.
#[test]
fn seeking_an_existing_cursor_allocates_nothing() {
    let order = Order::new(Seq::mix((0..100).map(|i| Seq::source(10_000).shuffle(i + 1)))).unwrap();
    let n = order.len();
    let mut cursor = order.iter(n / 2..).unwrap();
    cursor.by_ref().take(1000).for_each(drop);
    let count = allocations(|| {
        cursor.seek(n / 4).unwrap();
        cursor.next();
        cursor.seek(n / 4 + 200_000).unwrap();
        cursor.next();
        cursor.nth(100_000);
        cursor.seek(n / 4 + 200_000 + 100_003).unwrap();
        cursor.next();
    });
    assert_eq!(count, 0, "seeking an existing cursor allocated {count} times");
    assert_eq!(cursor.offset(), n / 4 + 200_000 + 100_004);
}

/// A cursor first entered near the end reserves its tournament for parts that a later
/// backward seek can revive, even though almost all of them have already finished.
#[test]
fn seeking_backward_revives_parts_without_allocating() {
    let order = Order::new(Seq::mix((0..100).map(|i| Seq::source(if i == 0 { 1_000_000 } else { 1000 })))).unwrap();
    let mut cursor = order.iter(order.len() - 1..).unwrap();
    assert_eq!(cursor.next().map(|item| (*item.source, item.record_index)), Some(element(&order, order.len() - 1)));
    let count = allocations(|| cursor.seek(0).unwrap());
    assert_eq!(count, 0, "reviving the exhausted mix parts allocated {count} times");
    assert_eq!(cursor.next().map(|item| (*item.source, item.record_index)), Some(element(&order, 0)));
}

/// A clone can grow new buffers on its first backward seek, then reuse them.
#[test]
fn cloned_cursors_revive_parts_and_reuse_new_buffers() {
    let order = Order::new(Seq::mix((0..100).map(|i| Seq::source(if i == 0 { 1_000_000 } else { 1000 })))).unwrap();
    let mut cursor = order.iter(order.len() - 1..).unwrap();
    cursor.next();
    let mut cloned = cursor.clone();
    cloned.seek(0).unwrap();
    assert_eq!(cloned.next().map(|item| (*item.source, item.record_index)), Some(element(&order, 0)));
    let checks = [order.len() - 1, 0, 100, order.len() / 3].map(|pos| (pos, order.get(pos)));
    let count = allocations(|| {
        for (pos, expected) in checks {
            cloned.seek(pos).unwrap();
            assert_eq!(cloned.next(), expected);
        }
    });
    assert_eq!(count, 0, "a warmed clone allocated {count} times");
    assert_eq!(cursor.next(), None);
}

#[test]
fn cloned_cursors_cross_concat_children_independently() {
    let part = |k| Seq::mix((0..k).map(|i| Seq::source(10).shuffle(i as u64)));
    let order = Order::new(Seq::concat([part(1000), part(2)]).repeat(2)).unwrap();
    let mut cursor = order.iter(..).unwrap();
    cursor.next();
    cursor.seek(10_000).unwrap();
    cursor.next();
    let mut cloned = cursor.clone();
    let expected = order.get(0).unwrap();
    let repeated = order.get(10_020).unwrap();
    cloned.seek(0).unwrap();
    assert_eq!(cloned.next(), Some(expected));
    cloned.set_range(10_020..).unwrap();
    assert_eq!(cloned.next(), Some(repeated));
    assert_eq!(cursor.offset(), 10_001);
    assert_eq!(cursor.next(), order.get(10_001));
}

#[test]
fn forward_seeks_land_in_the_target_repetition_and_part() {
    // Every repetition entered rebuilds the concat's current part cursor. Landing
    // directly enters only one repetition; the bound includes the mix state box.
    let epoch = || {
        Seq::concat([
            Seq::mix([Seq::source(1000).shuffle(1), Seq::source(500).shuffle(2)]),
            Seq::mix([Seq::source(300).shuffle(3), Seq::source(10)]),
        ])
    };
    let order = Order::new(epoch().repeat(100_000)).unwrap();
    let n = order.len();
    for target in [n - 7, n - 1000 - 500 - 1, n / 2] {
        let mut cursor = order.iter(..).unwrap();
        let count = allocations(|| cursor.seek(target).unwrap());
        assert!(count <= 8, "forward seek across 100 000 repetitions made {count} allocations");
        assert_eq!(cursor.next().map(|item| (*item.source, item.record_index)), Some(element(&order, target)));
        let mut cursor = order.iter(..).unwrap();
        let count = allocations(|| {
            cursor.nth(target - 1);
        });
        assert!(count <= 9, "nth across 100 000 repetitions made {count} allocations");
        assert_eq!(cursor.next().map(|item| (*item.source, item.record_index)), Some(element(&order, target)));
    }
    // A concat of many mixes: entering a part builds its cursor; landing directly builds one.
    let order = Order::new(Seq::concat((0..20_000).map(|c| Seq::mix([Seq::source(50).shuffle(c + 1), Seq::source(30)])))).unwrap();
    let n = order.len();
    for target in [n - 7, n - 80, n / 2 + 1] {
        let mut cursor = order.iter(..).unwrap();
        let count = allocations(|| cursor.seek(target).unwrap());
        assert!(count <= 8, "forward seek across 20 000 parts made {count} allocations");
        assert_eq!(cursor.next().map(|item| (*item.source, item.record_index)), Some(element(&order, target)));
    }
}

#[test]
fn count_and_last_on_a_large_shuffled_source() {
    let order = Order::new(Seq::source(1usize << 30).shuffle(1)).unwrap();
    assert_eq!(order.iter(5..).unwrap().count(), order.len() - 5);
    assert_eq!(order.iter(..).unwrap().last().map(|item| (*item.source, item.record_index)), Some(element(&order, order.len() - 1)));
}

/// Empty parts retain their source handles, but take no cursor slots or seek scratch.
#[test]
fn empty_parts_do_not_allocate_runtime_state() {
    let make = |empty| {
        Order::new(Seq::mix([Seq::source(1_000_000), Seq::source(1_000_000)].into_iter().chain((0..empty).map(|_| Seq::source(0)))))
            .unwrap()
    };
    let (plain, padded) = (make(0), make(100_000));
    assert_eq!(padded.sources().len(), 100_002);
    let bytes = |order: &Order<usize>| {
        let before = BYTES.get();
        let mut cursor = order.iter(500_000..).unwrap();
        black_box(cursor.next());
        black_box(order.get(1_000_001).unwrap());
        BYTES.get() - before
    };
    assert_eq!(bytes(&plain), bytes(&padded));
}

#[test]
fn shuffles_reuse_every_reached_mix_including_concat_children() {
    let mix = || Seq::mix((0..10).map(|_| Seq::source(100)));
    let order = Order::new(Seq::concat([mix(), Seq::mix([mix(), mix()])]).repeat(3).shuffle(8)).unwrap();
    let mut cursor = order.iter(..).unwrap();
    cursor.by_ref().for_each(drop); // Reach all cached mixes and exhaust the cursor.
    let mut clone = cursor.clone();
    // A clone warms its own seek scratch before allocation-free reuse.
    clone.set_range(..).unwrap();
    clone.by_ref().for_each(drop);
    for c in [&mut cursor, &mut clone] {
        let count = allocations(|| {
            c.set_range(..).unwrap();
            c.by_ref().take(1000).for_each(|item| {
                black_box(item);
            });
            c.seek(300).unwrap();
            black_box(c.next());
        });
        assert_eq!(count, 0, "a warmed shuffled composition allocated {count} times");
        assert_eq!(c.next(), order.get(301));
    }
}
