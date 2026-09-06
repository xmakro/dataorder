//! The cost model, pinned by counting allocations: seeking an existing cursor allocates
//! nothing, and a forward seek or `nth` across many repetitions or concat parts lands in
//! the target one instead of entering every one on the way.

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
    let (s, i) = order.get(pos);
    (*s, i)
}

#[test]
fn unary_traversals_do_not_allocate_temporary_child_lists() {
    let (parts, depth) = (128, 100);
    let make = || Seq::mix((0..parts).map(|_| (0..depth).fold(Seq::source(10), |seq, _| seq.take(10))));
    let seq = make();
    let count = allocations(|| {
        let order = Order::new(seq).unwrap();
        assert_eq!(order.len(), parts * 10);
    });
    // One box for each transformed output node plus shared traversal/compile
    // buffers. Temporary singleton Vecs would almost double this budget.
    assert!(count < parts * depth + 1000, "compilation allocated {count} times");
    let seq = make();
    let count = allocations(|| seq.map(|n| n + 1).dispose());
    assert!(count < parts * depth + 1000, "mapping allocated {count} times");
}

#[test]
fn concat_recycles_mix_buffers_across_epochs_and_seeks() {
    use dataorder::Sampling;
    let part = |k, len, scheduled| {
        Seq::mix_with((0..k).map(|i| {
            (Seq::source(len).shuffle(i as u64 + 1), if scheduled && i % 5 == 0 { Sampling::ramp(0.1, 0.6) } else { Sampling::Uniform })
        }))
    };
    let order = Order::new(Seq::concat([part(1000, 10, false), part(700, 20, true)]).repeat(3)).unwrap();
    let epoch = order.len() / 3;
    let mut cursor = order.iter(..);
    cursor.by_ref().take(epoch).for_each(|item| {
        black_box(item);
    });
    let count = allocations(|| {
        cursor.by_ref().take(epoch).for_each(|item| {
            black_box(item);
        });
        for pos in [10_000, 0, 17_001, epoch, 9999, 10_000, 10_001] {
            cursor.seek(pos);
            black_box(cursor.next());
        }
    });
    assert_eq!(count, 0, "recycled mix buffers allocated {count} times");
}

/// A mix builds a part's cursor when the part is first drawn from (nothing to allocate for
/// a shuffled source, whose cursor sits in the mix's slot for it); after that, seeks reuse
/// everything. Drawing a thousand elements first enters every part of a balanced mix of a
/// hundred, so the count below is of seeking alone.
#[test]
fn seeking_an_existing_cursor_allocates_nothing() {
    let order = Order::new(Seq::mix((0..100).map(|i| Seq::source(10_000).shuffle(i + 1)))).unwrap();
    let n = order.len();
    let mut cursor = order.iter(n / 2..);
    cursor.by_ref().take(1000).for_each(drop);
    let count = allocations(|| {
        cursor.seek(n / 4);
        cursor.next();
        cursor.seek(n / 4 + 200_000);
        cursor.next();
        cursor.nth(100_000);
        cursor.seek(n / 4 + 200_000 + 100_003);
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
    let mut cursor = order.iter(order.len() - 1..);
    assert_eq!(cursor.next().map(|(&s, i)| (s, i)), Some(element(&order, order.len() - 1)));
    let count = allocations(|| cursor.seek(0));
    assert_eq!(count, 0, "reviving the exhausted mix parts allocated {count} times");
    assert_eq!(cursor.next().map(|(&s, i)| (s, i)), Some(element(&order, 0)));
}

/// Cloning a partially exhausted mix retains the spare capacity of its seek buffers.
#[test]
fn cloned_cursors_revive_parts_without_allocating() {
    let order = Order::new(Seq::mix((0..100).map(|i| Seq::source(if i == 0 { 1_000_000 } else { 1000 })))).unwrap();
    let mut cursor = order.iter(order.len() - 1..);
    cursor.next();
    let mut cloned = cursor.clone();
    let count = allocations(|| cloned.seek(0));
    assert_eq!(count, 0, "a cloned cursor allocated {count} times when reviving parts");
    assert_eq!(cloned.next().map(|(&s, i)| (s, i)), Some(element(&order, 0)));
    assert_eq!(cursor.next(), None);
}

/// Empty ranges, counts and untouched clones need no runtime tree. `last` pays only for
/// its single random access, without first building a cursor over the beginning.
#[test]
fn cursors_allocate_only_when_drawing_an_element() {
    let order = Order::new(Seq::mix((0..1000).map(|i| Seq::source(1000).shuffle(i)))).unwrap();
    let count = allocations(|| {
        for start in [0, order.len() / 2, order.len()] {
            let mut cursor = order.iter(start..start);
            assert_eq!(cursor.next(), None);
            assert_eq!(cursor.nth(usize::MAX), None);
            cursor.set_range(..);
            cursor.seek(order.len() / 3);
            assert_eq!(cursor.clone().count(), order.len() - order.len() / 3);
            assert_eq!(cursor.clone().indexed().count(), order.len() - order.len() / 3);
            assert_eq!(cursor.count(), order.len() - order.len() / 3);
        }
        assert_eq!(order.iter(..).count(), order.len());
    });
    assert_eq!(count, 0, "an undrawn cursor allocated {count} times");
    let lookup = allocations(|| {
        black_box(order.get(order.len() - 1));
    });
    let last = allocations(|| {
        black_box(order.iter(..).last());
    });
    assert_eq!(last, lookup, "last built a cursor in addition to its random access");
}

#[test]
fn undrawn_cursors_resume_after_range_changes_and_skips() {
    let mix = || Seq::mix([Seq::source(100).shuffle(1), Seq::source(50).shuffle(2)]);
    let sequences = [
        Seq::source(100),
        Seq::source(100).shuffle(11),
        mix(),
        mix().repeat(3).stride(7, 2),
        Seq::concat([mix(), mix().shuffle(7)]).skip(20),
    ];
    for seq in sequences {
        let order = Order::with_seed(seq, 19).unwrap();
        for start in [0, order.len() / 2, order.len()] {
            let mut cursor = order.iter(start..start);
            assert_eq!(cursor.nth(usize::MAX), None);
            cursor.set_range(0..0);
            cursor.set_range(..);
            cursor.seek(order.len() / 3);
            let mut cloned = cursor.clone();
            let expected = Some(element(&order, order.len() / 3 + 1));
            assert_eq!(cursor.nth(1).map(|(&s, i)| (s, i)), expected);
            assert_eq!(cloned.nth(1).map(|(&s, i)| (s, i)), expected);
            cursor.set_range(order.len()..);
            assert_eq!(cursor.next(), None);
            cursor.set_range(..);
            assert_eq!(cursor.next().map(|(&s, i)| (s, i)), Some(element(&order, 0)));
        }
    }
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
        let mut cursor = order.iter(..);
        let count = allocations(|| cursor.seek(target));
        assert!(count <= 8, "forward seek across 100 000 repetitions made {count} allocations");
        assert_eq!(cursor.next().map(|(&s, i)| (s, i)), Some(element(&order, target)));
        let mut cursor = order.iter(..);
        let count = allocations(|| {
            cursor.nth(target - 1);
        });
        assert!(count <= 9, "nth across 100 000 repetitions made {count} allocations");
        assert_eq!(cursor.next().map(|(&s, i)| (s, i)), Some(element(&order, target)));
    }
    // A concat of many mixes: entering a part builds its cursor; landing directly builds one.
    let order = Order::new(Seq::concat((0..20_000).map(|c| Seq::mix([Seq::source(50).shuffle(c + 1), Seq::source(30)])))).unwrap();
    let n = order.len();
    for target in [n - 7, n - 80, n / 2 + 1] {
        let mut cursor = order.iter(..);
        let count = allocations(|| cursor.seek(target));
        assert!(count <= 8, "forward seek across 20 000 parts made {count} allocations");
        assert_eq!(cursor.next().map(|(&s, i)| (s, i)), Some(element(&order, target)));
    }
}

#[test]
fn count_and_last_on_a_large_shuffled_source() {
    let order = Order::new(Seq::source(1usize << 30).shuffle(1)).unwrap();
    assert_eq!(order.iter(5..).count(), order.len() - 5);
    assert_eq!(order.iter(..).last().map(|(&s, i)| (s, i)), Some(element(&order, order.len() - 1)));
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
        let mut cursor = order.iter(500_000..);
        black_box(cursor.next());
        black_box(order.get(1_000_001));
        BYTES.get() - before
    };
    assert_eq!(bytes(&plain), bytes(&padded));
}

#[test]
fn shuffles_reuse_every_reached_mix_including_concat_children() {
    let mix = || Seq::mix((0..10).map(|_| Seq::source(100)));
    let order = Order::new(Seq::concat([mix(), Seq::mix([mix(), mix()])]).repeat(3).shuffle(8)).unwrap();
    let mut cursor = order.iter(..);
    cursor.by_ref().for_each(drop); // Reach all cached mixes and exhaust the cursor.
    let mut clone = cursor.clone();
    for c in [&mut cursor, &mut clone] {
        let count = allocations(|| {
            c.set_range(..);
            c.by_ref().take(1000).for_each(|item| {
                black_box(item);
            });
            c.seek(300);
            black_box(c.next());
        });
        assert_eq!(count, 0, "a warmed shuffled composition allocated {count} times");
        assert_eq!(c.next(), Some(order.get(301)));
    }
}

#[test]
fn last_reuses_an_initialized_mix_for_both_cursor_types() {
    let order = Order::new(Seq::mix((0..100).map(|_| Seq::source(1000)))).unwrap();
    let mut cursor = order.iter(..);
    cursor.next();
    let indexed = cursor.clone().indexed();
    let expected = order.get(order.len() - 1);
    let count = allocations(|| assert_eq!(cursor.last(), Some(expected)));
    assert_eq!(count, 0);
    let expected = order.get_indexed(order.len() - 1);
    let count = allocations(|| assert_eq!(indexed.last(), Some(expected)));
    assert_eq!(count, 0);
}

#[test]
fn empty_ranges_defer_new_children_and_preserve_old_buffers() {
    let mix = || Seq::mix((0..100).map(|_| Seq::source(1000)));
    let order = Order::new(Seq::concat([mix(), mix()])).unwrap();
    let mut cursor = order.iter(..);
    cursor.next();
    let at = 100_005;
    assert_eq!(
        allocations(|| {
            cursor.set_range(at..at);
            assert_eq!(cursor.next(), None);
            assert_eq!(cursor.nth(usize::MAX), None);
            cursor.set_range(5..5);
            cursor.set_range(..10);
            black_box(cursor.next());
        }),
        0
    );
    assert_eq!(cursor.next(), Some(order.get(1)));
    cursor.set_range(at..at);
    let mut clone = cursor.clone();
    for c in [&mut cursor, &mut clone] {
        c.set_range(at..at + 3);
        assert_eq!(c.next(), Some(order.get(at)));
        assert_eq!(c.nth(usize::MAX), None);
        c.set_range(at + 3..at + 4);
        assert_eq!(c.next(), Some(order.get(at + 3)));
    }
}
