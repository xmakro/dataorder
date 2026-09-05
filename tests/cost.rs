//! The cost model, pinned by counting allocations: seeking an existing cursor allocates
//! nothing, and a forward seek or `nth` across many repetitions or concat parts lands in
//! the target one instead of entering every one on the way. One timing test: a shard of a
//! nested mix must skip the inner mixes' cursors rather than re-seek them per element.

use dataorder::{Order, Sampling, Seq};
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::hint::black_box;
use std::time::{Duration, Instant};

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
    assert_eq!(cursor.position(), n / 4 + 200_000 + 100_004);
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

#[test]
fn forward_seeks_land_in_the_target_repetition_and_part() {
    // Every repetition entered rebuilds the concat's current part cursor (a mix: two
    // allocations); landing directly enters one repetition.
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
        assert!(count <= 8, "nth across 100 000 repetitions made {count} allocations");
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

/// `count` and `last` do not walk: on a shuffled source far too long to walk in the time
/// allowed, they answer from the range and by one random access.
#[test]
fn count_and_last_do_not_walk() {
    let order = Order::new(Seq::source(1usize << 30).shuffle(1)).unwrap();
    let t = Instant::now();
    assert_eq!(order.iter(5..).count(), order.len() - 5);
    assert_eq!(order.iter(..).last().map(|(&s, i)| (s, i)), Some(element(&order, order.len() - 1)));
    assert!(t.elapsed() < Duration::from_secs(1), "count or last walked the order ({:?})", t.elapsed());
}

/// A seek's counting phase costs `O(k log S)` in the number `S` of distinct breakpoints among the
/// parts' schedules: from a cold hint the profile's segment is found by a binary search, not
/// by walking the segments. With 800 distinct starts among 4000 parts, walking made seeks
/// about 20 times dearer than with one start; searching keeps them within a few times (the
/// searches, and the larger uniform profile's cache footprint).
#[test]
fn seeks_do_not_walk_the_schedule_segments() {
    let mix = |distinct: bool| {
        Seq::mix_with((0..4000u32).map(move |i| {
            let start = if distinct { 0.1 + 0.6 * f64::from(i) / 4000.0 } else { 0.3 };
            let sampling = if i % 5 == 0 { Sampling::delayed(start) } else { Sampling::Uniform };
            (Seq::source(1000usize).shuffle(u64::from(i) + 1), sampling)
        }))
    };
    let (same, distinct) = (Order::new(mix(false)).unwrap(), Order::new(mix(true)).unwrap());
    let seeks = |order: &Order<usize>| {
        let n = order.len();
        let t = Instant::now();
        for i in 1..=10 {
            black_box(order.iter(n / 11 * i..).next());
        }
        t.elapsed()
    };
    let min = |order: &Order<usize>| (0..3).map(|_| seeks(order)).min().unwrap_or(Duration::MAX);
    let (a, b) = (min(&same), min(&distinct));
    assert!(b < a * 10, "seeks with distinct schedules took {b:?}, with one schedule {a:?}");
}

/// Walking a shard of a mix of mixes steps the outer interleave and skips the inner mixes'
/// cursors forward, so it costs a small multiple of the unsharded walk (it steps `count`
/// times as many interleave positions). Re-seeking an inner mix for every kept element, as
/// a stale part cursor once did, costs a full interleave seek per element: 80 times the
/// unsharded walk on this configuration, against 4 times when it skips.
#[test]
fn shards_of_nested_mixes_skip_the_inner_cursors() {
    let inner = |first: u64| Seq::mix((0..50).map(move |i| Seq::source(20_000).shuffle(first + i + 1)));
    let nested = || Seq::mix([inner(0), inner(50)]);
    let unsharded = Order::new(nested()).unwrap();
    let sharded = Order::new(nested().shard(8, 0)).unwrap();
    let walk = |order: &Order<usize>| {
        let t = Instant::now();
        black_box(order.iter(..20_000).fold(0usize, |acc, (s, i)| acc ^ s ^ i));
        t.elapsed()
    };
    let min = |order: &Order<usize>| (0..5).map(|_| walk(order)).min().unwrap_or(Duration::MAX);
    let (base, shard) = (min(&unsharded), min(&sharded));
    assert!(shard < base * 25, "a shard of a nested mix walked in {shard:?}, the mix itself in {base:?}");
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
