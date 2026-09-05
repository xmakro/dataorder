//! The cost model, pinned by counting allocations: seeking an existing cursor allocates
//! nothing, and a forward seek or `nth` across many repetitions or concat parts lands in
//! the target one instead of entering every one on the way.

use dataorder::{Order, Seq};
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};

struct Counting;

static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Relaxed);
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static ALLOCATOR: Counting = Counting;

/// The counter is global, so the tests run one at a time.
static ONE_AT_A_TIME: Mutex<()> = Mutex::new(());

fn allocations(f: impl FnOnce()) -> usize {
    let before = ALLOCATIONS.load(Relaxed);
    f();
    ALLOCATIONS.load(Relaxed) - before
}

fn element(order: &Order<usize>, pos: usize) -> (usize, usize) {
    let (s, i) = order.get(pos);
    (*s, i)
}

#[test]
fn seeking_an_existing_cursor_allocates_nothing() {
    let _serial = ONE_AT_A_TIME.lock().unwrap();
    let order = Order::new(Seq::mix((0..100).map(|i| Seq::source(10_000).shuffle(i + 1)))).unwrap();
    let n = order.len();
    let mut cursor = order.iter(n / 2..);
    cursor.next();
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

#[test]
fn forward_seeks_land_in_the_target_repetition_and_part() {
    let _serial = ONE_AT_A_TIME.lock().unwrap();
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
