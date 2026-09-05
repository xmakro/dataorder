//! A training-style schedule: three sources, one delayed, one ramped, shuffled per epoch
//! and sharded over four workers.
use dataorder::{Dataset, Order, Sampling, Seq};

#[derive(Clone)]
struct Src {
    name: char,
    len: usize,
}

impl Dataset for Src {
    fn len(&self) -> usize {
        self.len
    }
}

fn main() {
    let seq = Seq::mix_with([
        (Seq::source(Src { name: 'A', len: 60 }).shuffle(1), Sampling::Uniform),
        (Seq::source(Src { name: 'B', len: 20 }).shuffle(2), Sampling::DelayedLinear(0.5, 0.5)), // B: second half only
        (Seq::source(Src { name: 'C', len: 40 }).shuffle(3), Sampling::DelayedLinear(0.2, 0.6)), // C: ramps up from 20% to 60%
    ])
    .repeat(2);
    let order = Order::compile(seq.clone()).unwrap();
    let sources: Vec<(char, usize)> = order.sources().iter().map(|s| (s.name, s.len)).collect();
    println!("length {} over sources {sources:?}", order.len());

    let line: String = order.iter(0..order.len()).map(|(s, _)| s.name).collect();
    println!("order (two epochs):\n{line}");

    println!("\nepoch 1 in detail, first 20 elements:");
    let part: Vec<String> = order.iter(120..140).map(|(s, i)| format!("{}{i}", s.name)).collect();
    println!("  {}", part.join(" "));

    println!("\nfour shards, first 10 elements each (shard w holds every 4th element, offset w):");
    for w in 0..4 {
        let shard = Order::compile(seq.clone().shard(w, 4)).unwrap();
        let part: Vec<String> = shard.iter(0..10).map(|(s, i)| format!("{}{i}", s.name)).collect();
        println!("  shard {w}: {}", part.join(" "));
    }

    let (s, i) = order.get(137);
    let (t, j) = order.iter(137..138).next().unwrap();
    println!("\nrandom access agrees with iteration: get(137) = {}{i}, iter(137..138) = {}{j}", s.name, t.name);
}
