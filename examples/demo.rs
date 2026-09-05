//! A schedule over two epochs: three sources, one delayed, one ramped, each
//! shuffled afresh per epoch, and the whole order sharded over four workers. The parts are
//! repeated, not the mix, so that the schedules span the run rather than each epoch.
use dataorder::{Order, Sampling, Seq, Source};

#[derive(Clone)]
struct Src {
    name: char,
    len: usize,
}

impl Source for Src {
    fn len(&self) -> usize {
        self.len
    }

    fn salt(&self) -> u64 {
        self.name as u64
    }
}

fn main() {
    let seq = Seq::mix_with([
        (Seq::source(Src { name: 'A', len: 60 }).shuffle(1).repeat(2), Sampling::Uniform),
        (Seq::source(Src { name: 'B', len: 20 }).shuffle(2).repeat(2), Sampling::delayed(0.5)), // B: second half of the run only
        (Seq::source(Src { name: 'C', len: 40 }).shuffle(3).repeat(2), Sampling::ramp(0.2, 0.6)), // C: ramps up from 20% to 60%
    ]);
    let order = Order::new(seq.clone()).unwrap();
    let sources: Vec<(char, usize)> = order.sources().iter().map(|s| (s.name, s.len)).collect();
    println!("length {} over sources {sources:?}", order.len());

    let line: String = order.iter(..).map(|(s, _)| s.name).collect();
    println!("order (two epochs):\n{line}");

    println!("\nthe second half in detail, its first 20 elements:");
    let part: Vec<String> = order.iter(120..140).map(|(s, i)| format!("{}{i}", s.name)).collect();
    println!("  {}", part.join(" "));

    println!("\nfour shards, first 10 elements each (shard w holds every 4th element, offset w):");
    for w in 0..4 {
        let shard = Order::new(seq.clone().shard(4, w)).unwrap();
        let part: Vec<String> = shard.iter(0..10).map(|(s, i)| format!("{}{i}", s.name)).collect();
        println!("  shard {w}: {}", part.join(" "));
    }

    let (s, i) = order.get(137);
    let (t, j) = order.iter(137..138).next().unwrap();
    println!("\nrandom access agrees with iteration: get(137) = {}{i}, iter(137..138) = {}{j}", s.name, t.name);
}
