//! Expected positions come from rational CDFs in the independent Python oracle.
use dataorder::{Order, Sampling, Seq, Source};

#[derive(Clone)]
struct Dataset {
    ordinal: usize,
    len: usize,
}
impl Source for Dataset {
    fn len(&self) -> usize {
        self.len
    }
}

#[test]
fn schedules_match_independent_cdfs_and_minority_ranks() {
    let cases: serde_json::Value = serde_json::from_str(include_str!("fixtures/schedule_oracle.json")).unwrap();
    for (case, fixture) in cases.as_array().unwrap().iter().enumerate() {
        let lens = fixture["lens"].as_array().unwrap();
        if lens.iter().map(|n| n.as_u64().unwrap()).sum::<u64>() > usize::MAX as u64 {
            continue; // Large public lengths are checked on the 64-bit CI targets.
        }
        let order = Order::new(Seq::mix_with(lens.iter().enumerate().map(|(i, n)| {
            let sampling = match fixture["schedules"][i].as_array() {
                None => Sampling::Uniform,
                Some(p) => {
                    Sampling::trapezoid(p[0].as_f64().unwrap(), p[1].as_f64().unwrap(), p[2].as_f64().unwrap(), p[3].as_f64().unwrap())
                }
            };
            (Seq::source(Dataset { ordinal: i, len: n.as_u64().unwrap() as usize }), sampling)
        })))
        .unwrap();
        for sample in fixture["samples"].as_array().unwrap() {
            let pos = sample[0].as_u64().unwrap() as usize;
            let expected = (sample[1].as_u64().unwrap() as usize, sample[2].as_u64().unwrap() as usize);
            let (source, index) = order.get(pos);
            assert_eq!((source.ordinal, index), expected, "case {case}, position {pos}");
            let (source, index) = order.iter(pos..).next().unwrap();
            assert_eq!((source.ordinal, index), expected, "cursor: case {case}, position {pos}");
        }
    }
}
