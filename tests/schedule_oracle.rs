//! Expected positions come from rational CDFs in the independent Python oracle.
use dataorder::{Order, Schedule, Seq, Source};

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
        let order = Order::new(Seq::mix(lens.iter().enumerate().map(|(i, n)| {
            let schedule = match fixture["schedules"][i].as_array() {
                None => Schedule::Uniform,
                Some(p) => {
                    Schedule::trapezoid(p[0].as_f64().unwrap(), p[1].as_f64().unwrap(), p[2].as_f64().unwrap(), p[3].as_f64().unwrap())
                }
            };
            (Seq::source(Dataset { ordinal: i, len: n.as_u64().unwrap() as usize }), schedule)
        })))
        .unwrap();
        let samples = fixture["samples"].as_array().unwrap();
        let mut walking = order.cursor(0..0).unwrap();
        let mut next_position = None;
        for sample in samples {
            let pos = sample[0].as_u64().unwrap() as usize;
            let expected = (sample[1].as_u64().unwrap() as usize, sample[2].as_u64().unwrap() as usize);
            let dataorder::Item { source, record_index: index, .. } = order.get(pos).unwrap();
            assert_eq!((source.ordinal, index), expected, "case {case}, position {pos}");
            let dataorder::Item { source, record_index: index, .. } = order.cursor(pos..).unwrap().next().unwrap();
            assert_eq!((source.ordinal, index), expected, "cursor: case {case}, position {pos}");
            // Every small fixture is a complete walk. Large fixtures contain short
            // independently generated windows: seek only across gaps, then exercise
            // the tournament and cached segment transitions with consecutive nexts.
            if next_position != Some(pos) {
                walking.reset(pos..).unwrap();
            }
            let dataorder::Item { source, record_index: index, .. } = walking.next().unwrap();
            assert_eq!((source.ordinal, index), expected, "walk: case {case}, position {pos}");
            next_position = pos.checked_add(1);
        }
        if samples.len() == order.len() {
            assert!(walking.next().is_none());
        }
    }
}
