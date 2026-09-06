//! Versioned checkpoints using public APIs. Run with:
//! `cargo run --example checkpoint --features serde`
//!
//! Persist the complete configuration as its identity (structural equality, not a
//! collision-prone short hash), source versions/lengths/salts, seeds, shard settings
//! and the next worker-local position. Validate against current metadata on restore.
//! This example checkpoints a whole worker order; bounded-range users must also
//! save and validate their range end. The application owns checkpoint storage.
use dataorder::{ORDERING_VERSION, Order, Seq, Source};
use serde::{Deserialize, Serialize};

const FORMAT_VERSION: u32 = 1;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Dataset {
    name: String,
    // An immutable manifest/content version supplied by the storage system. A
    // length alone cannot detect a reordered or replaced dataset of the same size.
    version: String,
    records: usize,
    salt: u64,
}
impl Source for Dataset {
    fn len(&self) -> usize {
        self.records
    }
    fn salt(&self) -> u64 {
        self.salt
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Configuration {
    sequence: Seq<Dataset>,
    seed: u64,
    workers: usize,
    worker: usize,
}
impl Configuration {
    fn validate(mut self) -> Result<Self, String> {
        // Consuming validation also dismantles a rejected, excessively deep tree.
        self.sequence = self.sequence.validate().map_err(|e| e.to_string())?;
        Ok(self)
    }

    // Called with validated configurations before cloning the tree.
    fn order(&self) -> Result<Order<Dataset>, String> {
        let seq = self.sequence.clone().try_shard(self.workers, self.worker).map_err(|e| e.to_string())?;
        Order::with_seed(seq, self.seed).map_err(|e| e.to_string())
    }
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Checkpoint {
    format_version: u32,
    ordering_version: String,
    configuration: Configuration,
    next_offset: usize,
}

// Encapsulate the configuration and its order so a caller cannot accidentally
// save a cursor offset with a different seed, source manifest, or worker number.
struct Worker {
    configuration: Configuration,
    order: Order<Dataset>,
    next_offset: usize,
}
impl Worker {
    fn new(configuration: Configuration) -> Result<Self, String> {
        let configuration = configuration.validate()?;
        let order = configuration.order()?;
        let worker = Self { configuration, order, next_offset: 0 };
        // JSON's depth limit is smaller than the configuration compiler's. Check the
        // actual checkpoint envelope before accepting any work, using the same parser
        // as restore. Changing the offset later does not change its nesting depth.
        let _: Checkpoint = serde_json::from_str(&worker.checkpoint()?)
            .map_err(|e| format!("configuration cannot round-trip through the checkpoint format: {e}"))?;
        Ok(worker)
    }

    // Advance only after the caller has processed this item successfully.
    // Real loaders should checkpoint the last committed position, not prefetched work.
    // Seek once per batch, then stream using the cursor's reusable state.
    fn process_batch(&mut self, limit: usize, mut process: impl FnMut(&Dataset, usize) -> Result<(), String>) -> Result<usize, String> {
        let mut processed = 0;
        for (source, index) in self.order.iter(self.next_offset..).take(limit) {
            process(source, index)?;
            self.next_offset += 1;
            processed += 1;
        }
        Ok(processed)
    }

    fn checkpoint(&self) -> Result<String, String> {
        serde_json::to_string(&Checkpoint {
            format_version: FORMAT_VERSION,
            ordering_version: ORDERING_VERSION.into(),
            configuration: self.configuration.clone(),
            next_offset: self.next_offset,
        })
        .map_err(|e| e.to_string())
    }

    fn restore(json: &str, current: Configuration) -> Result<Self, String> {
        // Keep serde_json's default depth limit. Before calling this function,
        // load current source metadata from the storage system, not the checkpoint.
        let current = current.validate()?;
        let saved: Checkpoint = serde_json::from_str(json).map_err(|e| e.to_string())?;
        if saved.format_version != FORMAT_VERSION {
            return Err("unsupported checkpoint format".into());
        }
        if saved.ordering_version != ORDERING_VERSION {
            return Err("checkpoint ordering version differs".into());
        }
        if saved.configuration != current {
            return Err("checkpoint configuration or source metadata differs".into());
        }
        let mut worker = Self::new(current)?;
        if saved.next_offset > worker.order.len() {
            return Err("checkpoint offset is past the worker's end".into());
        }
        worker.next_offset = saved.next_offset;
        Ok(worker)
    }
}

fn configuration() -> Configuration {
    let source = |name: &str, version: &str, records| {
        Seq::source(Dataset { name: name.into(), version: version.into(), records, salt: dataorder::salt(name) })
    };
    Configuration {
        sequence: Seq::mix([source("text", "manifest-v3", 12).shuffle(1), source("code", "manifest-v7", 8).shuffle(2)]).repeat(2),
        seed: 42,
        workers: 3,
        worker: 1,
    }
}

fn main() -> Result<(), String> {
    let current = configuration(); // In production, read these manifests independently.
    let mut worker = Worker::new(current.clone())?;
    worker.process_batch(4, |source, index| {
        println!("processed {}[{index}]", source.name);
        Ok(())
    })?;
    let saved = worker.checkpoint()?;
    let mut resumed = Worker::restore(&saved, current.clone())?;
    println!("Resuming worker {} at position {}", resumed.configuration.worker, resumed.next_offset);
    resumed.process_batch(usize::MAX, |source, index| {
        println!("processed {}[{index}]", source.name);
        Ok(())
    })?;
    // Another restart observes all successfully processed work, including after resume.
    let finished = Worker::restore(&resumed.checkpoint()?, current)?;
    assert_eq!(finished.next_offset, finished.order.len());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_resumes_exactly_including_the_end() {
        let config = configuration();
        let expected: Vec<_> = config.order().unwrap().iter(..).map(|(s, i)| (s.name.clone(), i)).collect();
        let mut worker = Worker::new(config.clone()).unwrap();
        for split in 0..=expected.len() {
            let json = worker.checkpoint().unwrap();
            let resumed = Worker::restore(&json, config.clone()).unwrap();
            let rest: Vec<_> = resumed.order.iter(resumed.next_offset..).map(|(s, i)| (s.name.clone(), i)).collect();
            assert_eq!(rest, expected[split..]);
            worker.process_batch(1, |_, _| Ok(())).unwrap();
        }
        assert_eq!(worker.process_batch(1, |_, _| Ok(())).unwrap(), 0);
    }

    #[test]
    fn failed_processing_does_not_advance_the_checkpoint() {
        let mut worker = Worker::new(configuration()).unwrap();
        assert!(worker.process_batch(10, |_, _| Err("I/O failure".into())).is_err());
        assert_eq!(worker.next_offset, 0);
    }

    #[test]
    fn resumed_batches_commit_only_successful_items() {
        let config = configuration();
        let mut worker = Worker::new(config.clone()).unwrap();
        assert_eq!(worker.process_batch(4, |_, _| Ok(())).unwrap(), 4);
        let mut resumed = Worker::restore(&worker.checkpoint().unwrap(), config.clone()).unwrap();
        let expected: Vec<_> = resumed.order.iter(4..6).map(|(s, i)| (s.name.clone(), i)).collect();
        let mut processed = Vec::new();
        let error = resumed
            .process_batch(5, |source, index| {
                if processed.len() == 2 {
                    return Err("I/O failure".into());
                }
                processed.push((source.name.clone(), index));
                Ok(())
            })
            .unwrap_err();
        assert_eq!(error, "I/O failure");
        assert_eq!(processed, expected);
        let mut again = Worker::restore(&resumed.checkpoint().unwrap(), config).unwrap();
        assert_eq!(again.next_offset, 6);
        assert_eq!(again.process_batch(0, |_, _| panic!("empty batch")).unwrap(), 0);
        let remaining = again.order.len() - 6;
        assert_eq!(again.process_batch(usize::MAX, |_, _| Ok(())).unwrap(), remaining);
        assert_eq!(again.next_offset, again.order.len());
    }

    #[test]
    fn rejects_unrestorable_configurations_before_processing() {
        let mut config = configuration();
        config.sequence = (0..70).fold(config.sequence, |seq, _| seq.take(40));
        assert!(config.order().is_ok()); // Valid for dataorder, too deep for this JSON envelope.
        let error = Worker::new(config).err().unwrap();
        assert!(error.contains("cannot round-trip") && error.contains("recursion limit"), "{error}");
    }

    #[test]
    fn rejects_changed_configuration_and_source_metadata() {
        let config = configuration();
        let json = Worker::new(config.clone()).unwrap().checkpoint().unwrap();
        for change in 0..8 {
            let mut changed = config.clone();
            match change {
                0 => changed.seed += 1,
                1 => changed.workers += 1,
                2 => changed.worker = 0,
                3 => changed.sequence = changed.sequence.shuffle(99),
                _ => {
                    changed.sequence = changed.sequence.map(|mut source| {
                        match change {
                            4 => source.name.push('x'),
                            5 => source.version.push('x'),
                            6 => source.records += 1,
                            _ => source.salt += 1,
                        }
                        source
                    })
                }
            }
            assert!(Worker::restore(&json, changed).is_err(), "change {change}");
        }
        for (key, value) in [
            ("format_version", serde_json::json!(2)),
            ("ordering_version", serde_json::json!("0.0.0")),
            ("next_offset", serde_json::json!(usize::MAX)),
            ("unknown", serde_json::json!(true)),
        ] {
            let mut bad: serde_json::Value = serde_json::from_str(&json).unwrap();
            bad[key] = value;
            assert!(Worker::restore(&bad.to_string(), config.clone()).is_err(), "{key}");
        }
        for (workers, worker) in [(0, 0), (2, 2)] {
            let mut bad = config.clone();
            (bad.workers, bad.worker) = (workers, worker);
            assert!(Worker::new(bad).is_err());
        }
    }
}
