//! Versioned benchmark records shared by the producer and campaign runner.
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub type Measurements = Vec<Option<f64>>;
pub type Rows = BTreeMap<String, Measurements>;
pub const PREFIX: &str = "DATAORDER_BENCH_V2 ";

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Measurement {
    pub workload: String,
    /// Eight columns: seek µs, walk ns, get ns, build µs, reused seek µs,
    /// requested bytes, retained bytes and peak live bytes. Schema 1 has the first six.
    pub samples: Vec<Measurements>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Report {
    pub schema: u32,
    pub harness: String,
    pub mode: String,
    pub rows: BTreeMap<String, Measurement>,
}

impl Report {
    pub fn parse(stdout: &str) -> Result<Self, String> {
        let mut reports = stdout.lines().filter_map(|line| {
            line.strip_prefix(PREFIX).map(|body| (2, body)).or_else(|| line.strip_prefix("DATAORDER_BENCH_V1 ").map(|body| (1, body)))
        });
        let (schema, body) = reports.next().ok_or("benchmark did not produce a versioned report")?;
        let report: Self = serde_json::from_str(body).map_err(|e| format!("invalid benchmark report: {e}"))?;
        if reports.next().is_some() {
            return Err("benchmark produced multiple reports".into());
        }
        if report.schema != schema {
            return Err("benchmark prefix and schema disagree".into());
        }
        report.validate()?;
        Ok(report)
    }

    pub fn validate(&self) -> Result<(), String> {
        if !matches!(self.schema, 1 | 2) || self.harness.is_empty() || self.rows.is_empty() {
            return Err("unsupported or empty benchmark report".into());
        }
        if !["default", "phases", "lifecycle", "all"].contains(&self.mode.as_str()) {
            return Err("unknown benchmark report mode".into());
        }
        for (name, row) in &self.rows {
            if name.is_empty() || row.workload.is_empty() || row.samples.len() < 3 {
                return Err(format!("{name}: missing workload identity or samples"));
            }
            let lifecycle = name.starts_with("lifecycle: ");
            for sample in &row.samples {
                if sample.len() != if self.schema == 1 { 6 } else { 8 }
                    || sample.iter().enumerate().any(|(col, value)| {
                        let expected = if lifecycle { col >= 3 } else { col < 3 };
                        value.is_some() != expected || value.is_some_and(|v| !v.is_finite() || v < 0.0)
                    })
                {
                    return Err(format!("{name}: invalid measurement columns"));
                }
            }
        }
        Ok(())
    }

    pub fn compatible(&self, other: &Self) -> Result<(), String> {
        if self.schema != other.schema || self.harness != other.harness || self.mode != other.mode {
            return Err("benchmark harness or mode differs; use identical bench.rs and support files in both revisions".into());
        }
        if self.rows.keys().ne(other.rows.keys()) {
            return Err("benchmark row set changed between runs".into());
        }
        for (name, row) in &self.rows {
            if row.workload != other.rows[name].workload {
                return Err(format!("{name}: benchmark workload changed between runs"));
            }
        }
        Ok(())
    }
}

/// Median, minimum and maximum of each available column. Inputs must be validated.
pub fn summarize(samples: &[Measurements]) -> (Measurements, Measurements, Measurements) {
    let (mut median, mut low, mut high) = (Vec::new(), Vec::new(), Vec::new());
    for col in 0..8 {
        let mut values: Vec<_> = samples.iter().filter_map(|sample| sample.get(col).copied().flatten()).collect();
        values.sort_by(f64::total_cmp);
        let middle = if values.is_empty() { None } else { Some((values[(values.len() - 1) / 2] + values[values.len() / 2]) / 2.0) };
        median.push(middle);
        low.push(values.first().copied());
        high.push(values.last().copied());
    }
    (median, low, high)
}

/// Stable workload identity; this is an accidental-change check, not a security hash.
pub fn fingerprint(bytes: &[u8]) -> String {
    let hash = bytes.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, &b| (h ^ u64::from(b)).wrapping_mul(0x100_0000_01b3));
    format!("{hash:016x}")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn report() -> Report {
        Report {
            schema: 2,
            harness: "harness".into(),
            mode: "default".into(),
            rows: BTreeMap::from([(
                "source".into(),
                Measurement {
                    workload: "source10".into(),
                    samples: vec![vec![Some(1.0), Some(2.0), Some(3.0), None, None, None, None, None]; 5],
                },
            )]),
        }
    }

    #[test]
    fn malformed_and_partial_measurements_are_rejected() {
        let r = report();
        let raw = format!("human output\n{PREFIX}{}\n", serde_json::to_string(&r).unwrap());
        Report::parse(&raw).unwrap();
        assert!(Report::parse(&format!("{raw}{raw}")).is_err());
        assert!(Report::parse("source 1 µs 2 ns 3 ns").is_err());
        for col in 0..8 {
            let mut bad = r.clone();
            bad.rows.get_mut("source").unwrap().samples[0][col] = if col < 3 { None } else { Some(1.0) };
            assert!(bad.validate().is_err());
        }
        let mut bad = r.clone();
        bad.rows.get_mut("source").unwrap().samples[0][0] = Some(f64::NAN);
        assert!(bad.validate().is_err());
        let mut bad = r.clone();
        bad.rows.remove("source");
        assert!(r.compatible(&bad).is_err());
        let mut bad = r.clone();
        bad.rows.get_mut("source").unwrap().workload = "different".into();
        assert!(r.compatible(&bad).is_err());
        let mut bad = r.clone();
        bad.harness = "different".into();
        assert!(r.compatible(&bad).is_err());
    }

    #[test]
    fn legacy_reports_load_and_memory_columns_are_required_in_v2() {
        let mut old = report();
        old.schema = 1;
        for sample in &mut old.rows.get_mut("source").unwrap().samples {
            sample.truncate(6);
        }
        let json = serde_json::to_string(&old).unwrap();
        Report::parse(&format!("DATAORDER_BENCH_V1 {json}")).unwrap();
        assert!(Report::parse(&format!("{PREFIX}{json}")).is_err());
        old.schema = 2;
        assert!(old.validate().is_err());
        let mut lifecycle = Report {
            schema: 2,
            harness: "h".into(),
            mode: "lifecycle".into(),
            rows: BTreeMap::from([(
                "lifecycle: mix".into(),
                Measurement {
                    workload: "mix".into(),
                    samples: vec![vec![None, None, None, Some(1.0), Some(2.0), Some(1024.0), Some(512.0), Some(768.0)]; 3],
                },
            )]),
        };
        lifecycle.validate().unwrap();
        lifecycle.rows.get_mut("lifecycle: mix").unwrap().samples[0][6] = None;
        assert!(lifecycle.validate().is_err());
    }

    #[test]
    fn summary_reports_median_and_range() {
        let mut r = report();
        for (sample, value) in r.rows.get_mut("source").unwrap().samples.iter_mut().zip([9.0, 1.0, 5.0, 4.0, 3.0]) {
            sample[0] = Some(value);
        }
        let (median, low, high) = summarize(&r.rows["source"].samples);
        assert_eq!((median[0], low[0], high[0]), (Some(4.0), Some(1.0), Some(9.0)));
        assert_eq!(median[3], None);
    }
}
