//! Compensated sums for schedule construction.

/// A sum and its rounding residual, stored as two floats using Knuth's two-sum.
/// Sufficient for bounded rate changes. Signed slopes spanning many magnitudes
/// need an [`Expansion`] to retain smaller contributions after larger ones are removed.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Compensated {
    hi: f64,
    lo: f64,
}

impl Compensated {
    pub(crate) fn add(&mut self, x: f64) {
        let (sum, err) = two_sum(self.hi, x);
        (self.hi, self.lo) = two_sum(sum, self.lo + err);
    }

    pub(crate) fn value(self) -> f64 {
        self.hi + self.lo
    }
}

/// `a + b` rounded, and the rounding error, exactly.
fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let sum = a + b;
    let bb = sum - a;
    (sum, (a - (sum - bb)) + (b - bb))
}

/// An exact expansion of finite terms: retain every rounding residual, even when slopes
/// span hundreds of orders of magnitude. Removing a term later recovers all smaller
/// terms. The number of nonoverlapping partials is bounded by the binary64 exponent range.
#[derive(Default)]
pub(crate) struct Expansion {
    partials: Vec<f64>,
}

impl Expansion {
    pub(crate) fn add(&mut self, mut x: f64) {
        let mut kept = 0;
        for i in 0..self.partials.len() {
            let (sum, err) = two_sum(x, self.partials[i]);
            if err != 0.0 {
                self.partials[kept] = err;
                kept += 1;
            }
            x = sum;
        }
        self.partials.truncate(kept);
        if x != 0.0 {
            self.partials.push(x);
        }
    }

    pub(crate) fn value(&self) -> f64 {
        self.partials.iter().rev().sum()
    }
}
