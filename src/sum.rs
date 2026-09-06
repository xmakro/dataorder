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
    pub(crate) fn new(value: f64) -> Self {
        Self { hi: value, lo: 0.0 }
    }

    /// Preserve the residual of a division before subtracting a fraction near one.
    pub(crate) fn ratio(n: f64, d: f64) -> Self {
        Self::new(n).divided_by(Self::new(d))
    }

    pub(crate) fn divided_by(self, denominator: Self) -> Self {
        let hi = self.hi / denominator.hi;
        let product = denominator.scaled(hi);
        let mut remainder = self;
        remainder.add(-product.hi);
        remainder.add(-product.lo);
        Self { hi, lo: remainder.value() / denominator.hi }
    }

    pub(crate) fn difference(a: f64, b: f64) -> Self {
        let mut difference = Self::new(a);
        difference.add(-b);
        difference
    }

    /// Dekker's product residual, without fused arithmetic. Splitting by masking
    /// low significand bits avoids overflowing the usual splitter multiplication.
    fn product(a: f64, b: f64) -> Self {
        let hi = a * b;
        let split = |x: f64| {
            let high = f64::from_bits(x.to_bits() & !((1u64 << 27) - 1));
            (high, x - high)
        };
        let (ah, al) = split(a);
        let (bh, bl) = split(b);
        let lo = al * bl - (((hi - ah * bh) - al * bh) - ah * bl);
        Self { hi, lo }
    }

    pub(crate) fn scaled(self, scale: f64) -> Self {
        let mut product = Self::product(self.hi, scale);
        product.add(self.lo * scale);
        product
    }

    pub(crate) fn multiply(self, other: Self) -> Self {
        let mut product = self.scaled(other.hi);
        for term in self.scaled(other.lo).terms() {
            product.add(term);
        }
        product
    }

    pub(crate) fn terms(self) -> [f64; 2] {
        [self.hi, self.lo]
    }

    /// Subtract before rounding the accumulated value to a single float.
    pub(crate) fn remaining(self, capacity: f64) -> f64 {
        let mut difference = Self { hi: capacity, lo: 0.0 };
        difference.add(-self.hi);
        difference.add(-self.lo);
        difference.value()
    }

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

    /// Integrate every slope partial before rounding; a small slope can matter
    /// when nearly all capacity is scheduled and the uniform remainder is tiny.
    pub(crate) fn add_scaled(&self, into: &mut Compensated, width: Compensated) {
        for &partial in &self.partials {
            for term in width.scaled(partial).terms() {
                into.add(term);
            }
        }
    }
}
