//! Contrast stretches, computed from GPU histograms. Pure maths, no GPU.

/// A histogram of one band's finite values, as the GPU statistics produce.
#[derive(Clone, Debug, PartialEq)]
pub struct Histogram {
    /// The smallest and largest finite value seen. Bin `i` of `counts.len()`
    /// covers `min + (max - min) * [i, i + 1) / bins`; the last bin includes `max`.
    pub min: f32,
    pub max: f32,
    pub counts: Vec<u32>,
    /// Sum of `counts`: the number of finite pixels.
    pub total: u64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum StretchMode {
    /// Straight linear from the minimum to the maximum.
    MinMax,
    /// Linear between two percentiles (0–100), e.g. 2 and 98.
    Percentile { low: f32, high: f32 },
    /// ENVI's optimized linear: percentiles `low` and `high` (defaults 2.5 and
    /// 99), then widened by `low_adjust` (0.1) and `high_adjust` (0.5) times
    /// their difference.
    OptimizedLinear {
        low: f32,
        high: f32,
        low_adjust: f32,
        high_adjust: f32,
    },
    /// Mean ± `n` standard deviations.
    StdDev { n: f32 },
}

impl Histogram {
    /// The value below which `percent` (0–100) of the pixels fall,
    /// interpolating within the bin.
    pub fn percentile(&self, percent: f32) -> f32 {
        let _ = percent;
        todo!("agent B")
    }

    /// Mean and standard deviation, estimated from bin centres.
    pub fn mean_std(&self) -> (f32, f32) {
        todo!("agent B")
    }
}

impl StretchMode {
    /// The values that map to black and white. `low < high` unless the band is
    /// constant, when both are its value.
    pub fn range(&self, histogram: &Histogram) -> (f32, f32) {
        let _ = histogram;
        todo!("agent B")
    }
}
