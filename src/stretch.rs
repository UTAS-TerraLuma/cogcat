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
    ///
    /// Values are taken to be spread evenly across each bin, so 0 gives
    /// `min` and 100 gives `max` (the first and last bins always hold those).
    /// An empty histogram gives `min`.
    pub fn percentile(&self, percent: f32) -> f32 {
        if self.total == 0 || self.counts.is_empty() || self.max <= self.min {
            return self.min;
        }
        let target = f64::from(percent.clamp(0.0, 100.0)) / 100.0 * self.total as f64;
        let width = (f64::from(self.max) - f64::from(self.min)) / self.counts.len() as f64;
        let mut below = 0.0;
        for (i, &count) in self.counts.iter().enumerate() {
            let count = f64::from(count);
            // The first bin whose pixels reach the target. Empty bins are
            // skipped, so a percentile never lands in a gap between values.
            if count > 0.0 && below + count >= target {
                let fraction = ((target - below) / count).clamp(0.0, 1.0);
                let value = f64::from(self.min) + width * (i as f64 + fraction);
                return (value as f32).clamp(self.min, self.max);
            }
            below += count;
        }
        // Only reached if `total` is more than the counts add up to.
        self.max
    }

    /// Mean and standard deviation, estimated from bin centres.
    ///
    /// An empty histogram gives `(min, 0)`, and a constant band its value and 0.
    pub fn mean_std(&self) -> (f32, f32) {
        let total: f64 = self.counts.iter().map(|&count| f64::from(count)).sum();
        if total == 0.0 || self.max <= self.min {
            return (self.min, 0.0);
        }
        let width = (f64::from(self.max) - f64::from(self.min)) / self.counts.len() as f64;
        let centre = |i: usize| f64::from(self.min) + width * (i as f64 + 0.5);
        let weighted = || {
            self.counts
                .iter()
                .enumerate()
                .map(move |(i, &count)| (centre(i), f64::from(count)))
        };
        let mean = weighted().map(|(x, count)| x * count).sum::<f64>() / total;
        let variance = weighted()
            .map(|(x, count)| (x - mean).powi(2) * count)
            .sum::<f64>()
            / total;
        (mean as f32, variance.sqrt() as f32)
    }
}

impl Default for StretchMode {
    /// A 2–98% percentile clip, which suits most imagery.
    fn default() -> Self {
        StretchMode::Percentile {
            low: 2.0,
            high: 98.0,
        }
    }
}

impl StretchMode {
    /// ENVI's optimized linear stretch with its default parameters.
    pub const OPTIMIZED_LINEAR: StretchMode = StretchMode::OptimizedLinear {
        low: 2.5,
        high: 99.0,
        low_adjust: 0.1,
        high_adjust: 0.5,
    };

    /// The values that map to black and white. `low < high` unless the band is
    /// constant, when both are its value.
    ///
    /// - A band with no finite pixels gives the histogram's `min` and `max`
    ///   (both 0 from the GPU statistics).
    /// - If a percentile or standard deviation stretch collapses to a single
    ///   value (equal percentiles, or no spread), it falls back to min/max.
    /// - Optimized linear is not clamped to the data range, as in ENVI: its
    ///   widened ends can go past the minimum and maximum, which darkens the
    ///   darkest pixels less and brightens the brightest less.
    /// - Standard deviations are clamped to the data range.
    pub fn range(&self, histogram: &Histogram) -> (f32, f32) {
        let (min, max) = (histogram.min, histogram.max);
        if histogram.total == 0 || max <= min {
            return (min, max);
        }
        let (low, high) = match *self {
            StretchMode::MinMax => (min, max),
            StretchMode::Percentile { low, high } => {
                (histogram.percentile(low), histogram.percentile(high))
            }
            StretchMode::OptimizedLinear {
                low,
                high,
                low_adjust,
                high_adjust,
            } => {
                let a = histogram.percentile(low);
                let b = histogram.percentile(high);
                let spread = b - a;
                (a - low_adjust * spread, b + high_adjust * spread)
            }
            StretchMode::StdDev { n } => {
                let (mean, std) = histogram.mean_std();
                let n = n.abs();
                (
                    (mean - n * std).clamp(min, max),
                    (mean + n * std).clamp(min, max),
                )
            }
        };
        if low < high { (low, high) } else { (min, max) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A histogram of `values` with `bins` bins, binned as the GPU does.
    fn histogram_of(values: &[f32], bins: usize) -> Histogram {
        let min = values.iter().copied().fold(f32::INFINITY, f32::min);
        let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut counts = vec![0; bins];
        for &v in values {
            let bin = if max > min {
                (((v - min) / (max - min) * bins as f32) as usize).min(bins - 1)
            } else {
                0
            };
            counts[bin] += 1;
        }
        Histogram {
            min,
            max,
            counts,
            total: values.len() as u64,
        }
    }

    /// 0, 1, 2, ..., 1000: evenly spread, so percentile p is about 10·p.
    fn ramp() -> Histogram {
        let values: Vec<f32> = (0..=1000).map(|i| i as f32).collect();
        histogram_of(&values, 1000)
    }

    fn close(a: f32, b: f32, tolerance: f32) -> bool {
        (a - b).abs() <= tolerance
    }

    #[test]
    fn percentile_should_give_min_at_0_and_max_at_100() {
        let histogram = histogram_of(&[3.0, 5.0, 5.5, 9.0, 12.0], 64);
        assert_eq!(histogram.percentile(0.0), 3.0);
        assert_eq!(histogram.percentile(100.0), 12.0);
    }

    #[test]
    fn percentile_should_clamp_out_of_range_percents() {
        let histogram = ramp();
        assert_eq!(histogram.percentile(-5.0), 0.0);
        assert_eq!(histogram.percentile(150.0), 1000.0);
    }

    #[test]
    fn percentile_should_interpolate_on_an_even_spread() {
        let histogram = ramp();
        for percent in [2.0, 25.0, 50.0, 98.0] {
            let value = histogram.percentile(percent);
            assert!(close(value, percent * 10.0, 1.5), "{percent}% gave {value}");
        }
    }

    #[test]
    fn percentile_should_interpolate_within_a_bin() {
        // Two bins of 50 over 0..10: the 25th percentile is half way into bin 0.
        let histogram = Histogram {
            min: 0.0,
            max: 10.0,
            counts: vec![50, 50],
            total: 100,
        };
        assert!(close(histogram.percentile(25.0), 2.5, 1e-5));
        assert!(close(histogram.percentile(75.0), 7.5, 1e-5));
    }

    #[test]
    fn percentile_should_skip_empty_bins() {
        // Half the pixels at 0, half at 10, nothing between.
        let histogram = Histogram {
            min: 0.0,
            max: 10.0,
            counts: vec![50, 0, 0, 0, 50],
            total: 100,
        };
        // Exactly half lands at the end of the first bin, not in the gap.
        assert!(close(histogram.percentile(50.0), 2.0, 1e-5));
        assert!(histogram.percentile(50.1) >= 8.0);
    }

    #[test]
    fn percentile_should_be_monotonic() {
        let values: Vec<f32> = (0..500).map(|i| ((i * 37) % 101) as f32 * 0.3).collect();
        let histogram = histogram_of(&values, 37);
        let mut previous = f32::NEG_INFINITY;
        for step in 0..=200 {
            let value = histogram.percentile(step as f32 / 2.0);
            assert!(value >= previous);
            previous = value;
        }
    }

    #[test]
    fn percentile_should_give_min_for_an_empty_histogram() {
        let histogram = Histogram {
            min: 0.0,
            max: 0.0,
            counts: vec![0; 16],
            total: 0,
        };
        assert_eq!(histogram.percentile(50.0), 0.0);
    }

    #[test]
    fn mean_std_should_match_the_data() {
        let values: Vec<f32> = (0..10_000).map(|i| (i % 100) as f32).collect();
        let histogram = histogram_of(&values, 1000);
        let (mean, std) = histogram.mean_std();
        // 0..99 evenly: mean 49.5, std sqrt((100² − 1) / 12) ≈ 28.866.
        assert!(close(mean, 49.5, 0.1), "mean {mean}");
        assert!(close(std, 28.866, 0.1), "std {std}");
    }

    #[test]
    fn mean_std_should_give_zero_spread_for_a_constant_band() {
        let histogram = histogram_of(&[4.0; 20], 256);
        assert_eq!(histogram.mean_std(), (4.0, 0.0));
    }

    #[test]
    fn min_max_should_give_the_data_range() {
        let histogram = histogram_of(&[-2.0, 1.0, 7.5], 16);
        assert_eq!(StretchMode::MinMax.range(&histogram), (-2.0, 7.5));
    }

    #[test]
    fn percentile_stretch_should_clip_the_tails() {
        let (low, high) = StretchMode::default().range(&ramp());
        assert!(close(low, 20.0, 1.5) && close(high, 980.0, 1.5));
    }

    #[test]
    fn default_should_be_a_2_98_percentile_clip() {
        assert_eq!(
            StretchMode::default(),
            StretchMode::Percentile {
                low: 2.0,
                high: 98.0
            }
        );
    }

    #[test]
    fn optimized_linear_should_widen_the_percentiles() {
        let histogram = ramp();
        let (low, high) = StretchMode::OPTIMIZED_LINEAR.range(&histogram);
        let a = histogram.percentile(2.5);
        let b = histogram.percentile(99.0);
        assert!(close(low, a - 0.1 * (b - a), 1e-3));
        assert!(close(high, b + 0.5 * (b - a), 1e-3));
        // Not clamped: the high end goes well past the maximum, as in ENVI.
        assert!(low < 0.0 && high > 1000.0);
    }

    #[test]
    fn std_dev_should_be_centred_on_the_mean_and_clamped() {
        let histogram = ramp();
        let (mean, std) = histogram.mean_std();
        let (low, high) = StretchMode::StdDev { n: 1.0 }.range(&histogram);
        assert!(close(low, mean - std, 1e-3) && close(high, mean + std, 1e-3));
        // Three deviations of an even spread reach past both ends.
        assert_eq!(
            StretchMode::StdDev { n: 3.0 }.range(&histogram),
            (0.0, 1000.0)
        );
    }

    #[test]
    fn range_should_give_the_value_twice_for_a_constant_band() {
        let histogram = histogram_of(&[42.0; 100], 1024);
        let modes = [
            StretchMode::MinMax,
            StretchMode::default(),
            StretchMode::OPTIMIZED_LINEAR,
            StretchMode::StdDev { n: 2.0 },
        ];
        for mode in modes {
            assert_eq!(mode.range(&histogram), (42.0, 42.0), "{mode:?}");
        }
    }

    #[test]
    fn range_should_fall_back_to_min_max_when_percentiles_meet() {
        let histogram = ramp();
        let mode = StretchMode::Percentile {
            low: 50.0,
            high: 50.0,
        };
        assert_eq!(mode.range(&histogram), (0.0, 1000.0));
    }

    #[test]
    fn range_should_keep_low_below_high_on_skewed_data() {
        // Mostly zeros with a long tail, as in a water-heavy scene.
        let mut values = vec![0.0; 9_000];
        values.extend((0..1_000).map(|i| i as f32));
        let histogram = histogram_of(&values, 1024);
        for mode in [
            StretchMode::default(),
            StretchMode::OPTIMIZED_LINEAR,
            StretchMode::StdDev { n: 2.0 },
        ] {
            let (low, high) = mode.range(&histogram);
            assert!(low < high, "{mode:?} gave {low}..{high}");
        }
    }
}
