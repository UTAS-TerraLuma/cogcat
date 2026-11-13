//! GPU statistics checked against histograms computed on the CPU. Skipped
//! (with a note) when no GPU adapter is available.

use std::time::{Duration, Instant};

use cogcat::cog::TileKey;
use cogcat::render::{StatsRequest, StatsResult, TileRenderer, headless_device};
use cogcat::stretch::Histogram;

fn renderer() -> Option<TileRenderer> {
    match headless_device() {
        Ok((device, queue)) => Some(TileRenderer::headless(device, queue)),
        Err(error) => {
            eprintln!("skipping GPU test: {error}");
            None
        }
    }
}

fn key(x: u32, y: u32) -> TileKey {
    TileKey { level: 1, x, y }
}

/// A small xorshift generator, so the tests need no extra crates.
struct Random(u64);

impl Random {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    /// Uniform in 0..1.
    fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32
    }
}

/// Random values from `low` to `high`, about one in ten NaN (or infinite).
fn random_band(random: &mut Random, len: usize, low: f32, high: f32) -> Vec<f32> {
    (0..len)
        .map(|_| match random.next() % 20 {
            0 => f32::NAN,
            1 => f32::INFINITY,
            _ => low + (high - low) * random.unit(),
        })
        .collect()
}

/// Waits for request `id`, failing after a generous timeout.
fn wait_for(renderer: &mut TileRenderer, id: u64) -> StatsResult {
    wait_for_all(renderer, &[id]).pop().expect("one result")
}

/// Waits for every request in `ids`, returning results in that order. One
/// poll can finish several requests, so results are kept until wanted.
fn wait_for_all(renderer: &mut TileRenderer, ids: &[u64]) -> Vec<StatsResult> {
    let start = Instant::now();
    let mut done: Vec<StatsResult> = Vec::new();
    loop {
        done.extend(renderer.poll_stats());
        if ids.iter().all(|id| done.iter().any(|r| r.id == *id)) {
            return ids
                .iter()
                .map(|id| {
                    let index = done.iter().position(|r| r.id == *id).expect("found above");
                    done.swap_remove(index)
                })
                .collect();
        }
        assert!(
            start.elapsed() < Duration::from_secs(60),
            "statistics never arrived"
        );
        std::thread::sleep(Duration::from_millis(1));
    }
}

/// The bin `stats.wgsl` puts `value` in.
fn bin_of(value: f32, min: f32, max: f32, bins: u32) -> usize {
    let half_range = max * 0.5 - min * 0.5;
    if half_range <= 0.0 {
        return 0;
    }
    let t = (value * 0.5 - min * 0.5) / half_range;
    ((t * bins as f32).max(0.0) as u32).min(bins - 1) as usize
}

/// The histogram of the finite values in `regions` of `tiles`, as the GPU
/// should compute it.
fn cpu_histogram(regions: &[(&[f32], u32, [u32; 4])], bins: u32) -> Histogram {
    let values = || {
        regions.iter().flat_map(|&(data, width, [x0, y0, x1, y1])| {
            (y0..y1).flat_map(move |y| (x0..x1).map(move |x| data[(y * width + x) as usize]))
        })
    };
    let finite = || values().filter(|v| v.is_finite());
    let min = finite().fold(f32::INFINITY, f32::min);
    let max = finite().fold(f32::NEG_INFINITY, f32::max);
    let mut counts = vec![0; bins as usize];
    for value in finite() {
        counts[bin_of(value, min, max, bins)] += 1;
    }
    let total = counts.iter().map(|&c| u64::from(c)).sum();
    if total == 0 {
        return Histogram {
            min: 0.0,
            max: 0.0,
            counts,
            total,
        };
    }
    Histogram {
        min,
        max,
        counts,
        total,
    }
}

/// Exact min, max and total; counts may differ only where a value sits on a
/// bin boundary and the GPU divides slightly differently.
fn assert_matches(gpu: &Histogram, cpu: &Histogram, what: &str) {
    assert_eq!(
        (gpu.min, gpu.max, gpu.total),
        (cpu.min, cpu.max, cpu.total),
        "{what}"
    );
    assert_eq!(gpu.counts.len(), cpu.counts.len(), "{what}");
    let moved: u64 = gpu
        .counts
        .iter()
        .zip(&cpu.counts)
        .map(|(&a, &b)| u64::from(a.abs_diff(b)))
        .sum();
    assert!(moved <= 4, "{what}: {moved} counts in different bins");
}

#[test]
fn stats_should_match_the_cpu_over_tiles_bands_and_rectangles() {
    let Some(mut renderer) = renderer() else {
        return;
    };
    let mut random = Random(0x1234_5678_9abc_def0);
    let size = [200, 150];
    let len = (size[0] * size[1]) as usize;
    // Three tiles of three bands with different ranges, including negatives.
    let ranges = [(-50.0, 50.0), (0.0, 1.0), (1000.0, 65535.0)];
    let mut data = Vec::new();
    for t in 0..3 {
        let bands: Vec<Vec<f32>> = ranges
            .iter()
            .map(|&(low, high)| random_band(&mut random, len, low, high))
            .collect();
        for (band, values) in bands.iter().enumerate() {
            renderer.upload_band(key(t, 0), band as u32, size, values);
        }
        data.push(bands);
    }
    // A whole tile, a sub-rectangle, and one running off the texture.
    let rects = [[0, 0, 200, 150], [13, 7, 101, 99], [150, 100, 400, 400]];
    let bins = 256;
    let request = StatsRequest {
        bands: vec![2, 0, 1],
        regions: (0..3).map(|t| (key(t, 0), rects[t as usize])).collect(),
        bins,
    };
    let id = renderer.start_stats(request);
    let result = wait_for(&mut renderer, id);

    assert_eq!(result.histograms.len(), 3);
    for (&(band, ref gpu), expected_band) in result.histograms.iter().zip([2, 0, 1]) {
        assert_eq!(band, expected_band);
        let regions: Vec<(&[f32], u32, [u32; 4])> = (0..3)
            .map(|t| {
                let [x0, y0, x1, y1] = rects[t];
                let clamped = [x0, y0, x1.min(size[0]), y1.min(size[1])];
                (data[t][band as usize].as_slice(), size[0], clamped)
            })
            .collect();
        let cpu = cpu_histogram(&regions, bins);
        assert_matches(gpu, &cpu, &format!("band {band}"));
    }
}

#[test]
fn stats_should_put_a_constant_band_in_bin_zero() {
    let Some(mut renderer) = renderer() else {
        return;
    };
    let mut values = vec![3.25; 64 * 64];
    values[17] = f32::NAN;
    renderer.upload_band(key(0, 0), 0, [64, 64], &values);
    let id = renderer.start_stats(StatsRequest {
        bands: vec![0],
        regions: vec![(key(0, 0), [0, 0, 64, 64])],
        bins: 16,
    });
    let (_, histogram) = &wait_for(&mut renderer, id).histograms[0];
    assert_eq!((histogram.min, histogram.max), (3.25, 3.25));
    assert_eq!(histogram.counts[0], 64 * 64 - 1);
    assert_eq!(histogram.total, 64 * 64 - 1);
}

#[test]
fn stats_should_be_empty_without_finite_texels_or_uploads() {
    let Some(mut renderer) = renderer() else {
        return;
    };
    renderer.upload_band(key(0, 0), 0, [8, 8], &[f32::NAN; 64]);
    let id = renderer.start_stats(StatsRequest {
        // Band 1 was never uploaded, and the empty rectangle covers nothing.
        bands: vec![0, 1],
        regions: vec![(key(0, 0), [0, 0, 8, 8]), (key(5, 5), [4, 4, 4, 8])],
        bins: 32,
    });
    let result = wait_for(&mut renderer, id);
    for (_, histogram) in &result.histograms {
        assert_eq!(histogram.total, 0);
        assert_eq!((histogram.min, histogram.max), (0.0, 0.0));
        assert_eq!(histogram.counts, vec![0; 32]);
    }
}

#[test]
fn stats_should_count_the_maximum_in_the_last_bin() {
    let Some(mut renderer) = renderer() else {
        return;
    };
    // Bins of width 1 over 0..3, away from the bin edges but for the maximum.
    renderer.upload_band(key(0, 0), 0, [4, 1], &[0.0, 1.4, 1.6, 3.0]);
    let id = renderer.start_stats(StatsRequest {
        bands: vec![0],
        regions: vec![(key(0, 0), [0, 0, 4, 1])],
        bins: 3,
    });
    let (_, histogram) = &wait_for(&mut renderer, id).histograms[0];
    assert_eq!(histogram.counts, vec![1, 2, 1]);
    assert_eq!(
        (histogram.min, histogram.max, histogram.total),
        (0.0, 3.0, 4)
    );
}

#[test]
fn stats_should_handle_many_regions_and_the_most_bins() {
    let Some(mut renderer) = renderer() else {
        return;
    };
    let mut random = Random(42);
    let size = [70, 70];
    let len = (size[0] * size[1]) as usize;
    let mut data = Vec::new();
    for t in 0..100 {
        let bands: Vec<Vec<f32>> = (0..3)
            .map(|b| random_band(&mut random, len, -(b as f32), t as f32 + 1.0))
            .collect();
        for (band, values) in bands.iter().enumerate() {
            renderer.upload_band(key(t % 10, t / 10), band as u32, size, values);
        }
        data.push(bands);
    }
    let request = StatsRequest {
        bands: vec![0, 1, 2],
        regions: (0..100)
            .map(|t| (key(t % 10, t / 10), [0, 0, 70, 70]))
            .collect(),
        // More than the most, so clamped to 4096.
        bins: 10_000,
    };
    let id = renderer.start_stats(request);
    let result = wait_for(&mut renderer, id);
    for (band, gpu) in &result.histograms {
        let regions: Vec<(&[f32], u32, [u32; 4])> = data
            .iter()
            .map(|bands| (bands[*band as usize].as_slice(), size[0], [0, 0, 70, 70]))
            .collect();
        let cpu = cpu_histogram(&regions, 4096);
        assert_matches(gpu, &cpu, &format!("band {band}"));
    }
}

#[test]
fn stats_should_answer_requests_in_flight_together() {
    let Some(mut renderer) = renderer() else {
        return;
    };
    renderer.upload_band(key(0, 0), 0, [2, 1], &[1.0, 2.0]);
    renderer.upload_band(key(0, 0), 1, [2, 1], &[-5.0, 5.0]);
    let first = renderer.start_stats(StatsRequest {
        bands: vec![0],
        regions: vec![(key(0, 0), [0, 0, 2, 1])],
        bins: 8,
    });
    let second = renderer.start_stats(StatsRequest {
        bands: vec![1],
        regions: vec![(key(0, 0), [0, 0, 2, 1])],
        bins: 8,
    });
    assert_ne!(first, second);
    assert_eq!(renderer.pending_stats(), 2);
    let results = wait_for_all(&mut renderer, &[first, second]);
    let (_, a) = &results[0].histograms[0];
    let (_, b) = &results[1].histograms[0];
    assert_eq!((a.min, a.max), (1.0, 2.0));
    assert_eq!((b.min, b.max), (-5.0, 5.0));
    assert_eq!(renderer.pending_stats(), 0);
}

#[test]
fn stats_should_finish_after_the_textures_are_evicted() {
    let Some(mut renderer) = renderer() else {
        return;
    };
    renderer.upload_band(key(0, 0), 0, [16, 16], &[7.0; 256]);
    let id = renderer.start_stats(StatsRequest {
        bands: vec![0],
        regions: vec![(key(0, 0), [0, 0, 16, 16])],
        bins: 8,
    });
    renderer.clear();
    let (_, histogram) = &wait_for(&mut renderer, id).histograms[0];
    assert_eq!(histogram.total, 256);
}

/// Times 50 tiles × 3 bands of 512² texels, as a viewer might ask for on a
/// full screen of tiles. `cargo test --release -- --ignored --nocapture`.
#[test]
#[ignore = "timing only"]
fn stats_timing_for_fifty_tiles_of_three_bands() {
    let Some(mut renderer) = renderer() else {
        return;
    };
    let mut random = Random(7);
    let size = [512, 512];
    let values = random_band(&mut random, 512 * 512, 0.0, 10_000.0);
    for t in 0..50 {
        for band in 0..3 {
            renderer.upload_band(key(t, 0), band, size, &values);
        }
    }
    let request = || StatsRequest {
        bands: vec![0, 1, 2],
        regions: (0..50).map(|t| (key(t, 0), [0, 0, 512, 512])).collect(),
        bins: 1024,
    };
    // The first request also compiles the shaders on some drivers.
    let id = renderer.start_stats(request());
    wait_for(&mut renderer, id);
    let start = Instant::now();
    let id = renderer.start_stats(request());
    let started = start.elapsed();
    wait_for(&mut renderer, id);
    eprintln!(
        "50 tiles × 3 bands of 512²: started in {started:?}, finished in {:?}",
        start.elapsed()
    );
}
