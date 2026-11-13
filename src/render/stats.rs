//! Histograms computed on the GPU: a min/max pass, then a histogram pass,
//! over chosen texel rectangles of uploaded (tile, band) textures.

use crate::cog::TileKey;
use crate::stretch::Histogram;

pub struct StatsRequest {
    /// The bands to compute, each giving one histogram.
    pub bands: Vec<u32>,
    /// Which texels of which tiles to include: `[x0, y0, x1, y1)` in texels.
    /// Every band in `bands` must be uploaded for these tiles; any that
    /// aren't are skipped.
    pub regions: Vec<(TileKey, [u32; 4])>,
    /// Histogram bins per band (e.g. 1024).
    pub bins: u32,
}

pub struct StatsResult {
    pub id: u64,
    /// One per requested band, in order. `total` is 0 if no finite texel was seen.
    pub histograms: Vec<(u32, Histogram)>,
}
