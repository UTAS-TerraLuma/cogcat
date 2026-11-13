//! Drawing tiles with wgpu, and statistics on the GPU.
//!
//! The GPU holds one `R32Float` texture per (tile, band), for the bands the
//! current composite needs; the CPU cache in the loader holds every band.
//! Tiles are drawn into an offscreen texture that egui shows, as in Copcat.

mod colormap;
mod stats;

use std::collections::HashSet;

use eframe::egui;
use eframe::egui_wgpu::{self, wgpu};

pub use colormap::Colormap;
pub use stats::{StatsRequest, StatsResult};

use crate::cog::TileKey;

/// How to turn band values into colours.
#[derive(Clone, Debug, PartialEq)]
pub struct DisplayStyle {
    /// The band for red, green and blue. With `colormap`, only `bands[0]` is used.
    pub bands: [u32; 3],
    /// Draw `bands[0]` alone through this colour map.
    pub colormap: Option<Colormap>,
    /// Per channel, the value that maps to 0 and to 1.
    pub low: [f32; 3],
    pub high: [f32; 3],
    /// Applied after stretching: `out = stretched ^ (1 / gamma)`.
    pub gamma: f32,
    /// Shown where no tile is drawn, and under transparent (NaN) pixels.
    /// Straight RGBA, 0–1, as written to the 8-bit target.
    pub background: [f32; 4],
}

/// One tile to draw this frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TileDraw {
    pub key: TileKey,
    /// The tile's valid area (`TileData::valid`) on screen, in clip space
    /// (x right, y up, −1 to 1): top-left, top-right, bottom-right,
    /// bottom-left of the image region. Not necessarily axis aligned.
    pub corners: [[f32; 2]; 4],
    /// The valid area's size in texels, which the corners map to.
    pub texels: [f32; 2],
    /// Blended over the tile's colour by its alpha (debug "tint by level").
    pub tint: [f32; 4],
}

/// An image read back from the GPU, straight RGBA8, rows top to bottom.
pub struct Image {
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
}

/// An offscreen target for one view, and its egui texture.
#[derive(Default)]
pub struct Viewport {
    // Agent B: the target texture, its size, egui texture id, ...
}

pub struct TileRenderer {
    // Agent B.
}

impl TileRenderer {
    /// A renderer for the viewer window, whose images egui can show.
    pub fn new(render_state: &egui_wgpu::RenderState) -> Self {
        let _ = render_state;
        todo!("agent B")
    }

    /// A renderer without a window, for tests and offscreen use.
    pub fn headless(device: wgpu::Device, queue: wgpu::Queue) -> Self {
        let _ = (device, queue);
        todo!("agent B")
    }

    pub fn has_band(&self, key: TileKey, band: u32) -> bool {
        let _ = (key, band);
        todo!("agent B")
    }

    /// Uploads one band of a tile: `size` texels, row-major, NaN for no data.
    /// Replaces any texture already there.
    pub fn upload_band(&mut self, key: TileKey, band: u32, size: [u32; 2], data: &[f32]) {
        let _ = (key, band, size, data);
        todo!("agent B")
    }

    /// Bytes of tile textures on the GPU.
    pub fn resident_bytes(&self) -> u64 {
        todo!("agent B")
    }

    /// Frees least recently drawn textures until at most `max_bytes` remain,
    /// never those of tiles in `keep`. Returns what was freed.
    pub fn evict_to_budget(
        &mut self,
        max_bytes: u64,
        keep: &HashSet<TileKey>,
    ) -> Vec<(TileKey, u32)> {
        let _ = (max_bytes, keep);
        todo!("agent B")
    }

    /// Frees every tile texture (when another file opens).
    pub fn clear(&mut self) {
        todo!("agent B")
    }

    /// Draws `tiles` in order (later ones on top) into `viewport` at `size`
    /// physical pixels, skipping any whose bands for `style` aren't all
    /// uploaded. Returns the texture for egui to show.
    pub fn render(
        &mut self,
        viewport: &mut Viewport,
        size: [u32; 2],
        tiles: &[TileDraw],
        style: &DisplayStyle,
    ) -> egui::TextureId {
        let _ = (viewport, size, tiles, style);
        todo!("agent B")
    }

    /// Draws like `render`, then reads the image back, waiting for the GPU.
    pub fn capture(
        &mut self,
        viewport: &mut Viewport,
        size: [u32; 2],
        tiles: &[TileDraw],
        style: &DisplayStyle,
    ) -> Result<Image, String> {
        let _ = (viewport, size, tiles, style);
        todo!("agent B")
    }

    /// Starts computing histograms on the GPU. Returns at once; the result
    /// comes from `poll_stats` a frame or more later. Returns the request id.
    pub fn start_stats(&mut self, request: StatsRequest) -> u64 {
        let _ = request;
        todo!("agent B")
    }

    /// Finished statistics, without blocking. Call once per frame.
    pub fn poll_stats(&mut self) -> Vec<StatsResult> {
        todo!("agent B")
    }
}
