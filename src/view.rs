//! The 2D view: a camera over the image's own CRS, which overview to draw at,
//! and which tiles that needs.
//!
//! World coordinates are the file's CRS units (x east, y north). A file without
//! geo tags gets a stand-in transform with one unit per pixel and y up, so the
//! maths below is the same either way. Positions stay in `f64` until they're
//! relative to the camera, so UTM-sized coordinates don't lose precision.

use std::collections::BTreeSet;

use crate::cog::{CogInfo, GeoTransform, Level, TileKey};

/// Pixel (col, row) to world: the file's transform, or y-up pixels without one.
pub fn world_transform(info: &CogInfo) -> GeoTransform {
    info.geo
        .unwrap_or(GeoTransform([0.0, 1.0, 0.0, 0.0, 0.0, -1.0]))
}

/// The world size of one full-resolution pixel (the square root of the
/// transform's area scale, so rotated and non-square pixels get a fair size).
pub fn pixel_size(transform: &GeoTransform) -> f64 {
    let c = &transform.0;
    (c[1] * c[5] - c[2] * c[4]).abs().sqrt()
}

/// A camera looking straight down at the world.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Camera {
    /// The world point at the middle of the view.
    pub center: [f64; 2],
    /// World units per physical screen pixel.
    pub units_per_px: f64,
}

impl Camera {
    /// Fits the whole image into a view of `size_px`, with a small margin.
    pub fn fit(info: &CogInfo, size_px: [f32; 2]) -> Self {
        let transform = world_transform(info);
        let (min, max) = image_bounds(&transform, info.width as f64, info.height as f64);
        let [w, h] = size_px.map(|side| f64::from(side.max(1.0)));
        let units_per_px = ((max[0] - min[0]) / w).max((max[1] - min[1]) / h) * 1.05;
        Self {
            center: [(min[0] + max[0]) / 2.0, (min[1] + max[1]) / 2.0],
            units_per_px: if units_per_px > 0.0 {
                units_per_px
            } else {
                1.0
            },
        }
    }

    /// A screen position (physical pixels from the view's top-left) in the world.
    pub fn screen_to_world(&self, size_px: [f32; 2], pos_px: [f32; 2]) -> [f64; 2] {
        let dx = f64::from(pos_px[0] - size_px[0] / 2.0);
        let dy = f64::from(pos_px[1] - size_px[1] / 2.0);
        [
            self.center[0] + dx * self.units_per_px,
            self.center[1] - dy * self.units_per_px,
        ]
    }

    /// A world point in physical pixels from the view's top-left.
    pub fn world_to_screen(&self, size_px: [f32; 2], world: [f64; 2]) -> [f32; 2] {
        let dx = (world[0] - self.center[0]) / self.units_per_px;
        let dy = (world[1] - self.center[1]) / self.units_per_px;
        [
            (dx + f64::from(size_px[0]) / 2.0) as f32,
            (f64::from(size_px[1]) / 2.0 - dy) as f32,
        ]
    }

    /// A world point in clip space (−1 to 1, y up).
    pub fn world_to_clip(&self, size_px: [f32; 2], world: [f64; 2]) -> [f32; 2] {
        let half_w = f64::from(size_px[0].max(1.0)) / 2.0 * self.units_per_px;
        let half_h = f64::from(size_px[1].max(1.0)) / 2.0 * self.units_per_px;
        [
            ((world[0] - self.center[0]) / half_w) as f32,
            ((world[1] - self.center[1]) / half_h) as f32,
        ]
    }

    /// Moves the view by a screen drag of `delta_px` physical pixels.
    pub fn pan(&mut self, delta_px: [f32; 2]) {
        self.center[0] -= f64::from(delta_px[0]) * self.units_per_px;
        self.center[1] += f64::from(delta_px[1]) * self.units_per_px;
    }

    /// Zooms by `factor` (> 1 zooms in) keeping the world point under
    /// `anchor_px` where it is.
    pub fn zoom_at(&mut self, size_px: [f32; 2], anchor_px: [f32; 2], factor: f64) {
        if !(factor.is_finite() && factor > 0.0) {
            return;
        }
        let before = self.screen_to_world(size_px, anchor_px);
        self.units_per_px /= factor;
        let after = self.screen_to_world(size_px, anchor_px);
        self.center[0] += before[0] - after[0];
        self.center[1] += before[1] - after[1];
    }

    /// The world bounding box of a view of `size_px`.
    pub fn world_bounds(&self, size_px: [f32; 2]) -> ([f64; 2], [f64; 2]) {
        let half = [
            f64::from(size_px[0]) / 2.0 * self.units_per_px,
            f64::from(size_px[1]) / 2.0 * self.units_per_px,
        ];
        (
            [self.center[0] - half[0], self.center[1] - half[1]],
            [self.center[0] + half[0], self.center[1] + half[1]],
        )
    }
}

/// The world bounding box of a `cols` × `rows` pixel region from (0, 0).
fn image_bounds(transform: &GeoTransform, cols: f64, rows: f64) -> ([f64; 2], [f64; 2]) {
    bounds_of(
        [[0.0, 0.0], [cols, 0.0], [cols, rows], [0.0, rows]]
            .map(|[c, r]| transform.pixel_to_crs(c, r)),
    )
}

fn bounds_of(points: [[f64; 2]; 4]) -> ([f64; 2], [f64; 2]) {
    let mut min = [f64::INFINITY; 2];
    let mut max = [f64::NEG_INFINITY; 2];
    for point in points {
        for axis in 0..2 {
            min[axis] = min[axis].min(point[axis]);
            max[axis] = max[axis].max(point[axis]);
        }
    }
    (min, max)
}

/// A tile's valid area in full-resolution pixels: `[col0, row0, col1, row1)`.
pub fn tile_pixel_rect(level: &Level, key: TileKey) -> [f64; 4] {
    let [x0, y0, x1, y1] = tile_level_rect(level, key);
    [
        f64::from(x0) * level.scale[0],
        f64::from(y0) * level.scale[1],
        f64::from(x1) * level.scale[0],
        f64::from(y1) * level.scale[1],
    ]
}

/// A tile's valid area in its own level's pixels: `[x0, y0, x1, y1)`.
pub fn tile_level_rect(level: &Level, key: TileKey) -> [u32; 4] {
    let x0 = key.x * level.tile_width;
    let y0 = key.y * level.tile_height;
    [
        x0,
        y0,
        (x0 + level.tile_width).min(level.width),
        (y0 + level.tile_height).min(level.height),
    ]
}

/// A tile's valid area's corners in the world: TL, TR, BR, BL (in pixel
/// terms; the transform may flip or rotate them).
pub fn tile_world_corners(transform: &GeoTransform, level: &Level, key: TileKey) -> [[f64; 2]; 4] {
    let [c0, r0, c1, r1] = tile_pixel_rect(level, key);
    [[c0, r0], [c1, r0], [c1, r1], [c0, r1]].map(|[c, r]| transform.pixel_to_crs(c, r))
}

/// What a view needs: the overview to draw at and its tiles.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Visible {
    pub level: u8,
    /// The level's tiles that overlap the view, nearest the centre first.
    pub tiles: Vec<TileKey>,
    /// The view in that level's pixels, `[x0, y0, x1, y1)`, clipped to the image.
    pub level_rect: [f64; 4],
}

/// The level to draw at: the coarsest whose pixels are no bigger than
/// `lod_bias` screen pixels. A bias above 1 trades sharpness for fewer tiles.
pub fn choose_level(
    levels: &[Level],
    full_pixel_size: f64,
    units_per_px: f64,
    lod_bias: f64,
) -> u8 {
    let mut chosen = 0;
    for level in levels {
        let level_pixel = full_pixel_size * (level.scale[0] * level.scale[1]).sqrt();
        if level_pixel <= units_per_px * lod_bias * 1.0001 {
            chosen = level.index;
        }
    }
    chosen
}

/// The tiles of `level` that a view covers, nearest its centre first.
pub fn visible_tiles(
    transform: &GeoTransform,
    level: &Level,
    camera: &Camera,
    size_px: [f32; 2],
) -> Visible {
    let empty = Visible {
        level: level.index,
        ..Visible::default()
    };
    let (min, max) = camera.world_bounds(size_px);
    let corners = [
        [min[0], min[1]],
        [max[0], min[1]],
        [max[0], max[1]],
        [min[0], max[1]],
    ];
    let mut pixels = [[0.0; 2]; 4];
    for (pixel, world) in pixels.iter_mut().zip(corners) {
        match transform.crs_to_pixel(world[0], world[1]) {
            Some([c, r]) => *pixel = [c / level.scale[0], r / level.scale[1]],
            None => return empty,
        }
    }
    let (lo, hi) = bounds_of(pixels);
    let x0 = lo[0].max(0.0);
    let y0 = lo[1].max(0.0);
    let x1 = hi[0].min(f64::from(level.width));
    let y1 = hi[1].min(f64::from(level.height));
    if x0 >= x1 || y0 >= y1 {
        return empty;
    }
    let tx0 = (x0 / f64::from(level.tile_width)).floor() as u32;
    let ty0 = (y0 / f64::from(level.tile_height)).floor() as u32;
    let tx1 = ((x1 / f64::from(level.tile_width)).ceil() as u32).min(level.tiles_across);
    let ty1 = ((y1 / f64::from(level.tile_height)).ceil() as u32).min(level.tiles_down);
    let centre = [(x0 + x1) / 2.0, (y0 + y1) / 2.0];
    let mut tiles: Vec<(f64, TileKey)> = (ty0..ty1)
        .flat_map(|y| (tx0..tx1).map(move |x| (x, y)))
        .map(|(x, y)| {
            let key = TileKey {
                level: level.index,
                x,
                y,
            };
            let cx = (f64::from(x) + 0.5) * f64::from(level.tile_width) - centre[0];
            let cy = (f64::from(y) + 0.5) * f64::from(level.tile_height) - centre[1];
            (cx * cx + cy * cy, key)
        })
        .collect();
    tiles.sort_by(|a, b| a.0.total_cmp(&b.0));
    Visible {
        level: level.index,
        tiles: tiles.into_iter().map(|(_, key)| key).collect(),
        level_rect: [x0, y0, x1, y1],
    }
}

/// The tiles of `coarser` that cover any part of `key` (from a finer level).
pub fn covering_tiles(levels: &[Level], key: TileKey, coarser: u8) -> Vec<TileKey> {
    let (Some(fine), Some(coarse)) = (
        levels.get(usize::from(key.level)),
        levels.get(usize::from(coarser)),
    ) else {
        return Vec::new();
    };
    let [c0, r0, c1, r1] = tile_pixel_rect(fine, key);
    tiles_in_pixel_rect(coarse, [c0, r0, c1, r1])
}

/// The tiles of `level` overlapping a full-resolution pixel rectangle.
pub fn tiles_in_pixel_rect(level: &Level, [c0, r0, c1, r1]: [f64; 4]) -> Vec<TileKey> {
    let span = |lo: f64, hi: f64, scale: f64, tile: u32, count: u32| {
        let first = (lo / scale / f64::from(tile)).floor().max(0.0) as u32;
        // Exclusive end; nudge in so a shared edge doesn't count.
        let last = ((hi / scale - 1e-6) / f64::from(tile)).floor().max(0.0) as u32;
        first.min(count)..(last + 1).min(count)
    };
    let xs = span(c0, c1, level.scale[0], level.tile_width, level.tiles_across);
    let ys = span(r0, r1, level.scale[1], level.tile_height, level.tiles_down);
    ys.flat_map(|y| {
        xs.clone().map(move |x| TileKey {
            level: level.index,
            x,
            y,
        })
    })
    .collect()
}

/// The full-resolution pixel under a world point, if it's inside the image.
pub fn world_to_pixel(info: &CogInfo, world: [f64; 2]) -> Option<[f64; 2]> {
    let [c, r] = world_transform(info).crs_to_pixel(world[0], world[1])?;
    (c >= 0.0 && r >= 0.0 && c < f64::from(info.width) && r < f64::from(info.height))
        .then_some([c, r])
}

/// Which tile of `level` holds a full-resolution pixel, and where in it.
pub fn locate_pixel(level: &Level, pixel: [f64; 2]) -> Option<(TileKey, [u32; 2])> {
    let x = (pixel[0] / level.scale[0]).floor();
    let y = (pixel[1] / level.scale[1]).floor();
    if x < 0.0 || y < 0.0 || x >= f64::from(level.width) || y >= f64::from(level.height) {
        return None;
    }
    let (x, y) = (x as u32, y as u32);
    Some((
        TileKey {
            level: level.index,
            x: x / level.tile_width,
            y: y / level.tile_height,
        },
        [x % level.tile_width, y % level.tile_height],
    ))
}

/// The coarsest level with at most `max_tiles` tiles, for whole-image
/// statistics. Falls back to the coarsest level.
pub fn stats_level(levels: &[Level], max_tiles: u32) -> u8 {
    levels
        .iter()
        .find(|level| level.tiles_across * level.tiles_down <= max_tiles)
        .or(levels.last())
        .map_or(0, |level| level.index)
}

/// Every tile of a level.
pub fn all_tiles(level: &Level) -> Vec<TileKey> {
    (0..level.tiles_down)
        .flat_map(|y| {
            (0..level.tiles_across).map(move |x| TileKey {
                level: level.index,
                x,
                y,
            })
        })
        .collect()
}

/// Tiles of levels coarser than `level` that cover any of `missing`, coarsest
/// first, so they can be drawn under the finer tiles while those load.
pub fn fallback_tiles(levels: &[Level], missing: &[TileKey], level: u8) -> Vec<TileKey> {
    let mut out = Vec::new();
    for coarser in (usize::from(level) + 1..levels.len()).rev() {
        let set: BTreeSet<TileKey> = missing
            .iter()
            .flat_map(|&key| covering_tiles(levels, key, coarser as u8))
            .collect();
        out.extend(set);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn level(index: u8, width: u32, height: u32, tile: u32, full: [u32; 2]) -> Level {
        Level {
            index,
            width,
            height,
            tile_width: tile,
            tile_height: tile,
            tiles_across: width.div_ceil(tile),
            tiles_down: height.div_ceil(tile),
            scale: [
                f64::from(full[0]) / f64::from(width),
                f64::from(full[1]) / f64::from(height),
            ],
        }
    }

    fn levels() -> Vec<Level> {
        vec![
            level(0, 1000, 600, 256, [1000, 600]),
            level(1, 500, 300, 256, [1000, 600]),
            level(2, 250, 150, 256, [1000, 600]),
        ]
    }

    const UTM: GeoTransform = GeoTransform([500_000.0, 10.0, 0.0, 5_300_000.0, 0.0, -10.0]);

    #[test]
    fn screen_and_world_should_round_trip() {
        let camera = Camera {
            center: [500_123.0, 5_299_000.0],
            units_per_px: 2.5,
        };
        let size = [800.0, 600.0];
        let world = camera.screen_to_world(size, [100.0, 50.0]);
        let back = camera.world_to_screen(size, world);
        assert!((back[0] - 100.0).abs() < 1e-3 && (back[1] - 50.0).abs() < 1e-3);
    }

    #[test]
    fn world_to_clip_should_put_the_centre_at_zero_and_the_top_right_at_one() {
        let camera = Camera {
            center: [10.0, 20.0],
            units_per_px: 1.0,
        };
        assert_eq!(
            camera.world_to_clip([200.0, 100.0], [10.0, 20.0]),
            [0.0, 0.0]
        );
        assert_eq!(
            camera.world_to_clip([200.0, 100.0], [110.0, 70.0]),
            [1.0, 1.0]
        );
    }

    #[test]
    fn zoom_at_should_keep_the_anchor_still() {
        let mut camera = Camera {
            center: [0.0, 0.0],
            units_per_px: 1.0,
        };
        let size = [400.0, 300.0];
        let before = camera.screen_to_world(size, [50.0, 60.0]);
        camera.zoom_at(size, [50.0, 60.0], 2.0);
        let after = camera.screen_to_world(size, [50.0, 60.0]);
        assert!((before[0] - after[0]).abs() < 1e-9 && (before[1] - after[1]).abs() < 1e-9);
        assert_eq!(camera.units_per_px, 0.5);
    }

    #[test]
    fn choose_level_should_pick_full_resolution_when_zoomed_in() {
        assert_eq!(choose_level(&levels(), 10.0, 5.0, 1.0), 0);
    }

    #[test]
    fn choose_level_should_pick_the_coarsest_level_no_bigger_than_a_screen_pixel() {
        assert_eq!(choose_level(&levels(), 10.0, 20.0, 1.0), 1);
        assert_eq!(choose_level(&levels(), 10.0, 39.0, 1.0), 1);
        assert_eq!(choose_level(&levels(), 10.0, 40.0, 1.0), 2);
        assert_eq!(choose_level(&levels(), 10.0, 1000.0, 1.0), 2);
    }

    #[test]
    fn visible_tiles_should_cover_the_view_centre_first() {
        let levels = levels();
        // 1 m per screen pixel over 10 m pixels: centred on pixel (500, 300).
        let camera = Camera {
            center: [505_000.0, 5_297_000.0],
            units_per_px: 10.0,
        };
        let visible = visible_tiles(&UTM, &levels[0], &camera, [100.0, 100.0]);
        // Pixels 450..550 × 250..350 touch tile columns 1–2 and rows 0–1.
        assert_eq!(visible.tiles.len(), 4);
        assert_eq!(
            visible.tiles[0],
            TileKey {
                level: 0,
                x: 1,
                y: 1
            }
        );
        assert_eq!(visible.level_rect, [450.0, 250.0, 550.0, 350.0]);
    }

    #[test]
    fn visible_tiles_should_be_empty_off_the_image() {
        let levels = levels();
        let camera = Camera {
            center: [0.0, 0.0],
            units_per_px: 10.0,
        };
        assert!(
            visible_tiles(&UTM, &levels[0], &camera, [100.0, 100.0])
                .tiles
                .is_empty()
        );
    }

    #[test]
    fn covering_tiles_should_map_a_fine_tile_to_its_coarse_parent() {
        let levels = levels();
        let fine = TileKey {
            level: 0,
            x: 3,
            y: 2,
        };
        assert_eq!(
            covering_tiles(&levels, fine, 1),
            vec![TileKey {
                level: 1,
                x: 1,
                y: 1
            }]
        );
        assert_eq!(
            covering_tiles(&levels, fine, 2),
            vec![TileKey {
                level: 2,
                x: 0,
                y: 0
            }]
        );
    }

    #[test]
    fn locate_pixel_should_find_the_tile_and_offset_at_an_overview() {
        let levels = levels();
        assert_eq!(
            locate_pixel(&levels[1], [700.0, 530.0]),
            Some((
                TileKey {
                    level: 1,
                    x: 1,
                    y: 1
                },
                [94, 9]
            ))
        );
        assert_eq!(locate_pixel(&levels[1], [1000.0, 0.0]), None);
    }

    #[test]
    fn edge_tiles_should_only_cover_the_image() {
        let levels = levels();
        let rect = tile_level_rect(
            &levels[0],
            TileKey {
                level: 0,
                x: 3,
                y: 2,
            },
        );
        assert_eq!(rect, [768, 512, 1000, 600]);
    }

    #[test]
    fn fit_should_show_the_whole_image() {
        let info = test_info();
        let camera = Camera::fit(&info, [100.0, 100.0]);
        assert_eq!(camera.center, [505_000.0, 5_297_000.0]);
        assert!((camera.units_per_px - 105.0).abs() < 1e-9);
    }

    #[test]
    fn stats_level_should_pick_the_finest_small_enough_level() {
        assert_eq!(stats_level(&levels(), 4), 1);
        assert_eq!(stats_level(&levels(), 1), 2);
        assert_eq!(stats_level(&levels(), 0), 2);
    }

    fn test_info() -> CogInfo {
        CogInfo {
            width: 1000,
            height: 600,
            bands: 1,
            data_type: crate::cog::DataType::U8,
            compression: crate::cog::Compression::None,
            planar: crate::cog::Planar::Chunky,
            photometric: 1,
            alpha_band: None,
            nodata: None,
            has_mask: false,
            geo: Some(UTM),
            crs: Default::default(),
            band_info: vec![Default::default()],
            file_size: None,
        }
    }
}
