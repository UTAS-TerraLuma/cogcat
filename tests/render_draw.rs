//! Draws synthetic tiles headless and checks the pixels. Skipped (with a
//! note) when no GPU adapter is available, as on CI runners; in a container,
//! Mesa's lavapipe (software Vulkan) is enough.

use std::collections::HashSet;

use cogcat::cog::TileKey;
use cogcat::render::{
    Colormap, DisplayStyle, Image, TileDraw, TileRenderer, Viewport, headless_device,
};

fn renderer() -> Option<TileRenderer> {
    match headless_device() {
        Ok((device, queue)) => Some(TileRenderer::headless(device, queue)),
        Err(error) => {
            eprintln!("skipping GPU test: {error}");
            None
        }
    }
}

fn key(x: u32) -> TileKey {
    TileKey { level: 0, x, y: 0 }
}

/// The whole of clip space: top-left, top-right, bottom-right, bottom-left.
const FULL: [[f32; 2]; 4] = [[-1.0, 1.0], [1.0, 1.0], [1.0, -1.0], [-1.0, -1.0]];

fn tile(key: TileKey, corners: [[f32; 2]; 4], texels: [f32; 2]) -> TileDraw {
    TileDraw {
        key,
        corners,
        texels,
        tint: [0.0; 4],
    }
}

/// Bands 0, 1, 2 stretched 0..1 with gamma 1 on an opaque black background.
fn style() -> DisplayStyle {
    DisplayStyle {
        bands: [0, 1, 2],
        colormap: None,
        low: [0.0; 3],
        high: [1.0; 3],
        gamma: 1.0,
        background: [0.0, 0.0, 0.0, 1.0],
    }
}

/// Uploads `values` as all three of bands 0, 1 and 2.
fn upload_gray(renderer: &mut TileRenderer, key: TileKey, size: [u32; 2], values: &[f32]) {
    for band in 0..3 {
        renderer.upload_band(key, band, size, values);
    }
}

fn pixel(image: &Image, x: u32, y: u32) -> [u8; 4] {
    let at = ((y * image.width + x) * 4) as usize;
    image.rgba[at..at + 4].try_into().unwrap()
}

fn assert_close(actual: [u8; 4], expected: [u8; 4], tolerance: u8, what: &str) {
    let close = (0..4).all(|c| actual[c].abs_diff(expected[c]) <= tolerance);
    assert!(close, "{what}: got {actual:?}, expected {expected:?}");
}

fn capture(
    renderer: &mut TileRenderer,
    size: [u32; 2],
    tiles: &[TileDraw],
    style: &DisplayStyle,
) -> Image {
    let mut viewport = Viewport::default();
    renderer.capture(&mut viewport, size, tiles, style).unwrap()
}

#[test]
fn capture_should_stretch_each_band_linearly() {
    let Some(mut renderer) = renderer() else {
        return;
    };
    // A 4 × 4 tile drawn 1:1, values 0, 10, ..., 150 in band 0.
    let ramp: Vec<f32> = (0..16).map(|i| i as f32 * 10.0).collect();
    let reversed: Vec<f32> = ramp.iter().rev().copied().collect();
    renderer.upload_band(key(0), 0, [4, 4], &ramp);
    renderer.upload_band(key(0), 1, [4, 4], &reversed);
    renderer.upload_band(key(0), 2, [4, 4], &[7.0; 16]);
    let style = DisplayStyle {
        low: [0.0, 50.0, 7.0],
        high: [150.0, 100.0, 7.0],
        ..style()
    };
    let image = capture(
        &mut renderer,
        [4, 4],
        &[tile(key(0), FULL, [4.0, 4.0])],
        &style,
    );
    for i in 0..16u32 {
        let (x, y) = (i % 4, i / 4);
        let r = ramp[i as usize] / 150.0;
        let g = ((reversed[i as usize] - 50.0) / 50.0).clamp(0.0, 1.0);
        // A constant band (low == high) at its value shows as mid-grey.
        let expected = [r, g, 0.5, 1.0].map(|c| (c * 255.0).round() as u8);
        assert_close(pixel(&image, x, y), expected, 1, &format!("pixel {x}, {y}"));
    }
}

#[test]
fn capture_should_write_half_as_byte_128() {
    let Some(mut renderer) = renderer() else {
        return;
    };
    upload_gray(&mut renderer, key(0), [1, 1], &[0.5]);
    let image = capture(
        &mut renderer,
        [2, 2],
        &[tile(key(0), FULL, [1.0, 1.0])],
        &style(),
    );
    // No sRGB conversion: 0.5 is written as is.
    assert_close(pixel(&image, 1, 1), [128, 128, 128, 255], 1, "mid-grey");
}

#[test]
fn capture_should_apply_gamma_after_stretching() {
    let Some(mut renderer) = renderer() else {
        return;
    };
    upload_gray(&mut renderer, key(0), [2, 1], &[25.0, 0.0]);
    let style = DisplayStyle {
        low: [0.0; 3],
        high: [100.0; 3],
        gamma: 2.0,
        ..style()
    };
    let image = capture(
        &mut renderer,
        [2, 1],
        &[tile(key(0), FULL, [2.0, 1.0])],
        &style,
    );
    // 0.25 ^ (1 / 2) = 0.5, and 0 stays 0.
    assert_close(
        pixel(&image, 0, 0),
        [128, 128, 128, 255],
        1,
        "0.25 with gamma 2",
    );
    assert_close(pixel(&image, 1, 0), [0, 0, 0, 255], 0, "0 with gamma 2");
}

#[test]
fn capture_should_show_the_background_where_any_band_is_nan() {
    let Some(mut renderer) = renderer() else {
        return;
    };
    renderer.upload_band(key(0), 0, [2, 1], &[f32::NAN, 1.0]);
    renderer.upload_band(key(0), 1, [2, 1], &[1.0, 1.0]);
    // A NaN with a payload and a negative NaN are no data too.
    renderer.upload_band(key(0), 2, [2, 1], &[1.0, -f32::from_bits(0x7fc0_1234)]);
    let style = DisplayStyle {
        background: [0.2, 0.4, 0.6, 1.0],
        ..style()
    };
    let image = capture(
        &mut renderer,
        [2, 1],
        &[tile(key(0), FULL, [2.0, 1.0])],
        &style,
    );
    let background = [51, 102, 153, 255];
    assert_close(pixel(&image, 0, 0), background, 1, "NaN in red");
    assert_close(pixel(&image, 1, 0), background, 1, "NaN in blue");
}

#[test]
fn capture_should_show_tiles_underneath_nan_pixels() {
    let Some(mut renderer) = renderer() else {
        return;
    };
    upload_gray(&mut renderer, key(0), [1, 1], &[1.0]);
    upload_gray(&mut renderer, key(1), [2, 1], &[f32::NAN, 0.0]);
    let tiles = [
        tile(key(0), FULL, [1.0, 1.0]),
        tile(key(1), FULL, [2.0, 1.0]),
    ];
    let image = capture(&mut renderer, [2, 1], &tiles, &style());
    assert_close(
        pixel(&image, 0, 0),
        [255, 255, 255, 255],
        0,
        "under the NaN",
    );
    assert_close(pixel(&image, 1, 0), [0, 0, 0, 255], 0, "the finer tile");
}

#[test]
fn capture_should_draw_later_tiles_on_top() {
    let Some(mut renderer) = renderer() else {
        return;
    };
    upload_gray(&mut renderer, key(0), [1, 1], &[0.2]);
    upload_gray(&mut renderer, key(1), [1, 1], &[0.8]);
    // Key 2 is never uploaded, so it's skipped rather than drawn over the rest.
    let left_half = [[-1.0, 1.0], [0.0, 1.0], [0.0, -1.0], [-1.0, -1.0]];
    let tiles = [
        tile(key(0), FULL, [1.0, 1.0]),
        tile(key(1), left_half, [1.0, 1.0]),
        tile(key(2), FULL, [1.0, 1.0]),
    ];
    let image = capture(&mut renderer, [4, 2], &tiles, &style());
    assert_close(pixel(&image, 0, 0), [204, 204, 204, 255], 1, "top tile");
    assert_close(pixel(&image, 3, 1), [51, 51, 51, 255], 1, "bottom tile");
}

#[test]
fn capture_should_clear_to_the_background_where_nothing_is_drawn() {
    let Some(mut renderer) = renderer() else {
        return;
    };
    let style = DisplayStyle {
        background: [1.0, 0.0, 0.0, 0.0],
        ..style()
    };
    let image = capture(&mut renderer, [3, 3], &[], &style);
    assert!(image.rgba.chunks(4).all(|p| p == [255, 0, 0, 0]));
}

#[test]
fn capture_should_draw_a_rotated_quad() {
    let Some(mut renderer) = renderer() else {
        return;
    };
    let values: Vec<f32> = (0..16).map(|i| i as f32 / 15.0).collect();
    upload_gray(&mut renderer, key(0), [4, 4], &values);
    // Turned a quarter clockwise: the image's top-left corner is at the
    // screen's top-right, and its rows run down the screen.
    let rotated = [[1.0, 1.0], [1.0, -1.0], [-1.0, -1.0], [-1.0, 1.0]];
    let image = capture(
        &mut renderer,
        [4, 4],
        &[tile(key(0), rotated, [4.0, 4.0])],
        &style(),
    );
    for py in 0..4 {
        for px in 0..4 {
            let (x, y) = (py, 3 - px);
            let v = (values[(y * 4 + x) as usize] * 255.0).round() as u8;
            assert_close(
                pixel(&image, px, py),
                [v, v, v, 255],
                1,
                &format!("{px}, {py}"),
            );
        }
    }
}

#[test]
fn capture_should_map_a_partial_edge_tile_to_its_valid_texels() {
    let Some(mut renderer) = renderer() else {
        return;
    };
    // An 8 × 8 texture whose valid area is the top-left 2 × 4 texels.
    let values: Vec<f32> = (0..64).map(|i| i as f32 / 63.0).collect();
    upload_gray(&mut renderer, key(0), [8, 8], &values);
    let left_half = [[-1.0, 1.0], [0.0, 1.0], [0.0, -1.0], [-1.0, -1.0]];
    let style = DisplayStyle {
        background: [0.0, 0.0, 1.0, 1.0],
        ..style()
    };
    let image = capture(
        &mut renderer,
        [4, 4],
        &[tile(key(0), left_half, [2.0, 4.0])],
        &style,
    );
    for y in 0..4 {
        for x in 0..2 {
            let v = (values[(y * 8 + x) as usize] * 255.0).round() as u8;
            assert_close(pixel(&image, x, y), [v, v, v, 255], 1, &format!("{x}, {y}"));
        }
        assert_close(
            pixel(&image, 3, y),
            [0, 0, 255, 255],
            0,
            "right of the tile",
        );
    }
}

#[test]
fn capture_should_scale_texels_to_pixels_by_nearest_neighbour() {
    let Some(mut renderer) = renderer() else {
        return;
    };
    upload_gray(&mut renderer, key(0), [2, 1], &[0.0, 1.0]);
    let image = capture(
        &mut renderer,
        [8, 2],
        &[tile(key(0), FULL, [2.0, 1.0])],
        &style(),
    );
    for x in 0..8 {
        let v = if x < 4 { 0 } else { 255 };
        assert_close(
            pixel(&image, x, 1),
            [v, v, v, 255],
            0,
            &format!("column {x}"),
        );
    }
}

#[test]
fn capture_should_draw_one_band_through_a_colormap() {
    let Some(mut renderer) = renderer() else {
        return;
    };
    // Only band 5 is uploaded: the others in `bands` aren't needed.
    renderer.upload_band(key(0), 5, [3, 1], &[-10.0, 10.0, 30.0]);
    let style = DisplayStyle {
        bands: [5, 6, 7],
        colormap: Some(Colormap::Viridis),
        low: [10.0, 0.0, 0.0],
        high: [30.0, 0.0, 0.0],
        ..style()
    };
    let image = capture(
        &mut renderer,
        [3, 1],
        &[tile(key(0), FULL, [3.0, 1.0])],
        &style,
    );
    let lut = Colormap::Viridis.lut();
    assert_close(pixel(&image, 0, 0), lut[0], 1, "below low");
    assert_close(pixel(&image, 1, 0), lut[0], 1, "at low");
    assert_close(pixel(&image, 2, 0), lut[255], 1, "at high");

    // Switching maps rewrites the table.
    let magma = DisplayStyle {
        colormap: Some(Colormap::Magma),
        ..style
    };
    let image = capture(
        &mut renderer,
        [3, 1],
        &[tile(key(0), FULL, [3.0, 1.0])],
        &magma,
    );
    assert_close(
        pixel(&image, 2, 0),
        Colormap::Magma.lut()[255],
        1,
        "magma at high",
    );
}

#[test]
fn capture_should_interpolate_the_colormap_between_entries() {
    let Some(mut renderer) = renderer() else {
        return;
    };
    renderer.upload_band(key(0), 0, [1, 1], &[0.5]);
    let style = DisplayStyle {
        colormap: Some(Colormap::Turbo),
        ..style()
    };
    let image = capture(
        &mut renderer,
        [1, 1],
        &[tile(key(0), FULL, [1.0, 1.0])],
        &style,
    );
    // 0.5 lies half way between entries 127 and 128.
    let lut = Colormap::Turbo.lut();
    let expected: [u8; 4] =
        std::array::from_fn(|c| ((u16::from(lut[127][c]) + u16::from(lut[128][c])) / 2) as u8);
    assert_close(pixel(&image, 0, 0), expected, 2, "turbo at 0.5");
}

#[test]
fn capture_should_blend_the_tint_by_its_alpha() {
    let Some(mut renderer) = renderer() else {
        return;
    };
    upload_gray(&mut renderer, key(0), [1, 1], &[0.0]);
    let tinted = TileDraw {
        tint: [1.0, 0.0, 0.0, 0.5],
        ..tile(key(0), FULL, [1.0, 1.0])
    };
    let image = capture(&mut renderer, [1, 1], &[tinted], &style());
    assert_close(
        pixel(&image, 0, 0),
        [128, 0, 0, 255],
        1,
        "half red over black",
    );
}

#[test]
fn capture_should_reject_impossible_sizes() {
    let Some(mut renderer) = renderer() else {
        return;
    };
    let mut viewport = Viewport::default();
    let too_big = renderer.max_image_size() + 1;
    assert!(
        renderer
            .capture(&mut viewport, [0, 4], &[], &style())
            .is_err()
    );
    assert!(
        renderer
            .capture(&mut viewport, [4, too_big], &[], &style())
            .is_err()
    );
}

#[test]
fn viewport_should_follow_size_changes() {
    let Some(mut renderer) = renderer() else {
        return;
    };
    upload_gray(&mut renderer, key(0), [1, 1], &[1.0]);
    let mut viewport = Viewport::default();
    let tiles = [tile(key(0), FULL, [1.0, 1.0])];
    for size in [[3, 2], [5, 7], [3, 2]] {
        let image = renderer
            .capture(&mut viewport, size, &tiles, &style())
            .unwrap();
        assert_eq!([image.width, image.height], size);
        assert!(image.rgba.chunks(4).all(|p| p == [255; 4]));
    }
}

#[test]
fn render_should_grow_the_vertex_buffer_for_many_tiles() {
    let Some(mut renderer) = renderer() else {
        return;
    };
    // 100 tiles in a 10 × 10 grid, each its own value.
    let mut tiles = Vec::new();
    for i in 0..100u32 {
        upload_gray(&mut renderer, key(i), [1, 1], &[i as f32 / 99.0]);
        let (col, row) = ((i % 10) as f32, (i / 10) as f32);
        let (left, top) = (-1.0 + col * 0.2, 1.0 - row * 0.2);
        let corners = [
            [left, top],
            [left + 0.2, top],
            [left + 0.2, top - 0.2],
            [left, top - 0.2],
        ];
        tiles.push(tile(key(i), corners, [1.0, 1.0]));
    }
    let mut viewport = Viewport::default();
    // A small draw first, so the buffer has to grow.
    renderer.render(&mut viewport, [20, 20], &tiles[..1], &style());
    let image = renderer
        .capture(&mut viewport, [20, 20], &tiles, &style())
        .unwrap();
    for i in 0..100u32 {
        let (x, y) = ((i % 10) * 2 + 1, (i / 10) * 2 + 1);
        let v = (i as f32 / 99.0 * 255.0).round() as u8;
        assert_close(pixel(&image, x, y), [v, v, v, 255], 1, &format!("tile {i}"));
    }
}

#[test]
fn upload_should_replace_a_band_and_its_bytes() {
    let Some(mut renderer) = renderer() else {
        return;
    };
    upload_gray(&mut renderer, key(0), [4, 4], &[0.0; 16]);
    assert_eq!(renderer.resident_bytes(), 3 * 16 * 4);
    // Drawn once, so a bind group exists for the old textures.
    capture(
        &mut renderer,
        [1, 1],
        &[tile(key(0), FULL, [4.0, 4.0])],
        &style(),
    );
    renderer.upload_band(key(0), 0, [2, 2], &[1.0; 4]);
    assert_eq!(renderer.resident_bytes(), 2 * 16 * 4 + 4 * 4);
    assert!(renderer.has_band(key(0), 0) && !renderer.has_band(key(0), 3));
    // The new texture is drawn, not the replaced one.
    let image = capture(
        &mut renderer,
        [1, 1],
        &[tile(key(0), FULL, [1.0, 1.0])],
        &style(),
    );
    assert_close(pixel(&image, 0, 0), [255, 0, 0, 255], 0, "new red band");
}

#[test]
fn evict_should_free_least_recently_drawn_first_and_keep_what_is_asked() {
    let Some(mut renderer) = renderer() else {
        return;
    };
    for x in 0..4 {
        upload_gray(&mut renderer, key(x), [4, 4], &[0.5; 16]);
    }
    let per_tile = 3 * 16 * 4;
    // Draw tiles 2 then 1, so 0 and 3 are the least recently drawn.
    capture(
        &mut renderer,
        [1, 1],
        &[tile(key(2), FULL, [4.0, 4.0])],
        &style(),
    );
    capture(
        &mut renderer,
        [1, 1],
        &[tile(key(1), FULL, [4.0, 4.0])],
        &style(),
    );

    let keep = HashSet::from([key(0)]);
    // Room for two tiles: 3 is freed (0 is kept), then 2 (drawn before 1).
    let evicted = renderer.evict_to_budget(2 * per_tile, &keep);
    let tiles: HashSet<TileKey> = evicted.iter().map(|(key, _)| *key).collect();
    assert_eq!(tiles, HashSet::from([key(3), key(2)]));
    assert_eq!(evicted.len(), 6);
    assert_eq!(renderer.resident_bytes(), 2 * per_tile);
    assert!(renderer.has_band(key(0), 0) && renderer.has_band(key(1), 2));
    assert_eq!(renderer.resident_textures(), 6);

    // Nothing more fits under a budget of 0, but kept tiles stay.
    let evicted = renderer.evict_to_budget(0, &keep);
    assert_eq!(evicted.len(), 3);
    assert!(renderer.has_band(key(0), 1));

    renderer.clear();
    assert_eq!(renderer.resident_bytes(), 0);
    assert!(!renderer.has_band(key(0), 0));
}

#[test]
fn evicted_tiles_should_be_skipped_when_drawing() {
    let Some(mut renderer) = renderer() else {
        return;
    };
    upload_gray(&mut renderer, key(0), [1, 1], &[1.0]);
    let tiles = [tile(key(0), FULL, [1.0, 1.0])];
    capture(&mut renderer, [1, 1], &tiles, &style());
    renderer.evict_to_budget(0, &HashSet::new());
    let image = capture(&mut renderer, [1, 1], &tiles, &style());
    assert_close(pixel(&image, 0, 0), [0, 0, 0, 255], 0, "background");
}
