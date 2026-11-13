# Cogcat: plan and goals

Cogcat is a viewer for Cloud Optimized GeoTIFFs (COGs), in the spirit of
[Copcat](https://github.com/UTAS-TerraLuma/copcat) for COPC point clouds: a native Rust
app (egui + wgpu) that streams only the tiles it needs, from a local file, an HTTP(S)
URL or cloud storage, and draws them in the image's own projection.

## Goals

1. **Open any COG**, local or remote, and stream it tile by tile.
   - Sources: local paths, `http(s)://` (range requests), `s3://`, `gs://`, `az://`
     (via `object_store`; public buckets work without credentials).
   - TIFF and BigTIFF, tiled, chunky or band-separate (planar) layouts, overviews and
     internal masks.
   - Sample types u8/i8/u16/i16/u32/i32/f32/f64. Compression: none, Deflate, LZW,
     PackBits, ZSTD, JPEG (incl. YCbCr), WebP. Horizontal and floating-point predictors.
   - Nodata and mask pixels become NaN when decoded, so drawing and statistics treat
     them the same way everywhere.
2. **Native projection.** Pixels are placed with the file's affine geotransform
   (`ModelPixelScale` + `ModelTiepoint`, or `ModelTransformation`) in CRS units. No
   reprojection. The status bar shows CRS coordinates, the EPSG code, the pixel row
   and column, and the values under the cursor.
3. **Level of detail.** Pick the overview whose pixels best match screen pixels,
   fetch the visible tiles coarse-first and centre-out, and draw coarser tiles under
   finer ones until those arrive. Memory budgets for the CPU and GPU tile caches.
4. **Tile debug view.** Outline each tile, coloured by state (drawn, fetching, queued,
   failed), with its `level/x/y` label, an optional tint by overview level, and
   streaming statistics (requests, bytes, tiles per level, cache use).
5. **Band composites.**
   - RGB COGs (3 or 4 samples of 8-bit RGB/YCbCr) open as true colour.
   - Multispectral COGs: pick a band for each of R, G and B, or a single band with a
     colour map. Presets (true colour, colour infrared, SWIR, agriculture, ...) are
     chosen from band wavelengths or names (Sentinel-2, Landsat, MicaSense, ...) where
     the file says what they are.
6. **Contrast stretching on the GPU.**
   - Modes: straight linear (min/max), percentile clip, optimized linear (ENVI style:
     2.5% / 99% percentiles widened by 10% / 50%), standard deviations, and manual.
   - Statistics come from a compute shader: a min/max pass, then a histogram pass, over
     either the whole image (from the coarsest overview) or the tiles in view. Only the
     small histogram is read back, and percentiles are taken from it.
   - The stretch itself, gamma, and colour mapping run in the fragment shader.
     Histograms are drawn in the stretch panel with the cut-offs marked.
7. **Spectral viewer** for multispectral COGs (more than 3 bands, or not RGB): a plot of
   every band's value under the cursor, updated as it moves, against wavelength when
   the file gives it. Click to pin spectra for comparison. Hidden for plain RGB COGs.
8. **Command line** like Copcat's: `cogcat open <source> [--bands 4,3,2] [--stretch ...]`
   starts a detached viewer window, `cogcat info <source>` prints the COG's structure,
   and `cogcat update` updates an installed release.
9. **CI/CD** like Copcat's: `cargo-dist` builds Linux x64 and Windows x64 binaries plus
   shell and PowerShell installers, only when a version tag is pushed. A check workflow
   (fmt, clippy, tests) runs on tags and by hand.

## Architecture

```
src/
  bin/cogcat.rs         command line: open / info / update
  bin/cogcat-viewer.rs  the viewer window
  lib.rs                run_viewer(), shared options
  source.rs             ByteSource: local file, HTTP(S), object_store (s3/gs/az)
  cog/                  TIFF/BigTIFF IFD parsing, geo keys, GDAL metadata, decoding
  loader.rs             worker threads, tile request queue, CPU tile cache
  view.rs               2D camera in CRS units, overview choice, visible tiles
  composite.rs          band roles, wavelength/name detection, presets
  stretch.rs            histogram type and stretch modes (pure, unit tested)
  render/               wgpu: (tile, band) textures, tile drawing, GPU statistics
  spectral.rs           spectral plot panel
  app.rs                egui UI
```

Data flow: the UI asks `view` for the tiles it wants, the `loader` fetches and decodes
them on worker threads into band-sequential `f32` tiles kept in a CPU cache, and the app
uploads the bands the current composite needs to the GPU as `R32Float` textures. The
renderer draws each tile's quad, sampling its three band textures, stretching and
colour mapping in the shader, into an offscreen texture that egui shows (as Copcat does).

Keeping only the displayed bands on the GPU matters for hyperspectral COGs: a
270-band tile of 256 × 256 pixels is 70 MB of `f32`, so every band of every visible
tile won't fit. The CPU cache keeps all bands, which the spectral viewer reads.

## Milestones

1. COG reader and sources, with `cogcat info`, tested on generated COGs and public
   Sentinel-2 COGs.
2. GPU renderer and statistics, tested headless (lavapipe in CI-less dev containers).
3. Viewer: streaming, LOD, pan/zoom, composites, stretch UI, debug view.
4. Spectral viewer.
5. CLI, README, release workflow.

## Later

- Reprojection to a common CRS / web mercator basemap.
- Multiple COGs as layers; STAC item browsing.
- `cogcat render` to write a PNG without a window, as `copcat render` does.
- LERC and JPEG XL compression.
