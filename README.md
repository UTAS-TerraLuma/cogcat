# Cogcat

A viewer for [Cloud Optimized GeoTIFFs](https://cogeo.org/) (COGs), local or streamed from
HTTP(S), Amazon S3, Google Cloud Storage or Azure. It fetches only the tiles in view, at
the overview that suits the zoom, and draws them in the image's own projection.

Cogcat is the raster sibling of [Copcat](https://github.com/UTAS-TerraLuma/copcat), the
COPC point cloud viewer.

- **Any COG**: TIFF or BigTIFF; 8 to 64-bit integer or float samples; Deflate, LZW,
  ZSTD, JPEG, WebP, PackBits or no compression; pixel or band interleaved; overviews,
  nodata and internal masks.
- **Native projection**: pixels sit where the file's geotransform puts them, in its CRS.
  The status bar shows coordinates, the pixel's row and column, and its values.
- **Band composites**: pick any band for red, green and blue, or show one band through a
  colour map. Presets such as true colour, colour infrared and short-wave infrared are
  picked from the bands' wavelengths or names (Sentinel-2, Landsat, MicaSense, ...).
- **Contrast stretching on the GPU**: straight linear (min–max), percentile clip,
  optimized linear and standard deviation stretches, from histograms computed by a
  compute shader over the whole image or just the view. Gamma, linked or per-channel
  ranges, and manual ranges.
- **Spectral viewer**: for multispectral images, a plot of every band under the cursor,
  against wavelength when the file gives it. Click to pin spectra and compare them;
  copy them as CSV.
- **Tile debug view**: outlines every tile with its level and position, coloured by
  whether it's drawn, decoded, fetching, queued or failed, with streaming statistics.

## Install

Prebuilt binaries are available for Windows x64 and Linux x64. You don't need Rust.

Windows (PowerShell):

```powershell
powershell -ExecutionPolicy Bypass -c "irm https://github.com/UTAS-TerraLuma/cogcat/releases/latest/download/cogcat-installer.ps1 | iex"
```

Linux:

```sh
curl --proto '=https' --tlsv1.2 -LsSf https://github.com/UTAS-TerraLuma/cogcat/releases/latest/download/cogcat-installer.sh | sh
```

This installs `cogcat`, `cogcat-viewer` and `cogcat-update` into `~/.local/bin`
(`%USERPROFILE%\.local\bin` on Windows), or `$XDG_BIN_HOME` if it's set, and adds that
folder to `PATH`. Open a new terminal afterwards.

The Windows binaries aren't code signed, so the first time `cogcat-viewer.exe` runs,
SmartScreen may show "Windows protected your PC". Click "More info", then "Run anyway".

## Update

```sh
cogcat update
```

## Usage

Open a file or URL in a new viewer window:

```sh
cogcat open path/to/image.tif
cogcat open https://example.com/image.tif
cogcat open s3://sentinel-cogs/sentinel-s2-l2a-cogs/55/G/EN/2024/1/S2A_55GEN_20240109_0_L2A/TCI.tif
```

Choose how it first looks:

```sh
# Bands for red, green and blue, counting from 1
cogcat open multispectral.tif --bands 8,4,3 --stretch percentile --percentiles 2,98
# One band through a colour map
cogcat open dem.tif --band 1 --colormap terrain --stretch minmax
# Outline the tiles as they load
cogcat open image.tif --debug-tiles
```

`--stretch` takes `minmax`, `percentile`, `optimized`, `stddev` or `none`. Run
`cogcat open --help` for everything.

Print a COG's structure (size, bands, compression, CRS, overviews and tiles):

```sh
cogcat info s3://bucket/image.tif
```

You can also start `cogcat-viewer` on its own and paste a path or URL into it, or click
**Browse…**.

### In the viewer

| | |
|---|---|
| Drag | Pan |
| Scroll, pinch | Zoom about the cursor |
| Double-click | Zoom in |
| <kbd>F</kbd> or <kbd>Home</kbd> | Fit the whole image |
| <kbd>1</kbd> | One image pixel per screen pixel |
| <kbd>+</kbd> / <kbd>-</kbd> | Zoom |
| Click | Pin the spectrum under the cursor (multispectral images) |

### Contrast stretches

| Stretch | Range |
|---|---|
| Linear (min–max) | The smallest to the largest value. |
| Percentile clip | Between two percentiles, 2% and 98% by default. |
| Optimized linear | ENVI's: percentiles *a* = 2.5% and *b* = 99%, widened to *a* − 0.1(*b* − *a*) and *b* + 0.5(*b* − *a*). Usually darker and less harsh than a percentile clip. |
| Standard deviation | The mean ± 2 standard deviations. |
| Manual | Type or drag the ranges. |

Statistics come from **Whole image** (the finest overview with at most 64 tiles) or the
**View**. With **Follow view**, the stretch updates whenever the view stops moving.
**Same range for R, G and B** keeps colours balanced, for true colour. 8-bit RGB images
open unstretched; everything else opens with a 2–98% percentile clip from the whole image.

## Cloud storage

`s3://`, `gs://` and `az://` (or `abfs://`) URLs are read with the
[`object_store`](https://docs.rs/object_store) crate, which picks up credentials from the
usual environment variables:

- **S3**: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_SESSION_TOKEN`, `AWS_REGION`,
  `AWS_ENDPOINT` (for S3-compatible stores). Without credentials, requests are unsigned,
  which works for public buckets; the bucket's region is looked up automatically.
- **Google Cloud Storage**: `GOOGLE_APPLICATION_CREDENTIALS` or `GOOGLE_SERVICE_ACCOUNT`.
  Without them, requests are unsigned, for public buckets.
- **Azure**: `AZURE_STORAGE_ACCOUNT_NAME` with `AZURE_STORAGE_ACCOUNT_KEY` or a SAS token.

Any `https://` URL of a server that supports range requests works too, including
pre-signed URLs.

## Building from source

```sh
cargo run --release -- path/to/image.tif
```

`cargo test` runs the unit tests and the fixture tests in `tests/`. GPU tests skip
themselves when there's no GPU; on Linux, Mesa's lavapipe (`mesa-vulkan-drivers`) gives
them a software one. `scripts/make_fixtures.py` regenerates the test fixtures (it needs
`rasterio`).

See [docs/plan.md](docs/plan.md) for the design and [docs/releasing.md](docs/releasing.md)
for how releases are made.
