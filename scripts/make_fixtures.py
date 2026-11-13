#!/usr/bin/env python3
"""Generates the small GeoTIFF fixtures in tests/data/ that the COG reader's tests
decode, and what GDAL (through rasterio) reads from them, to compare against.

    python3 scripts/make_fixtures.py

For each fixture `name.tif` it writes `name.L0.f32.zz` (full resolution) and, when the
file has overviews, `name.L1.f32.zz` (the first overview): GDAL's pixel values as
band-sequential little-endian f32, masked and nodata pixels as NaN, zlib-compressed.
`fixtures.json` lists every fixture with its expected structure and georeferencing.

Images are small (200 x 150 with 64-pixel tiles, so the right and bottom tiles are
partial) and their content is smooth or regular so that they, and the expected
values, compress well. Needs rasterio (GDAL >= 3.7 for int8) and numpy.
"""

import json
import os
import shutil
import tempfile
import zlib

import numpy as np
import rasterio
import rasterio.shutil
import rasterio.windows
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.transform import Affine

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests", "data")
WIDTH, HEIGHT = 200, 150
TILE = 64
UTM55S = CRS.from_epsg(32755)
UTM_TRANSFORM = Affine(10.0, 0.0, 523980.0, 0.0, -10.0, 5272020.0)

xx, yy = np.meshgrid(np.arange(WIDTH, dtype=np.float64), np.arange(HEIGHT, dtype=np.float64))


def smooth_rgb():
    """Smooth colours, so lossy codecs stay close to the original."""
    r = 128 + 100 * np.sin(xx / 23.0)
    g = 128 + 100 * np.cos(yy / 17.0)
    b = (xx + yy) * 255.0 / (WIDTH + HEIGHT)
    return np.stack([r, g, b]).round().astype(np.uint8)


def circle_mask():
    """255 = valid, 0 = masked: a disc and the top-left corner are masked out."""
    mask = np.full((HEIGHT, WIDTH), 255, np.uint8)
    mask[(xx - 130) ** 2 + (yy - 80) ** 2 < 30**2] = 0
    mask[:10, :20] = 0
    return mask


def profile(dtype, count, **extra):
    p = dict(
        driver="GTiff",
        width=WIDTH,
        height=HEIGHT,
        count=count,
        dtype=dtype,
        crs=UTM55S,
        transform=UTM_TRANSFORM,
        tiled=True,
        blockxsize=TILE,
        blockysize=TILE,
    )
    p.update(extra)
    return p


def write_source(tmp, name, prof, data, mask=None, setup=None):
    """Writes an uncompressed GTiff in `tmp` to convert to a COG from."""
    path = os.path.join(tmp, name)
    with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True):
        with rasterio.open(path, "w", **prof) as dst:
            dst.write(data)
            if mask is not None:
                dst.write_mask(mask)
            if setup:
                setup(dst)
    return path


def to_cog(src, name, **options):
    out = os.path.join(ROOT, name)
    options.setdefault("BLOCKSIZE", TILE)
    options.setdefault("OVERVIEW_RESAMPLING", "AVERAGE")
    rasterio.shutil.copy(src, out, driver="COG", **options)
    return out


def write_gtiff(name, prof, data, overviews=None, setup=None):
    """Writes a plain (non-COG) GTiff straight into tests/data."""
    out = os.path.join(ROOT, name)
    with rasterio.open(out, "w", **prof) as dst:
        dst.write(data)
        if setup:
            setup(dst)
    if overviews:
        with rasterio.open(out, "r+") as dst:
            dst.build_overviews(overviews, Resampling.average)
    return out


def expected(path, overview_level=None):
    kwargs = {} if overview_level is None else {"overview_level": overview_level}
    with rasterio.open(path, **kwargs) as ds:
        data = ds.read(masked=True).astype(np.float32).filled(np.nan)
    return data


def save_expected(path, suffix, data):
    out = path[: -len(".tif")] + suffix
    with open(out, "wb") as f:
        f.write(zlib.compress(np.ascontiguousarray(data, "<f4").tobytes(), 9))
    return os.path.basename(out)


def describe(path, tolerance=0.0, wavelengths=None, extra=None):
    with rasterio.open(path) as ds:
        levels = [[ds.width, ds.height]]
        for factor in ds.overviews(1):
            with rasterio.open(path, overview_level=len(levels) - 1) as ov:
                levels.append([ov.width, ov.height])
        entry = {
            "file": os.path.basename(path),
            "width": ds.width,
            "height": ds.height,
            "bands": ds.count,
            "dtype": ds.dtypes[0],
            "nodata": None if ds.nodata is None else (str(ds.nodata)),
            "block": list(ds.block_shapes[0])[::-1],
            "levels": levels,
            "transform": list(ds.transform.to_gdal()) if not ds.transform.is_identity else None,
            "epsg": ds.crs.to_epsg() if ds.crs else None,
            "descriptions": list(ds.descriptions),
            "scales": list(ds.scales),
            "offsets": list(ds.offsets),
            "tolerance": tolerance,
        }
        has_overviews = len(levels) > 1
    entry["expected_l0"] = save_expected(path, ".L0.f32.zz", expected(path))
    entry["expected_l1"] = (
        save_expected(path, ".L1.f32.zz", expected(path, 0)) if has_overviews else None
    )
    entry["wavelengths_nm"] = wavelengths
    if extra:
        entry.update(extra)
    return entry


def main():
    if os.path.isdir(ROOT):
        shutil.rmtree(ROOT)
    os.makedirs(ROOT)
    fixtures = []
    tmp = tempfile.mkdtemp()
    try:
        # u8 RGB, JPEG in YCbCr, with an internal mask.
        src = write_source(tmp, "rgb.tif", profile("uint8", 3, photometric="RGB"), smooth_rgb(), circle_mask())
        cog = to_cog(src, "rgb_jpeg_mask.tif", COMPRESS="JPEG", QUALITY=90)
        fixtures.append(describe(cog, tolerance=3.0, extra={"compression": "Jpeg", "has_mask": True, "photometric": 6}))

        # JPEG without YCbCr: RGB photometric, and a single grey band.
        prof = profile("uint8", 3, compress="jpeg", photometric="RGB", jpeg_quality=90)
        gt = write_gtiff("rgb_jpeg_rgb.tif", prof, smooth_rgb())
        fixtures.append(describe(gt, tolerance=3.0, extra={"compression": "Jpeg", "photometric": 2}))
        prof = profile("uint8", 1, compress="jpeg", jpeg_quality=90)
        gt = write_gtiff("grey_jpeg.tif", prof, smooth_rgb()[1:2])
        fixtures.append(describe(gt, tolerance=3.0, extra={"compression": "Jpeg", "photometric": 1}))

        # u8 RGB, lossy WebP.
        src = write_source(tmp, "rgb2.tif", profile("uint8", 3, photometric="RGB"), smooth_rgb())
        cog = to_cog(src, "rgb_webp.tif", COMPRESS="WEBP", QUALITY=90)
        fixtures.append(describe(cog, tolerance=3.0, extra={"compression": "Webp", "photometric": 2}))

        # u16, 5 bands pixel-interleaved, Deflate + horizontal predictor, nodata 0
        # (a block in every band, plus single pixels in band 2 only), band
        # descriptions, wavelengths in several spellings, a scale and offset.
        data = np.stack([1000 * (b + 1) + (xx * 7 + yy * 13) % 1000 for b in range(5)]).astype(np.uint16)
        data[:, 100:120, 30:70] = 0
        data[1, 5:7, 150:190] = 0

        def setup_multi(dst):
            for b, name in enumerate(["blue", "green", "red", "rededge", "nir"]):
                dst.set_band_description(b + 1, name)
            dst.update_tags(1, wavelength="0.49", wavelength_units="micrometers")
            dst.update_tags(2, wavelength="560", wavelength_units="nm")
            dst.update_tags(3, WAVELENGTH="665")
            dst.update_tags(4, ns="IMAGERY", CENTRAL_WAVELENGTH_UM="0.705")
            dst.update_tags(5, wavelength="0.842")
            dst.scales = (0.0001, 1, 1, 1, 1)
            dst.offsets = (-0.1, 0, 0, 0, 0)

        src = write_source(tmp, "multi.tif", profile("uint16", 5, nodata=0), data, setup=setup_multi)
        cog = to_cog(src, "u16_5band_deflate_p2.tif", COMPRESS="DEFLATE", PREDICTOR=2)
        fixtures.append(
            describe(cog, wavelengths=[490.0, 560.0, 665.0, 705.0, 842.0], extra={"compression": "Deflate"})
        )

        # u16, 3 bands planar separate, ZSTD + predictor, ENVI-style wavelength list
        # (not a COG: overviews after the image).
        data = np.stack([500 * (b + 1) + (xx * 3 + yy * 11 + b) % 700 for b in range(3)]).astype(np.uint16)

        def setup_envi(dst):
            dst.update_tags(ns="ENVI", wavelength="{450.5, 550, 650.25}", wavelength_units="Nanometers")

        # The COG driver drops the ENVI domain, so write a tiled GTiff directly.
        prof = profile("uint16", 3, interleave="band", compress="zstd", predictor=2)
        cog = write_gtiff("u16_planar_zstd.tif", prof, data, overviews=[2, 4], setup=setup_envi)
        fixtures.append(
            describe(cog, wavelengths=[450.5, 550.0, 650.25], extra={"compression": "Zstd", "planar": "Separate"})
        )

        # f32, LZW + floating-point predictor, NaNs and nodata -9999.
        data = (np.sin(xx / 10.0) * np.cos(yy / 13.0) * 1000 + 0.123).astype(np.float32)[None]
        data[0, 20:30, 20:60] = np.nan
        data[0, 60:64, 100:180] = -9999
        src = write_source(tmp, "f32.tif", profile("float32", 1, nodata=-9999), data)
        cog = to_cog(src, "f32_lzw_p3.tif", COMPRESS="LZW", PREDICTOR=3)
        fixtures.append(describe(cog, extra={"compression": "Lzw"}))

        # i16, PackBits, negative values and runs.
        data = ((xx // 10) * 50 - 3000 + (yy // 4) * 7).astype(np.int16)[None]
        src = write_source(tmp, "i16.tif", profile("int16", 1), data)
        cog = to_cog(src, "i16_packbits.tif", COMPRESS="PACKBITS")
        fixtures.append(describe(cog, extra={"compression": "PackBits"}))

        # f64, uncompressed: smaller, with 32-pixel tiles, to keep the file small.
        sx, sy = np.meshgrid(np.arange(90.0), np.arange(70.0))
        data = (sx * 0.001 + sy * 1000.5 - 12345.678)[None]
        prof = profile("float64", 1, width=90, height=70, blockxsize=32, blockysize=32)
        src = write_source(tmp, "f64.tif", prof, data)
        cog = to_cog(src, "f64_none.tif", COMPRESS="NONE", BLOCKSIZE=32)
        fixtures.append(describe(cog, extra={"compression": "None"}))

        # Big-endian TIFFs (not COG layout: overviews appended after the image).
        data = (xx * 300 + yy * 7).astype(np.uint16)[None]
        prof = profile("uint16", 1, compress="deflate", predictor=2, ENDIANNESS="BIG")
        gt = write_gtiff("u16_bigendian_p2.tif", prof, data, overviews=[2])
        fixtures.append(describe(gt, extra={"compression": "Deflate"}))

        data = (np.sin(xx / 7.0) * yy * 3.5).astype(np.float32)[None]
        prof = profile("float32", 1, compress="deflate", predictor=3, ENDIANNESS="BIG")
        gt = write_gtiff("f32_bigendian_p3.tif", prof, data)
        fixtures.append(describe(gt, extra={"compression": "Deflate"}))

        # BigTIFF, u32 with values beyond f32's integer precision.
        data = (xx * 20_000_000 + yy * 3 + 7).astype(np.uint32)[None]
        src = write_source(tmp, "u32.tif", profile("uint32", 1), data)
        cog = to_cog(src, "u32_bigtiff.tif", COMPRESS="DEFLATE", BIGTIFF="YES")
        fixtures.append(describe(cog, extra={"compression": "Deflate", "bigtiff": True}))

        # A stripped (not tiled) i8 TIFF, 2 bands, LZW + predictor, 7 rows per strip.
        data = np.stack([(xx + yy) % 200 - 100, (xx - yy) % 120 - 60]).astype(np.int8)
        prof = profile("int8", 2, compress="lzw", predictor=2)
        prof.update(tiled=False, blockysize=7)
        del prof["blockxsize"]
        gt = write_gtiff("i8_stripped.tif", prof, data)
        fixtures.append(describe(gt, extra={"compression": "Lzw"}))

        # A rotated image: ModelTransformation instead of scale + tiepoint.
        data = ((xx * 1000 - yy * 1000) - 50_000).astype(np.int32)[None]
        rotated = Affine(8.660254, -5.0, 500000.0, -5.0, -8.660254, 5270000.0)
        src = write_source(tmp, "rot.tif", profile("int32", 1, transform=rotated), data)
        cog = to_cog(src, "i32_rotated.tif", COMPRESS="DEFLATE")
        fixtures.append(describe(cog, extra={"compression": "Deflate"}))

        # PixelIsPoint, geographic CRS.
        data = ((xx + 2 * yy) % 256).astype(np.uint8)[None]
        geo = Affine(0.001, 0.0, 147.0, 0.0, -0.001, -42.0)

        def setup_point(dst):
            dst.update_tags(AREA_OR_POINT="Point")

        src = write_source(
            tmp, "point.tif", profile("uint8", 1, crs=CRS.from_epsg(4326), transform=geo), data, setup=setup_point
        )
        cog = to_cog(src, "u8_pixel_is_point.tif", COMPRESS="DEFLATE")
        fixtures.append(describe(cog, extra={"compression": "Deflate"}))

        # Sparse tiles (no bytes at all) where nothing was written; nodata 255.
        out = os.path.join(ROOT, "u8_sparse.tif")
        with rasterio.open(out, "w", **profile("uint8", 1, nodata=255, compress="deflate", sparse_ok=True)) as dst:
            window = rasterio.windows.Window(0, 0, 64, 150)
            dst.write(((xx[:, :64] + yy[:, :64]) % 250).astype(np.uint8)[None], window=window)
        fixtures.append(describe(out, extra={"compression": "Deflate"}))
    finally:
        shutil.rmtree(tmp)

    with open(os.path.join(ROOT, "fixtures.json"), "w") as f:
        json.dump(fixtures, f, indent=1)
        f.write("\n")
    total = sum(os.path.getsize(os.path.join(ROOT, n)) for n in os.listdir(ROOT))
    print(f"wrote {len(fixtures)} fixtures, {total / 1024:.0f} KiB in {ROOT}")


if __name__ == "__main__":
    main()
