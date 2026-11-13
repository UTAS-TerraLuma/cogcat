//! Decodes the fixtures in `tests/data/` (made by `scripts/make_fixtures.py`)
//! and compares them with what GDAL reads from the same files.

use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use cogcat::cog::{Cog, CogError, Compression, DataType, Planar, TileKey};
use cogcat::source::{ByteSource, SourceError, open_source};
use serde_json::Value;

fn data_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data")
}

fn fixtures() -> Vec<Value> {
    let text = std::fs::read_to_string(data_dir().join("fixtures.json")).unwrap();
    serde_json::from_str::<Value>(&text)
        .unwrap()
        .as_array()
        .unwrap()
        .clone()
}

fn fixture(name: &str) -> Value {
    fixtures()
        .into_iter()
        .find(|f| f["file"] == name)
        .unwrap_or_else(|| panic!("no fixture {name}"))
}

fn open(name: &str) -> Cog {
    Cog::open(data_dir().join(name).to_str().unwrap())
        .unwrap_or_else(|e| panic!("couldn't open {name}: {e}"))
}

/// GDAL's pixels: band-sequential f32, NaN where masked.
fn expected_pixels(file: &str) -> Vec<f32> {
    let mut bytes = Vec::new();
    let compressed = std::fs::read(data_dir().join(file)).unwrap();
    flate2::read::ZlibDecoder::new(&compressed[..])
        .read_to_end(&mut bytes)
        .unwrap();
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Reads every tile of a level and stitches them into one band-sequential
/// image, checking each tile's shape on the way.
fn read_level(cog: &Cog, level: usize) -> Vec<f32> {
    let l = cog.levels()[level].clone();
    let bands = cog.info().bands as usize;
    let (w, h) = (l.width as usize, l.height as usize);
    let keys: Vec<TileKey> = (0..l.tiles_down)
        .flat_map(|y| {
            (0..l.tiles_across).map(move |x| TileKey {
                level: level as u8,
                x,
                y,
            })
        })
        .collect();
    let mut image = vec![0.0; w * h * bands];
    for (key, tile) in keys.iter().zip(cog.read_tiles(&keys)) {
        let tile = tile.unwrap_or_else(|e| panic!("tile {key}: {e}"));
        assert_eq!(tile.key, *key);
        assert_eq!((tile.width, tile.height), (l.tile_width, l.tile_height));
        assert_eq!(tile.bands as usize, bands);
        assert_eq!(tile.data.len(), (tile.width * tile.height) as usize * bands);
        let x0 = (key.x * l.tile_width) as usize;
        let y0 = (key.y * l.tile_height) as usize;
        assert_eq!(
            tile.valid,
            [
                l.tile_width.min(l.width - x0 as u32),
                l.tile_height.min(l.height - y0 as u32)
            ]
        );
        // Only sparse tiles, which are all NaN, come from no bytes.
        assert!(tile.fetched_bytes > 0 || tile.data.iter().all(|v| v.is_nan()));
        for b in 0..bands {
            let band = tile.band(b as u32);
            for ty in 0..tile.height as usize {
                for tx in 0..tile.width as usize {
                    let v = band[ty * tile.width as usize + tx];
                    if tx >= tile.valid[0] as usize || ty >= tile.valid[1] as usize {
                        assert!(v.is_nan(), "tile {key} padding at {tx},{ty} is {v}");
                    } else {
                        image[(b * h + y0 + ty) * w + x0 + tx] = v;
                    }
                }
            }
        }
    }
    image
}

fn compare(name: &str, level: usize, actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len(), "{name} level {level}: size");
    let mut worst = 0.0f32;
    for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        assert_eq!(
            a.is_nan(),
            e.is_nan(),
            "{name} level {level} sample {i}: got {a}, GDAL has {e}"
        );
        if !a.is_nan() {
            let diff = (a - e).abs();
            worst = worst.max(diff);
            assert!(
                diff <= tolerance,
                "{name} level {level} sample {i}: got {a}, GDAL has {e}"
            );
        }
    }
    if tolerance > 0.0 {
        println!("{name} level {level}: largest difference from GDAL {worst}");
    }
}

fn data_type_name(t: DataType) -> &'static str {
    match t {
        DataType::U8 => "uint8",
        DataType::I8 => "int8",
        DataType::U16 => "uint16",
        DataType::I16 => "int16",
        DataType::U32 => "uint32",
        DataType::I32 => "int32",
        DataType::F32 => "float32",
        DataType::F64 => "float64",
    }
}

fn check_structure(f: &Value, cog: &Cog) {
    let name = f["file"].as_str().unwrap();
    let info = cog.info();
    assert_eq!(info.width as u64, f["width"].as_u64().unwrap(), "{name}");
    assert_eq!(info.height as u64, f["height"].as_u64().unwrap(), "{name}");
    assert_eq!(info.bands as u64, f["bands"].as_u64().unwrap(), "{name}");
    assert_eq!(data_type_name(info.data_type), f["dtype"], "{name}");
    let levels: Vec<[u64; 2]> = cog
        .levels()
        .iter()
        .map(|l| [l.width as u64, l.height as u64])
        .collect();
    let expected_levels: Vec<[u64; 2]> = f["levels"]
        .as_array()
        .unwrap()
        .iter()
        .map(|l| [l[0].as_u64().unwrap(), l[1].as_u64().unwrap()])
        .collect();
    assert_eq!(levels, expected_levels, "{name} levels");
    for (i, level) in cog.levels().iter().enumerate() {
        assert_eq!(level.index as usize, i);
        assert_eq!(level.tiles_across, level.width.div_ceil(level.tile_width));
        assert_eq!(level.tiles_down, level.height.div_ceil(level.tile_height));
        assert!((level.scale[0] - info.width as f64 / level.width as f64).abs() < 1e-12);
    }
    if !name.contains("stripped") {
        let block = &f["block"];
        assert_eq!(
            cog.levels()[0].tile_width as u64,
            block[0].as_u64().unwrap()
        );
        assert_eq!(
            cog.levels()[0].tile_height as u64,
            block[1].as_u64().unwrap()
        );
    }
    match f["nodata"].as_str() {
        Some(nodata) => assert_eq!(info.nodata, Some(nodata.parse().unwrap()), "{name}"),
        None => assert_eq!(info.nodata, None, "{name}"),
    }
    match f["transform"].as_array() {
        Some(t) => {
            let gt = info
                .geo
                .unwrap_or_else(|| panic!("{name} has no geotransform"));
            for (a, e) in gt.0.iter().zip(t) {
                let e = e.as_f64().unwrap();
                assert!(
                    (a - e).abs() <= 1e-9 * e.abs().max(1.0),
                    "{name}: {:?} vs {t:?}",
                    gt.0
                );
            }
        }
        None => assert!(info.geo.is_none()),
    }
    assert_eq!(info.crs.epsg.map(u64::from), f["epsg"].as_u64(), "{name}");
    for (b, band) in info.band_info.iter().enumerate() {
        assert_eq!(
            band.description.as_deref(),
            f["descriptions"][b].as_str(),
            "{name}"
        );
        let scale = f["scales"][b].as_f64().unwrap();
        assert_eq!(band.scale.unwrap_or(1.0), scale, "{name} band {b} scale");
        let offset = f["offsets"][b].as_f64().unwrap();
        assert_eq!(band.offset.unwrap_or(0.0), offset, "{name} band {b} offset");
        let wavelength = f["wavelengths_nm"].as_array().and_then(|w| w[b].as_f64());
        match (band.wavelength_nm, wavelength) {
            (Some(a), Some(e)) => assert!((a - e).abs() < 1e-6, "{name} band {b}: {a} nm vs {e}"),
            (a, e) => assert_eq!(a, e, "{name} band {b} wavelength"),
        }
    }
    let compression = match info.compression {
        Compression::None => "None",
        Compression::Lzw => "Lzw",
        Compression::Deflate => "Deflate",
        Compression::PackBits => "PackBits",
        Compression::Jpeg => "Jpeg",
        Compression::Zstd => "Zstd",
        Compression::Webp => "Webp",
        other => panic!("{name}: unexpected {other:?}"),
    };
    assert_eq!(compression, f["compression"], "{name}");
    assert_eq!(
        info.has_mask,
        f["has_mask"].as_bool().unwrap_or(false),
        "{name}"
    );
    if let Some(p) = f["photometric"].as_u64() {
        assert_eq!(info.photometric as u64, p, "{name}");
    }
    let planar = if f["planar"] == "Separate" {
        Planar::Separate
    } else {
        Planar::Chunky
    };
    assert_eq!(info.planar, planar, "{name}");
}

fn check_fixture(f: &Value) {
    let name = f["file"].as_str().unwrap();
    let cog = open(name);
    check_structure(f, &cog);
    let tolerance = f["tolerance"].as_f64().unwrap() as f32;
    let l0 = read_level(&cog, 0);
    compare(
        name,
        0,
        &l0,
        &expected_pixels(f["expected_l0"].as_str().unwrap()),
        tolerance,
    );
    if let Some(l1) = f["expected_l1"].as_str() {
        compare(
            name,
            1,
            &read_level(&cog, 1),
            &expected_pixels(l1),
            tolerance,
        );
    }
}

macro_rules! fixture_tests {
    ($($test:ident => $file:literal),* $(,)?) => {$(
        #[test]
        fn $test() {
            check_fixture(&fixture($file));
        }
    )*};
}

fixture_tests! {
    rgb_jpeg_with_mask_should_match_gdal => "rgb_jpeg_mask.tif",
    rgb_jpeg_rgb_photometric_should_match_gdal => "rgb_jpeg_rgb.tif",
    grey_jpeg_should_match_gdal => "grey_jpeg.tif",
    rgb_webp_should_match_gdal => "rgb_webp.tif",
    u16_deflate_predictor_should_match_gdal => "u16_5band_deflate_p2.tif",
    u16_planar_zstd_should_match_gdal => "u16_planar_zstd.tif",
    f32_lzw_float_predictor_should_match_gdal => "f32_lzw_p3.tif",
    i16_packbits_should_match_gdal => "i16_packbits.tif",
    f64_uncompressed_should_match_gdal => "f64_none.tif",
    big_endian_u16_should_match_gdal => "u16_bigendian_p2.tif",
    big_endian_f32_float_predictor_should_match_gdal => "f32_bigendian_p3.tif",
    bigtiff_u32_should_match_gdal => "u32_bigtiff.tif",
    stripped_i8_should_match_gdal => "i8_stripped.tif",
    rotated_i32_should_match_gdal => "i32_rotated.tif",
    pixel_is_point_should_match_gdal => "u8_pixel_is_point.tif",
    sparse_tiles_should_match_gdal => "u8_sparse.tif",
}

#[test]
fn every_fixture_should_have_a_test() {
    let tested = 16;
    assert_eq!(
        fixtures().len(),
        tested,
        "add new fixtures to fixture_tests!"
    );
}

/// A source that counts requests, to check reads are batched.
struct Counting {
    inner: Arc<dyn ByteSource>,
    requests: std::sync::atomic::AtomicUsize,
}

impl ByteSource for Counting {
    fn read_range(&self, offset: u64, length: u64) -> Result<Vec<u8>, SourceError> {
        self.requests
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.inner.read_range(offset, length)
    }

    fn size(&self) -> Result<Option<u64>, SourceError> {
        self.inner.size()
    }
}

fn counting(name: &str) -> Arc<Counting> {
    Arc::new(Counting {
        inner: open_source(data_dir().join(name).to_str().unwrap()).unwrap(),
        requests: Default::default(),
    })
}

#[test]
fn opening_with_a_tiny_prefix_should_read_the_same_structure() {
    // A 300-byte first read leaves IFDs and tag values outside it.
    for f in fixtures() {
        let name = f["file"].as_str().unwrap();
        let cog = Cog::from_source_with_prefix(counting(name), 300)
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        check_structure(&f, &cog);
    }
}

#[test]
fn opening_should_read_the_header_in_one_request() {
    // A COG bigger than the 64 KiB first read.
    let source = counting("f32_lzw_p3.tif");
    let cog = Cog::from_source(source.clone()).unwrap();
    assert_eq!(
        source.requests.load(std::sync::atomic::Ordering::Relaxed),
        1
    );
    // Every full-resolution tile in one merged read.
    let keys: Vec<TileKey> = (0..3)
        .flat_map(|y| (0..4).map(move |x| TileKey { level: 0, x, y }))
        .collect();
    assert!(cog.read_tiles(&keys).iter().all(|t| t.is_ok()));
    assert_eq!(
        source.requests.load(std::sync::atomic::Ordering::Relaxed),
        2
    );
}

#[test]
fn nodata_should_only_blank_the_bands_that_have_it() {
    // Band 2 (index 1) has nodata pixels at rows 5-6, columns 150-189.
    let cog = open("u16_5band_deflate_p2.tif");
    let tile = cog
        .read_tile(TileKey {
            level: 0,
            x: 2,
            y: 0,
        })
        .unwrap();
    let values: Vec<f32> = tile.pixel(150 - 128, 5).collect();
    assert!(values[1].is_nan());
    assert!(
        values
            .iter()
            .enumerate()
            .all(|(b, v)| b == 1 || !v.is_nan())
    );
}

#[test]
fn sparse_tiles_should_be_all_nan_without_reading_anything() {
    let cog = open("u8_sparse.tif");
    let tile = cog
        .read_tile(TileKey {
            level: 0,
            x: 2,
            y: 1,
        })
        .unwrap();
    assert_eq!(tile.fetched_bytes, 0);
    assert!(tile.data.iter().all(|v| v.is_nan()));
}

#[test]
fn read_tiles_should_report_bad_keys_per_tile() {
    let cog = open("i16_packbits.tif");
    let results = cog.read_tiles(&[
        TileKey {
            level: 0,
            x: 0,
            y: 0,
        },
        TileKey {
            level: 0,
            x: 99,
            y: 0,
        },
        TileKey {
            level: 9,
            x: 0,
            y: 0,
        },
    ]);
    assert!(results[0].is_ok());
    assert!(matches!(results[1], Err(CogError::Invalid(_))));
    assert!(matches!(results[2], Err(CogError::Invalid(_))));
}

#[test]
fn rgb_files_should_say_so() {
    assert!(open("rgb_jpeg_mask.tif").info().is_rgb());
    assert!(open("rgb_webp.tif").info().is_rgb());
    assert!(!open("u16_5band_deflate_p2.tif").info().is_rgb());
}

#[test]
fn describe_should_summarise_the_structure() {
    let text = cogcat::cog::describe(&open("u16_5band_deflate_p2.tif"));
    for needle in [
        "200 × 150",
        "5 × uint16",
        "Deflate",
        "horizontal",
        "EPSG:32755 (WGS 84 / UTM zone 55S)",
        "Nodata        0",
        "blue, 490 nm, scale 0.0001, offset -0.1",
        "100 × 75",
    ] {
        assert!(text.contains(needle), "missing {needle:?} in:\n{text}");
    }
    let text = cogcat::cog::describe(&open("rgb_jpeg_mask.tif"));
    assert!(
        text.contains("internal mask") && text.contains("YCbCr"),
        "{text}"
    );
}

#[test]
fn non_tiff_files_should_be_refused() {
    let path = std::env::temp_dir().join(format!("cogcat-not-tiff-{}.tif", std::process::id()));
    std::fs::write(&path, b"GIF89a this is not a tiff at all").unwrap();
    let result = Cog::open(path.to_str().unwrap());
    std::fs::remove_file(&path).unwrap();
    assert!(matches!(result, Err(CogError::NotTiff(_))));
}

#[test]
fn bad_locations_should_give_readable_errors() {
    let Err(e) = Cog::open("/definitely/not/here.tif") else {
        panic!("opened a missing file")
    };
    assert!(e.to_string().contains("/definitely/not/here.tif"), "{e}");
}
