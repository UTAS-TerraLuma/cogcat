//! Real Sentinel-2 COGs, local and remote. Ignored by default because they
//! need big files or the network:
//!
//!     cargo test --release --test cog_real -- --ignored --nocapture
//!
//! Local files come from `$COGCAT_TESTDATA` (default `/home/user/testdata`).
//! Expected pixel values were read with rasterio (GDAL 3.12).

use std::path::PathBuf;
use std::time::Instant;

use cogcat::cog::{Cog, Compression, DataType, Planar, TileKey, describe};

const TCI_URL: &str = "https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/55/G/EN/2024/1/S2A_55GEN_20240109_0_L2A/TCI.tif";
const B04_HTTPS: &str = "https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/55/G/EN/2024/1/S2A_55GEN_20240109_0_L2A/B04.tif";
const B04_S3: &str =
    "s3://sentinel-cogs/sentinel-s2-l2a-cogs/55/G/EN/2024/1/S2A_55GEN_20240109_0_L2A/B04.tif";

fn testdata(name: &str) -> String {
    let dir = std::env::var("COGCAT_TESTDATA").unwrap_or_else(|_| "/home/user/testdata".into());
    PathBuf::from(dir).join(name).to_string_lossy().into_owned()
}

/// Every band's value at a full-image pixel of a level.
fn pixel(cog: &Cog, level: u8, col: u32, row: u32) -> Vec<f32> {
    let l = &cog.levels()[level as usize];
    let key = TileKey {
        level,
        x: col / l.tile_width,
        y: row / l.tile_height,
    };
    let tile = cog.read_tile(key).unwrap();
    tile.pixel(col % l.tile_width, row % l.tile_height)
        .collect()
}

/// Times decoding a tile already fetched: reads it a few times and keeps
/// the fastest (local files, so the fetch is a page-cache copy).
fn time_tile(cog: &Cog, key: TileKey) -> f64 {
    (0..5)
        .map(|_| {
            let start = Instant::now();
            cog.read_tile(key).unwrap();
            start.elapsed().as_secs_f64() * 1000.0
        })
        .fold(f64::INFINITY, f64::min)
}

fn check_s2_multi(cog: &Cog) {
    let info = cog.info();
    assert_eq!((info.width, info.height, info.bands), (2048, 2048, 10));
    assert_eq!(info.data_type, DataType::U16);
    assert_eq!(info.nodata, Some(0.0));
    assert_eq!(info.crs.epsg, Some(32755));
    assert_eq!(
        info.geo.unwrap().0,
        [523980.0, 20.0, 0.0, 5272020.0, 0.0, -20.0]
    );
    let sizes: Vec<_> = cog.levels().iter().map(|l| l.width).collect();
    assert_eq!(sizes, [2048, 1024, 512]);
    assert_eq!(cog.levels()[0].tile_width, 512);
    let names: Vec<_> = info
        .band_info
        .iter()
        .map(|b| b.description.clone().unwrap())
        .collect();
    assert_eq!(
        names,
        [
            "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"
        ]
    );
    assert_eq!(info.band_info[0].wavelength_nm, Some(490.0));
    assert!(info.band_info.iter().all(|b| b.wavelength_nm.is_some()));
    assert_eq!(
        pixel(cog, 0, 612, 712),
        [
            1796.0, 1976.0, 2062.0, 2461.0, 2921.0, 3100.0, 3236.0, 3226.0, 3236.0, 2428.0
        ]
    );
    assert_eq!(
        pixel(cog, 0, 2047, 2047),
        [
            450.0, 569.0, 529.0, 948.0, 1525.0, 1754.0, 1913.0, 1995.0, 1353.0, 698.0
        ]
    );
    assert_eq!(
        pixel(cog, 1, 300, 50),
        [
            1001.0, 1325.0, 1521.0, 2008.0, 2594.0, 2838.0, 3058.0, 3144.0, 3645.0, 2448.0
        ]
    );
}

#[test]
#[ignore = "needs the Sentinel-2 test files"]
fn s2_pixel_interleaved_deflate_should_match_gdal() {
    let cog = Cog::open(&testdata("s2_multi_pixel_deflate.tif")).unwrap();
    println!("{}", describe(&cog));
    assert_eq!(cog.info().planar, Planar::Chunky);
    assert_eq!(cog.info().compression, Compression::Deflate);
    check_s2_multi(&cog);
    let key = TileKey {
        level: 0,
        x: 1,
        y: 1,
    };
    println!("10-band 512² Deflate tile: {:.1} ms", time_tile(&cog, key));
}

#[test]
#[ignore = "needs the Sentinel-2 test files"]
fn s2_band_interleaved_zstd_should_match_gdal() {
    let cog = Cog::open(&testdata("s2_multi_band_zstd.tif")).unwrap();
    assert_eq!(cog.info().planar, Planar::Separate);
    assert_eq!(cog.info().compression, Compression::Zstd);
    check_s2_multi(&cog);
    let key = TileKey {
        level: 0,
        x: 1,
        y: 1,
    };
    println!("10-band 512² ZSTD tile: {:.1} ms", time_tile(&cog, key));
}

#[test]
#[ignore = "needs the Sentinel-2 test files"]
fn s2_layouts_should_decode_to_the_same_pixels() {
    let pixel_cog = Cog::open(&testdata("s2_multi_pixel_deflate.tif")).unwrap();
    let band_cog = Cog::open(&testdata("s2_multi_band_zstd.tif")).unwrap();
    for level in pixel_cog.levels() {
        let keys: Vec<TileKey> = (0..level.tiles_down)
            .flat_map(|y| {
                (0..level.tiles_across).map(move |x| TileKey {
                    level: level.index,
                    x,
                    y,
                })
            })
            .collect();
        let a = pixel_cog.read_tiles(&keys);
        let b = band_cog.read_tiles(&keys);
        for (a, b) in a.into_iter().zip(b) {
            let (a, b) = (a.unwrap(), b.unwrap());
            let same = a
                .data
                .iter()
                .zip(&b.data)
                .all(|(x, y)| x == y || (x.is_nan() && y.is_nan()));
            assert!(same, "tile {} differs", a.key);
        }
    }
}

#[test]
#[ignore = "needs the network"]
fn remote_tci_over_https_should_match_gdal() {
    let start = Instant::now();
    let cog = Cog::open(TCI_URL).unwrap();
    println!("opened in {:.0} ms", start.elapsed().as_secs_f64() * 1000.0);
    println!("{}", describe(&cog));
    let info = cog.info();
    assert_eq!((info.width, info.height, info.bands), (10980, 10980, 3));
    assert!(info.is_rgb());
    assert_eq!(info.compression, Compression::Deflate);
    assert_eq!(info.crs.epsg, Some(32755));
    let sizes: Vec<_> = cog.levels().iter().map(|l| l.width).collect();
    assert_eq!(sizes, [10980, 5490, 2745, 1373, 687]);
    assert_eq!(cog.levels()[0].tile_width, 1024);
    let start = Instant::now();
    assert_eq!(pixel(&cog, 0, 5000, 6000), [27.0, 34.0, 40.0]);
    println!(
        "fetched and decoded a 1024² tile in {:.0} ms",
        start.elapsed().as_secs_f64() * 1000.0
    );
    assert_eq!(pixel(&cog, 0, 1030, 10), [34.0, 35.0, 21.0]);
    assert_eq!(pixel(&cog, 4, 300, 400), [254.0, 255.0, 255.0]);
    // Several tiles at once, in merged parallel requests.
    let keys: Vec<TileKey> = (0..3)
        .flat_map(|y| (0..3).map(move |x| TileKey { level: 2, x, y }))
        .collect();
    let start = Instant::now();
    let tiles = cog.read_tiles(&keys);
    let bytes: u64 = tiles
        .iter()
        .map(|t| t.as_ref().unwrap().fetched_bytes)
        .sum();
    println!(
        "9 overview tiles, {bytes} bytes, in {:.0} ms",
        start.elapsed().as_secs_f64() * 1000.0
    );
}

#[test]
#[ignore = "needs the network"]
fn remote_b04_over_s3_should_match_https() {
    let start = Instant::now();
    let s3 = Cog::open(B04_S3).unwrap_or_else(|e| panic!("{e}"));
    println!(
        "opened over s3:// in {:.0} ms",
        start.elapsed().as_secs_f64() * 1000.0
    );
    let https = Cog::open(B04_HTTPS).unwrap();
    assert_eq!(s3.info().width, 10980);
    assert_eq!(s3.info().data_type, DataType::U16);
    assert_eq!(s3.info().file_size, https.info().file_size);
    assert_eq!(pixel(&s3, 0, 5000, 6000), [259.0]);
    let key = TileKey {
        level: 3,
        x: 1,
        y: 0,
    };
    let (a, b) = (s3.read_tile(key).unwrap(), https.read_tile(key).unwrap());
    assert!(
        a.data
            .iter()
            .zip(&b.data)
            .all(|(x, y)| x == y || (x.is_nan() && y.is_nan()))
    );
}

#[test]
#[ignore = "needs the network"]
fn missing_remote_files_should_say_so() {
    let Err(e) = Cog::open(&TCI_URL.replace("TCI.tif", "NOPE.tif")) else {
        panic!("opened a missing file");
    };
    let message = e.to_string();
    println!("{message}");
    assert!(message.contains("NOPE.tif"), "{message}");
}
