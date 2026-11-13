//! Reading Cloud Optimized GeoTIFFs: the header and IFDs once on open, then
//! tiles on demand.

mod decode;
mod geo;
mod metadata;
mod tiff;

use std::sync::Arc;

pub use geo::{Crs, GeoTransform};

use crate::source::{ByteSource, SourceError};

#[derive(Debug, thiserror::Error)]
pub enum CogError {
    #[error(transparent)]
    Source(#[from] SourceError),
    #[error("not a TIFF: {0}")]
    NotTiff(String),
    #[error("invalid TIFF: {0}")]
    Invalid(String),
    #[error("unsupported: {0}")]
    Unsupported(String),
    #[error("couldn't decode tile {key}: {message}")]
    Decode { key: TileKey, message: String },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DataType {
    U8,
    I8,
    U16,
    I16,
    U32,
    I32,
    F32,
    F64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Compression {
    None,
    Lzw,
    Deflate,
    PackBits,
    Jpeg,
    Zstd,
    Webp,
    Lerc,
    Other(u16),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Planar {
    /// Each tile holds every band, pixel by pixel (`PlanarConfiguration` 1).
    Chunky,
    /// Each band has its own tiles (`PlanarConfiguration` 2).
    Separate,
}

/// What the file says about one band.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct BandInfo {
    /// GDAL's band description, e.g. `B04` or `red`.
    pub description: Option<String>,
    /// Centre wavelength in nanometres, from GDAL metadata (`wavelength`,
    /// `CENTRAL_WAVELENGTH_UM`, ENVI-style `wavelength` lists, ...), converted
    /// from µm when the units say so.
    pub wavelength_nm: Option<f64>,
    pub scale: Option<f64>,
    pub offset: Option<f64>,
}

/// The full-resolution image's properties.
#[derive(Clone, Debug)]
pub struct CogInfo {
    pub width: u32,
    pub height: u32,
    /// Samples per pixel, including any alpha sample.
    pub bands: u32,
    pub data_type: DataType,
    pub compression: Compression,
    pub planar: Planar,
    /// `PhotometricInterpretation`: 1 min-is-black, 2 RGB, 6 YCbCr, ...
    pub photometric: u16,
    /// The band index of an alpha sample (`ExtraSamples` 1 or 2), if any.
    pub alpha_band: Option<u32>,
    /// `GDAL_NODATA`. Pixels with this value decode to NaN.
    pub nodata: Option<f64>,
    /// Whether the file has an internal mask (a `NewSubfileType` 4 IFD).
    pub has_mask: bool,
    /// Full-resolution pixel (col, row) to CRS coordinates. `None` for a
    /// TIFF without geo tags; the viewer then uses pixel coordinates with y up.
    pub geo: Option<GeoTransform>,
    pub crs: Crs,
    /// One per band.
    pub band_info: Vec<BandInfo>,
    /// The file's size in bytes, if the source knows it.
    pub file_size: Option<u64>,
}

impl CogInfo {
    /// An 8-bit RGB(A) image, which opens as true colour with no spectral view.
    pub fn is_rgb(&self) -> bool {
        self.data_type == DataType::U8
            && matches!(self.photometric, 2 | 6)
            && (self.bands == 3 || (self.bands == 4 && self.alpha_band == Some(3)))
    }
}

/// One resolution: the full image (index 0) or an overview.
#[derive(Clone, Debug)]
pub struct Level {
    pub index: u8,
    pub width: u32,
    pub height: u32,
    pub tile_width: u32,
    pub tile_height: u32,
    pub tiles_across: u32,
    pub tiles_down: u32,
    /// Full-resolution pixels per pixel of this level, across and down.
    pub scale: [f64; 2],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TileKey {
    /// 0 is full resolution; higher is coarser.
    pub level: u8,
    pub x: u32,
    pub y: u32,
}

impl std::fmt::Display for TileKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}/{}/{}", self.level, self.x, self.y)
    }
}

/// A decoded tile.
#[derive(Clone, Debug)]
pub struct TileData {
    pub key: TileKey,
    /// The tile's full size in pixels: the level's tile size.
    pub width: u32,
    pub height: u32,
    /// How much of the tile is inside the image: less than `width` and
    /// `height` for tiles on the right and bottom edges.
    pub valid: [u32; 2],
    pub bands: u32,
    /// Band-sequential: band `b`, row `y`, column `x` is
    /// `data[(b * height + y) * width + x]`. Nodata, masked, and (for
    /// floats) NaN pixels are NaN. Integer samples are converted to f32
    /// without scaling.
    pub data: Vec<f32>,
    /// Bytes fetched to make this tile.
    pub fetched_bytes: u64,
}

impl TileData {
    pub fn band(&self, band: u32) -> &[f32] {
        let size = (self.width * self.height) as usize;
        &self.data[band as usize * size..][..size]
    }

    /// Every band's value at a pixel of the tile.
    pub fn pixel(&self, x: u32, y: u32) -> impl Iterator<Item = f32> + '_ {
        let size = (self.width * self.height) as usize;
        let at = (y * self.width + x) as usize;
        (0..self.bands as usize).map(move |b| self.data[b * size + at])
    }
}

/// An open COG.
pub struct Cog {
    info: CogInfo,
    levels: Vec<Level>,
    source: Arc<dyn ByteSource>,
    // Agent A: whatever else is needed (per-level tile offsets, mask IFDs,
    // JPEG tables, predictor, ...).
}

impl Cog {
    /// Opens a path or URL; see [`crate::source::open_source`].
    pub fn open(location: &str) -> Result<Cog, CogError> {
        Self::from_source(crate::source::open_source(location)?)
    }

    /// Reads the header and every IFD. Tiles are read later.
    pub fn from_source(source: Arc<dyn ByteSource>) -> Result<Cog, CogError> {
        let _ = source;
        todo!("agent A")
    }

    pub fn info(&self) -> &CogInfo {
        &self.info
    }

    /// Full resolution first, then overviews, coarser and coarser.
    pub fn levels(&self) -> &[Level] {
        &self.levels
    }

    /// Reads and decodes tiles. Nearby byte ranges are merged into fewer
    /// requests. Results are in the order of `keys`. A tile with no bytes
    /// (sparse) decodes to all NaN.
    pub fn read_tiles(&self, keys: &[TileKey]) -> Vec<Result<TileData, CogError>> {
        let _ = (keys, &self.source);
        todo!("agent A")
    }

    pub fn read_tile(&self, key: TileKey) -> Result<TileData, CogError> {
        self.read_tiles(&[key]).pop().expect("one result per key")
    }
}
