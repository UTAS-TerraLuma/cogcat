//! Reading Cloud Optimized GeoTIFFs: the header and IFDs once on open, then
//! tiles on demand.

mod decode;
mod geo;
mod metadata;
mod tiff;

use std::sync::Arc;

pub use geo::{Crs, GeoTransform};

use crate::source::{ByteSource, SourceError};
use decode::{ChunkLayout, Placement};
use geo::GeoTags;
use tiff::{ByteOrder, Ifd};

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
    /// `CENTRAL_WAVELENGTH_UM`, `WAVELENGTH_NM`, ENVI-style `wavelength`
    /// lists, ...), converted from µm when the units say so.
    pub wavelength_nm: Option<f64>,
    /// Full width at half maximum (the band's width) in nanometres, from
    /// `FWHM_NM`, `FWHM_UM` or `FWHM`.
    pub fwhm_nm: Option<f64>,
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
    /// `height` for tiles on the right and bottom edges. Pixels outside it
    /// are NaN.
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
    /// Where each level's tiles are, parallel to `levels`.
    images: Vec<LevelImages>,
    order: ByteOrder,
    big_tiff: bool,
    predictor: u16,
    /// Whether the full-resolution image is stored in strips, not tiles.
    stripped: bool,
}

/// One TIFF image (IFD): its chunks (tiles, or strips) and how they're stored.
#[derive(Clone, Debug)]
struct Image {
    width: u32,
    height: u32,
    /// Tile size, or for strips the image width and rows per strip.
    chunk_width: u32,
    chunk_height: u32,
    tiled: bool,
    chunks_across: u32,
    chunks_down: u32,
    /// Per chunk, plane by plane for `Planar::Separate`.
    offsets: Vec<u64>,
    byte_counts: Vec<u64>,
    planar: Planar,
    layout: ChunkLayout,
    samples_per_pixel: u32,
}

impl Image {
    fn planes(&self) -> u32 {
        match self.planar {
            Planar::Chunky => 1,
            Planar::Separate => self.samples_per_pixel,
        }
    }
}

/// A level's image and its internal mask.
#[derive(Clone, Debug)]
struct LevelImages {
    image: Image,
    mask: Option<Image>,
    /// Strips stacked into one tile; 1 for tiled images.
    strips_per_tile: u32,
}

/// One chunk a tile needs.
#[derive(Clone, Copy, Debug)]
struct ChunkRead {
    offset: u64,
    length: u64,
    /// The band plane (`Planar::Separate`), else 0.
    plane: u32,
    /// The chunk's first row in the tile, and its rows.
    row0: u32,
    rows: u32,
    mask: bool,
}

/// Byte gaps up to this size between tiles are read rather than skipped.
const MERGE_GAP: u64 = 64 * 1024;
/// Merged requests stop growing at this size.
const MERGE_MAX: u64 = 4 * 1024 * 1024;
/// A sanity limit on one chunk's compressed size.
const MAX_CHUNK_BYTES: u64 = 512 * 1024 * 1024;
/// Stripped images bigger than this have no overviews and too-wide strips
/// to view well, so they're refused.
const MAX_STRIPPED_SIZE: u32 = 4096;
/// Strips are grouped into tiles about this tall.
const STRIP_TILE_ROWS: u32 = 256;
/// Upper bound on threads decoding one batch of tiles.
const MAX_DECODE_THREADS: usize = 8;

impl Cog {
    /// Opens a path or URL; see [`crate::source::open_source`].
    pub fn open(location: &str) -> Result<Cog, CogError> {
        Self::from_source(crate::source::open_source(location)?)
    }

    /// Reads the header and every IFD. Tiles are read later.
    pub fn from_source(source: Arc<dyn ByteSource>) -> Result<Cog, CogError> {
        Self::from_source_with_prefix(source, tiff::PREFIX_SIZE)
    }

    /// [`Cog::from_source`] reading `prefix` bytes up front. Tests use small
    /// prefixes to exercise reading IFDs and tag data beyond it.
    #[doc(hidden)]
    pub fn from_source_with_prefix(
        source: Arc<dyn ByteSource>,
        prefix: u64,
    ) -> Result<Cog, CogError> {
        // Read first and ask the size after: an HTTP source learns the size
        // from the range reply, saving a round trip.
        let head = match source.read_range(0, prefix) {
            Ok(head) => head,
            // A file shorter than the prefix: read all of it.
            Err(e) => match source.size() {
                Ok(Some(size)) if size < prefix => source.read_range(0, size)?,
                _ => return Err(e.into()),
            },
        };
        let file_size = source.size()?;
        let (header, ifds) = tiff::read_ifds(&*source, head, file_size)?;
        Self::from_ifds(source, header, &ifds, file_size)
    }

    fn from_ifds(
        source: Arc<dyn ByteSource>,
        header: tiff::Header,
        ifds: &[Ifd],
        file_size: Option<u64>,
    ) -> Result<Cog, CogError> {
        // The full-resolution image: the first IFD that is neither reduced
        // nor a mask (GDAL puts it first).
        let main_index = ifds
            .iter()
            .position(|ifd| (ifd.u64(tiff::NEW_SUBFILE_TYPE).ok().flatten().unwrap_or(0) & 5) == 0)
            .unwrap_or(0);
        let main_ifd = &ifds[main_index];
        let main = parse_image(main_ifd, false)?;
        check_supported(&main)?;
        if !main.tiled && (main.width > MAX_STRIPPED_SIZE || main.height > MAX_STRIPPED_SIZE) {
            return Err(CogError::Unsupported(format!(
                "this TIFF is {} × {} pixels and stored in strips, not tiles, with no \
                 overviews; convert it to a Cloud Optimized GeoTIFF first, e.g. \
                 `gdal_translate -of COG in.tif out.tif`",
                main.width, main.height
            )));
        }

        let mut images = vec![main.clone()];
        let mut masks = Vec::new();
        for (i, ifd) in ifds.iter().enumerate() {
            if i == main_index {
                continue;
            }
            let subfile = ifd.u64(tiff::NEW_SUBFILE_TYPE).ok().flatten().unwrap_or(0);
            if subfile & 4 != 0 {
                if let Ok(mask) = parse_image(ifd, true) {
                    masks.push(mask);
                }
            } else if subfile & 1 != 0 {
                // An overview, if it's tiled and matches the main image.
                if let Ok(image) = parse_image(ifd, false)
                    && image.tiled
                    && image.width < main.width
                    && image.layout.data_type == main.layout.data_type
                    && image.samples_per_pixel == main.samples_per_pixel
                    && check_supported(&image).is_ok()
                    && !images
                        .iter()
                        .any(|o: &Image| o.width == image.width && o.height == image.height)
                {
                    images.push(image);
                }
            }
        }
        images.sort_by(|a, b| b.width.cmp(&a.width).then(b.height.cmp(&a.height)));
        images.truncate(u8::MAX as usize);

        let level_images: Vec<LevelImages> = images
            .into_iter()
            .map(|image| {
                let mask = masks
                    .iter()
                    .find(|m| {
                        m.width == image.width
                            && m.height == image.height
                            && m.chunk_width == image.chunk_width
                            && m.chunk_height == image.chunk_height
                            && m.planes() == 1
                    })
                    .cloned();
                let strips_per_tile = if image.tiled {
                    1
                } else {
                    STRIP_TILE_ROWS.div_ceil(image.chunk_height).max(1)
                };
                LevelImages {
                    image,
                    mask,
                    strips_per_tile,
                }
            })
            .collect();

        let levels =
            level_images
                .iter()
                .enumerate()
                .map(|(index, li)| {
                    let image = &li.image;
                    let tile_width = image.chunk_width;
                    let tile_height = (image.chunk_height * li.strips_per_tile)
                        .min(if image.tiled { u32::MAX } else { image.height });
                    Level {
                        index: index as u8,
                        width: image.width,
                        height: image.height,
                        tile_width,
                        tile_height,
                        tiles_across: image.width.div_ceil(tile_width),
                        tiles_down: image.height.div_ceil(tile_height),
                        scale: [
                            main.width as f64 / image.width as f64,
                            main.height as f64 / image.height as f64,
                        ],
                    }
                })
                .collect();

        let info = cog_info(main_ifd, &main, level_images[0].mask.is_some(), file_size)?;
        Ok(Cog {
            info,
            levels,
            source,
            images: level_images,
            order: header.order,
            big_tiff: header.big_tiff,
            predictor: main.layout.predictor,
            stripped: !main.tiled,
        })
    }

    pub fn info(&self) -> &CogInfo {
        &self.info
    }

    /// Full resolution first, then overviews, coarser and coarser.
    pub fn levels(&self) -> &[Level] {
        &self.levels
    }

    /// The source the COG is read from.
    pub fn source(&self) -> &Arc<dyn ByteSource> {
        &self.source
    }

    /// Reads and decodes tiles. Nearby byte ranges are merged into fewer
    /// requests. Results are in the order of `keys`. A tile with no bytes
    /// (sparse) decodes to all NaN.
    pub fn read_tiles(&self, keys: &[TileKey]) -> Vec<Result<TileData, CogError>> {
        let plans: Vec<Result<Vec<ChunkRead>, CogError>> =
            keys.iter().map(|&key| self.plan_tile(key)).collect();
        let wanted: Vec<(u64, u64)> = plans
            .iter()
            .flatten()
            .flatten()
            .filter(|c| c.length > 0)
            .map(|c| (c.offset, c.length))
            .collect();
        let merged = merge_ranges(wanted);
        let fetched = if merged.is_empty() {
            Ok(Vec::new())
        } else {
            self.source.read_ranges(&merged)
        };
        let fetched = match fetched {
            Ok(fetched) => fetched,
            Err(e) => {
                return plans
                    .into_iter()
                    .map(|plan| plan.and(Err(CogError::Source(copy_source_error(&e)))))
                    .collect();
            }
        };
        let jobs: Vec<(TileKey, Result<Vec<ChunkRead>, CogError>)> =
            keys.iter().copied().zip(plans).collect();
        // With fewer tiles than threads, decode each tile's chunks (band
        // planes, strips) in parallel too.
        let parallel_chunks = jobs.len() < decode_threads();
        let decode = |(key, plan): &(TileKey, Result<Vec<ChunkRead>, CogError>)| match plan {
            Ok(chunks) => self.decode_tile(*key, chunks, &merged, &fetched, parallel_chunks),
            Err(e) => Err(copy_cog_error(e)),
        };
        parallel_map(&jobs, decode)
    }

    pub fn read_tile(&self, key: TileKey) -> Result<TileData, CogError> {
        self.read_tiles(&[key]).pop().expect("one result per key")
    }

    /// The chunks a tile is made of, data and mask.
    fn plan_tile(&self, key: TileKey) -> Result<Vec<ChunkRead>, CogError> {
        let (Some(level), Some(li)) = (
            self.levels.get(key.level as usize),
            self.images.get(key.level as usize),
        ) else {
            return Err(CogError::Invalid(format!(
                "no level {} in tile {key}",
                key.level
            )));
        };
        if key.x >= level.tiles_across || key.y >= level.tiles_down {
            return Err(CogError::Invalid(format!(
                "tile {key} is outside level {}'s {} × {} tiles",
                key.level, level.tiles_across, level.tiles_down
            )));
        }
        let mut reads = Vec::new();
        let mut add = |image: &Image, mask: bool| -> Result<(), CogError> {
            let per_plane = (image.chunks_across * image.chunks_down) as usize;
            for s in 0..li.strips_per_tile {
                let chunk_row = key.y * li.strips_per_tile + s;
                if chunk_row >= image.chunks_down {
                    break;
                }
                let rows = if image.tiled {
                    image.chunk_height
                } else {
                    image
                        .chunk_height
                        .min(image.height - chunk_row * image.chunk_height)
                };
                for plane in 0..image.planes() {
                    let index = plane as usize * per_plane
                        + chunk_row as usize * image.chunks_across as usize
                        + key.x as usize;
                    let (offset, length) = (image.offsets[index], image.byte_counts[index]);
                    if length > MAX_CHUNK_BYTES {
                        return Err(CogError::Invalid(format!(
                            "tile {key} claims {length} bytes"
                        )));
                    }
                    reads.push(ChunkRead {
                        offset,
                        length: if offset == 0 { 0 } else { length },
                        plane,
                        row0: s * image.chunk_height,
                        rows,
                        mask,
                    });
                }
            }
            Ok(())
        };
        add(&li.image, false)?;
        if let Some(mask) = &li.mask {
            add(mask, true)?;
        }
        Ok(reads)
    }

    fn decode_tile(
        &self,
        key: TileKey,
        chunks: &[ChunkRead],
        merged: &[(u64, u64)],
        fetched: &[Vec<u8>],
        parallel_chunks: bool,
    ) -> Result<TileData, CogError> {
        let level = &self.levels[key.level as usize];
        let li = &self.images[key.level as usize];
        let (width, height) = (level.tile_width, level.tile_height);
        let bands = self.info.bands;
        let valid = [
            width.min(level.width - key.x * width),
            height.min(level.height - key.y * height),
        ];
        let mut data = vec![f32::NAN; width as usize * height as usize * bands as usize];
        let mut fetched_bytes = 0;
        let error = |message: String| CogError::Decode { key, message };
        // Data first, then the mask over it.
        let image = &li.image;
        let data_chunks: Vec<&ChunkRead> =
            chunks.iter().filter(|c| !c.mask && c.length > 0).collect();
        let decode_chunk = |chunk: &&ChunkRead| {
            let bytes = slice_fetched(merged, fetched, chunk.offset, chunk.length);
            decode::decode_chunk(&image.layout, bytes, image.chunk_width, chunk.rows)
        };
        let mut place = |chunk: &ChunkRead, decoded: Result<decode::Chunk, String>| {
            let placement = Placement {
                tile_width: width,
                tile_height: height,
                bands,
                valid,
                row0: chunk.row0,
                first_band: chunk.plane,
                nodata: self.info.nodata,
            };
            decode::place(&decoded.map_err(error)?, &mut data, &placement);
            fetched_bytes += chunk.length;
            Ok::<(), CogError>(())
        };
        if parallel_chunks && data_chunks.len() > 1 {
            let decoded = parallel_map(&data_chunks, decode_chunk);
            for (chunk, decoded) in data_chunks.iter().zip(decoded) {
                place(chunk, decoded)?;
            }
        } else {
            for chunk in &data_chunks {
                place(chunk, decode_chunk(chunk))?;
            }
        }
        if let Some(mask) = &li.mask {
            for chunk in chunks.iter().filter(|c| c.mask) {
                let placement = Placement {
                    tile_width: width,
                    tile_height: height,
                    bands,
                    valid,
                    row0: chunk.row0,
                    first_band: 0,
                    nodata: None,
                };
                let pixels = if chunk.length == 0 {
                    // A sparse mask chunk masks everything out, as in GDAL.
                    vec![0; (mask.chunk_width * chunk.rows) as usize]
                } else {
                    fetched_bytes += chunk.length;
                    let bytes = slice_fetched(merged, fetched, chunk.offset, chunk.length);
                    decode::decode_mask(&mask.layout, bytes, mask.chunk_width, chunk.rows)
                        .map_err(error)?
                };
                decode::apply_mask(&pixels, mask.chunk_width, chunk.rows, &mut data, &placement);
            }
        }
        Ok(TileData {
            key,
            width,
            height,
            valid,
            bands,
            data,
            fetched_bytes,
        })
    }
}

/// Builds [`CogInfo`] from the full-resolution IFD.
fn cog_info(
    ifd: &Ifd,
    image: &Image,
    has_mask: bool,
    file_size: Option<u64>,
) -> Result<CogInfo, CogError> {
    let bands = image.samples_per_pixel;
    let extra = ifd.u64s(tiff::EXTRA_SAMPLES)?.unwrap_or_default();
    let first_extra = bands.saturating_sub(extra.len() as u32);
    let alpha_band = extra
        .iter()
        .position(|&e| e == 1 || e == 2)
        .map(|i| first_extra + i as u32);
    let nodata = ifd
        .ascii(tiff::GDAL_NODATA)
        .and_then(|text| metadata::parse_nodata(&text));
    let items = ifd
        .ascii(tiff::GDAL_METADATA)
        .map(|xml| metadata::parse_items(&xml))
        .unwrap_or_default();
    let geo_tags = GeoTags {
        pixel_scale: ifd.f64s(tiff::MODEL_PIXEL_SCALE)?,
        tiepoints: ifd.f64s(tiff::MODEL_TIEPOINT)?,
        transformation: ifd.f64s(tiff::MODEL_TRANSFORMATION)?,
        key_directory: ifd
            .u64s(tiff::GEO_KEY_DIRECTORY)?
            .map(|keys| keys.into_iter().map(|k| k as u16).collect()),
        double_params: ifd.f64s(tiff::GEO_DOUBLE_PARAMS)?,
        ascii_params: ifd.ascii(tiff::GEO_ASCII_PARAMS),
    };
    Ok(CogInfo {
        width: image.width,
        height: image.height,
        bands,
        data_type: image.layout.data_type,
        compression: image.layout.compression,
        planar: image.planar,
        photometric: image.layout.photometric,
        alpha_band,
        nodata,
        has_mask,
        geo: geo::geo_transform(&geo_tags),
        crs: geo::crs(&geo_tags),
        band_info: metadata::band_info(&items, bands),
        file_size,
    })
}

fn compression_from_code(code: u64) -> Compression {
    match code {
        1 => Compression::None,
        5 => Compression::Lzw,
        8 | 32946 => Compression::Deflate,
        32773 => Compression::PackBits,
        7 => Compression::Jpeg,
        50000 => Compression::Zstd,
        50001 => Compression::Webp,
        34887 => Compression::Lerc,
        other => Compression::Other(other.min(u16::MAX as u64) as u16),
    }
}

/// A name for a compression, e.g. for `cogcat info`.
pub fn compression_name(compression: Compression) -> String {
    match compression {
        Compression::None => "none".into(),
        Compression::Lzw => "LZW".into(),
        Compression::Deflate => "Deflate".into(),
        Compression::PackBits => "PackBits".into(),
        Compression::Jpeg => "JPEG".into(),
        Compression::Zstd => "ZSTD".into(),
        Compression::Webp => "WebP".into(),
        Compression::Lerc => "LERC".into(),
        Compression::Other(code) => {
            let name = match code {
                2 => "CCITT RLE",
                3 => "CCITT Group 3 fax",
                4 => "CCITT Group 4 fax",
                6 => "old-style JPEG",
                34712 => "JPEG 2000",
                34925 => "LZMA",
                50002 | 52546 => "JPEG XL",
                _ => return format!("code {code}"),
            };
            format!("{name} (code {code})")
        }
    }
}

fn data_type(bits: u64, format: u64) -> Option<DataType> {
    Some(match (format, bits) {
        (1, 8) => DataType::U8,
        (2, 8) => DataType::I8,
        (1, 16) => DataType::U16,
        (2, 16) => DataType::I16,
        (1, 32) => DataType::U32,
        (2, 32) => DataType::I32,
        (3, 32) => DataType::F32,
        (3, 64) => DataType::F64,
        _ => return None,
    })
}

/// Reads one IFD's image structure. Masks may be 1-bit.
fn parse_image(ifd: &Ifd, is_mask: bool) -> Result<Image, CogError> {
    let at = ifd.offset;
    let required = |tag: u16, name: &str| -> Result<u64, CogError> {
        ifd.u64(tag)?
            .ok_or_else(|| CogError::Invalid(format!("the IFD at offset {at} has no {name}")))
    };
    let width = required(tiff::IMAGE_WIDTH, "ImageWidth")?;
    let height = required(tiff::IMAGE_LENGTH, "ImageLength")?;
    if width == 0 || height == 0 || width > u32::MAX as u64 || height > u32::MAX as u64 {
        return Err(CogError::Invalid(format!("image size {width} × {height}")));
    }
    let (width, height) = (width as u32, height as u32);
    let samples_per_pixel = ifd.u64(tiff::SAMPLES_PER_PIXEL)?.unwrap_or(1);
    if samples_per_pixel == 0 || samples_per_pixel > 65535 {
        return Err(CogError::Invalid(format!(
            "{samples_per_pixel} samples per pixel"
        )));
    }
    let samples_per_pixel = samples_per_pixel as u32;
    let bits = ifd.u64s(tiff::BITS_PER_SAMPLE)?.unwrap_or_else(|| vec![1]);
    if bits.iter().any(|&b| b != bits[0]) {
        return Err(CogError::Unsupported(format!(
            "bands with different bit depths ({bits:?})"
        )));
    }
    let bits = bits[0];
    let formats = ifd.u64s(tiff::SAMPLE_FORMAT)?.unwrap_or_else(|| vec![1]);
    if formats.iter().any(|&f| f != formats[0]) {
        return Err(CogError::Unsupported(format!(
            "bands with different sample formats ({formats:?})"
        )));
    }
    let data_type = match data_type(bits, formats[0]) {
        Some(t) => t,
        None if is_mask && bits == 1 => DataType::U8,
        None => {
            let kind = match formats[0] {
                1 => "unsigned integer",
                2 => "signed integer",
                3 => "floating-point",
                5 | 6 => "complex",
                _ => "unknown-format",
            };
            return Err(CogError::Unsupported(format!("{bits}-bit {kind} samples")));
        }
    };
    let planar = match ifd.u64(tiff::PLANAR_CONFIGURATION)?.unwrap_or(1) {
        2 if samples_per_pixel > 1 => Planar::Separate,
        _ => Planar::Chunky,
    };
    let tiled = ifd.has(tiff::TILE_WIDTH);
    let (chunk_width, chunk_height, offsets, byte_counts) = if tiled {
        let tw = required(tiff::TILE_WIDTH, "TileWidth")?;
        let th = required(tiff::TILE_LENGTH, "TileLength")?;
        let offsets = ifd.u64s(tiff::TILE_OFFSETS)?.unwrap_or_default();
        let counts = ifd.u64s(tiff::TILE_BYTE_COUNTS)?.unwrap_or_default();
        (tw, th, offsets, counts)
    } else {
        let rows = ifd.u64(tiff::ROWS_PER_STRIP)?.unwrap_or(height as u64);
        let offsets = ifd.u64s(tiff::STRIP_OFFSETS)?.unwrap_or_default();
        let counts = ifd.u64s(tiff::STRIP_BYTE_COUNTS)?.unwrap_or_default();
        (width as u64, rows.clamp(1, height as u64), offsets, counts)
    };
    if chunk_width == 0 || chunk_height == 0 || chunk_width > 65536 || chunk_height > 65536 {
        return Err(CogError::Unsupported(format!(
            "blocks of {chunk_width} × {chunk_height} pixels"
        )));
    }
    let (chunk_width, chunk_height) = (chunk_width as u32, chunk_height as u32);
    let chunks_across = width.div_ceil(chunk_width);
    let chunks_down = height.div_ceil(chunk_height);
    let planes = if planar == Planar::Separate {
        samples_per_pixel
    } else {
        1
    };
    let needed = chunks_across as u64 * chunks_down as u64 * planes as u64;
    if (offsets.len() as u64) < needed || (byte_counts.len() as u64) < needed {
        return Err(CogError::Invalid(format!(
            "the IFD at offset {at} has {} offsets and {} byte counts for {needed} {}",
            offsets.len(),
            byte_counts.len(),
            if tiled { "tiles" } else { "strips" }
        )));
    }
    let compression = compression_from_code(ifd.u64(tiff::COMPRESSION)?.unwrap_or(1));
    let photometric = ifd.u64(tiff::PHOTOMETRIC)?.unwrap_or(1) as u16;
    Ok(Image {
        width,
        height,
        chunk_width,
        chunk_height,
        tiled,
        chunks_across,
        chunks_down,
        offsets,
        byte_counts,
        planar,
        layout: ChunkLayout {
            compression,
            predictor: ifd.u64(tiff::PREDICTOR)?.unwrap_or(1) as u16,
            order: ifd.order,
            bits: bits as u16,
            data_type,
            samples: if planar == Planar::Separate {
                1
            } else {
                samples_per_pixel
            },
            photometric,
            jpeg_tables: ifd.bytes(tiff::JPEG_TABLES).map(Arc::from),
        },
        samples_per_pixel,
    })
}

/// Refuses images this reader can't decode, with a reason for the user.
fn check_supported(image: &Image) -> Result<(), CogError> {
    let layout = &image.layout;
    match layout.compression {
        Compression::Lerc => {
            return Err(CogError::Unsupported(
                "LERC compression (34887) is not supported yet".into(),
            ));
        }
        Compression::Other(_) => {
            return Err(CogError::Unsupported(format!(
                "{} compression is not supported",
                compression_name(layout.compression)
            )));
        }
        Compression::Jpeg if layout.data_type != DataType::U8 => {
            return Err(CogError::Unsupported(
                "JPEG compression with samples other than 8-bit".into(),
            ));
        }
        Compression::Webp
            if layout.data_type != DataType::U8
                || image.planar == Planar::Separate
                || !matches!(image.samples_per_pixel, 3 | 4) =>
        {
            return Err(CogError::Unsupported(
                "WebP compression of anything but 8-bit RGB or RGBA".into(),
            ));
        }
        _ => {}
    }
    if layout.photometric == 6 && layout.compression != Compression::Jpeg {
        return Err(CogError::Unsupported(
            "YCbCr samples without JPEG compression".into(),
        ));
    }
    match layout.predictor {
        1 | 2 => {}
        3 if matches!(layout.data_type, DataType::F32 | DataType::F64) => {}
        p => {
            return Err(CogError::Unsupported(format!(
                "predictor {p} with {:?} samples",
                layout.data_type
            )));
        }
    }
    Ok(())
}

/// Sorts and merges byte ranges that are close together, so fewer requests
/// fetch them. Merged ranges cover every input range.
fn merge_ranges(mut ranges: Vec<(u64, u64)>) -> Vec<(u64, u64)> {
    ranges.sort_unstable();
    ranges.dedup();
    let mut merged: Vec<(u64, u64)> = Vec::new();
    for (offset, length) in ranges {
        let end = offset + length;
        if let Some((start, len)) = merged.last_mut() {
            let current_end = *start + *len;
            if offset <= current_end + MERGE_GAP && end.max(current_end) - *start <= MERGE_MAX {
                *len = end.max(current_end) - *start;
                continue;
            }
            if end <= current_end {
                continue;
            }
        }
        merged.push((offset, length));
    }
    merged
}

/// The bytes of one chunk within the merged reads.
fn slice_fetched<'a>(
    merged: &[(u64, u64)],
    fetched: &'a [Vec<u8>],
    offset: u64,
    length: u64,
) -> &'a [u8] {
    // The last merged range starting at or before `offset` contains it.
    let i = merged.partition_point(|&(start, _)| start <= offset) - 1;
    let start = (offset - merged[i].0) as usize;
    &fetched[i][start..start + length as usize]
}

/// How many threads decode a batch of tiles.
fn decode_threads() -> usize {
    std::thread::available_parallelism()
        .map_or(4, |n| n.get())
        .min(MAX_DECODE_THREADS)
}

/// Runs `f` over `items` on a few threads, keeping the order.
fn parallel_map<T: Sync, R: Send>(items: &[T], f: impl Fn(&T) -> R + Sync) -> Vec<R> {
    let threads = decode_threads().min(items.len());
    if threads <= 1 {
        return items.iter().map(f).collect();
    }
    let next = std::sync::atomic::AtomicUsize::new(0);
    let mut results: Vec<(usize, R)> = std::thread::scope(|scope| {
        let workers: Vec<_> = (0..threads)
            .map(|_| {
                scope.spawn(|| {
                    let mut done = Vec::new();
                    loop {
                        let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        let Some(item) = items.get(i) else { break };
                        done.push((i, f(item)));
                    }
                    done
                })
            })
            .collect();
        workers
            .into_iter()
            .flat_map(|w| w.join().expect("tile decoding thread panicked"))
            .collect()
    });
    results.sort_by_key(|(i, _)| *i);
    results.into_iter().map(|(_, r)| r).collect()
}

/// `SourceError` isn't `Clone`; one failed batch read fails every tile in it.
fn copy_source_error(e: &SourceError) -> SourceError {
    match e {
        SourceError::Io(io) => SourceError::Io(std::io::Error::new(io.kind(), io.to_string())),
        SourceError::Http(m) => SourceError::Http(m.clone()),
        SourceError::ObjectStore(m) => SourceError::ObjectStore(m.clone()),
        SourceError::Unsupported(m) => SourceError::Unsupported(m.clone()),
        SourceError::ShortRead {
            offset,
            expected,
            received,
        } => SourceError::ShortRead {
            offset: *offset,
            expected: *expected,
            received: *received,
        },
    }
}

fn copy_cog_error(e: &CogError) -> CogError {
    match e {
        CogError::Source(s) => CogError::Source(copy_source_error(s)),
        CogError::NotTiff(m) => CogError::NotTiff(m.clone()),
        CogError::Invalid(m) => CogError::Invalid(m.clone()),
        CogError::Unsupported(m) => CogError::Unsupported(m.clone()),
        CogError::Decode { key, message } => CogError::Decode {
            key: *key,
            message: message.clone(),
        },
    }
}

fn format_bytes(bytes: u64) -> String {
    const UNITS: [&str; 5] = ["bytes", "KiB", "MiB", "GiB", "TiB"];
    let mut value = bytes as f64;
    let mut unit = 0;
    while value >= 1024.0 && unit < UNITS.len() - 1 {
        value /= 1024.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{bytes} bytes")
    } else {
        format!("{value:.1} {} ({bytes} bytes)", UNITS[unit])
    }
}

/// An overview factor: "2" for exact ones, "2.003" otherwise.
fn format_scale(scale: f64) -> String {
    if (scale - scale.round()).abs() < 1e-9 {
        format!("{}", scale.round())
    } else {
        format!("{scale:.3}")
    }
}

fn data_type_name(data_type: DataType) -> &'static str {
    match data_type {
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

fn photometric_name(photometric: u16) -> &'static str {
    match photometric {
        0 => "min-is-white",
        1 => "min-is-black",
        2 => "RGB",
        3 => "palette",
        4 => "transparency mask",
        5 => "CMYK",
        6 => "YCbCr",
        8 => "CIELab",
        _ => "other",
    }
}

/// A human-readable summary of a COG's structure, for `cogcat info`.
pub fn describe(cog: &Cog) -> String {
    use std::fmt::Write;
    let info = cog.info();
    let mut s = String::new();
    let mut line = |label: &str, value: String| {
        let _ = writeln!(s, "{label:<14}{value}");
    };
    let format = match (cog.big_tiff, cog.order) {
        (true, ByteOrder::Little) => "BigTIFF, little-endian",
        (true, ByteOrder::Big) => "BigTIFF, big-endian",
        (false, ByteOrder::Little) => "TIFF, little-endian",
        (false, ByteOrder::Big) => "TIFF, big-endian",
    };
    line(
        "File",
        match info.file_size {
            Some(size) => format!("{}, {format}", format_bytes(size)),
            None => format!("size unknown, {format}"),
        },
    );
    line("Size", format!("{} × {} pixels", info.width, info.height));
    let alpha = info
        .alpha_band
        .map(|b| format!(", band {} is alpha", b + 1))
        .unwrap_or_default();
    line(
        "Bands",
        format!("{} × {}{alpha}", info.bands, data_type_name(info.data_type)),
    );
    line("Compression", compression_name(info.compression));
    line(
        "Interleave",
        match info.planar {
            Planar::Chunky => "pixel (chunky)".into(),
            Planar::Separate => "band (planar separate)".into(),
        },
    );
    line(
        "Predictor",
        match cog.predictor {
            1 => "none".into(),
            2 => "horizontal (2)".into(),
            3 => "floating point (3)".into(),
            p => p.to_string(),
        },
    );
    line(
        "Photometric",
        format!(
            "{} ({})",
            photometric_name(info.photometric),
            info.photometric
        ),
    );
    line(
        "Nodata",
        info.nodata.map_or("none".into(), |v| v.to_string()),
    );
    line(
        "Mask",
        if info.has_mask {
            "internal mask"
        } else {
            "none"
        }
        .into(),
    );
    if cog.stripped {
        line("Layout", "strips, grouped into tiles (not a COG)".into());
    }
    let units = info.crs.units.as_deref().unwrap_or("CRS units");
    line(
        "CRS",
        match &info.crs.units {
            Some(u) => format!("{}, {u}", info.crs.label()),
            None => info.crs.label(),
        },
    );
    match &info.geo {
        Some(gt) => {
            let c = gt.0;
            line(
                "Geotransform",
                format!(
                    "[{}, {}, {}, {}, {}, {}]",
                    c[0], c[1], c[2], c[3], c[4], c[5]
                ),
            );
            let size_x = (c[1] * c[1] + c[4] * c[4]).sqrt();
            let size_y = (c[2] * c[2] + c[5] * c[5]).sqrt();
            let rotated = if c[2] != 0.0 || c[4] != 0.0 {
                " (rotated)"
            } else {
                ""
            };
            line(
                "Pixel size",
                format!("{size_x} × {size_y} {units}{rotated}"),
            );
            let (w, h) = (info.width as f64, info.height as f64);
            let corners = [
                gt.pixel_to_crs(0.0, 0.0),
                gt.pixel_to_crs(w, 0.0),
                gt.pixel_to_crs(0.0, h),
                gt.pixel_to_crs(w, h),
            ];
            let min_x = corners.iter().map(|p| p[0]).fold(f64::INFINITY, f64::min);
            let max_x = corners
                .iter()
                .map(|p| p[0])
                .fold(f64::NEG_INFINITY, f64::max);
            let min_y = corners.iter().map(|p| p[1]).fold(f64::INFINITY, f64::min);
            let max_y = corners
                .iter()
                .map(|p| p[1])
                .fold(f64::NEG_INFINITY, f64::max);
            line(
                "Bounds",
                format!("x {min_x} to {max_x}, y {min_y} to {max_y}"),
            );
        }
        None => line("Geotransform", "none (pixel coordinates)".into()),
    }
    let _ = writeln!(s, "Levels:");
    for (level, images) in cog.levels().iter().zip(&cog.images) {
        let _ = writeln!(
            s,
            "  {:>2}: {:>6} × {:<6} tiles {} × {}, grid {} × {} = {} tiles{}{}",
            level.index,
            level.width,
            level.height,
            level.tile_width,
            level.tile_height,
            level.tiles_across,
            level.tiles_down,
            level.tiles_across as u64 * level.tiles_down as u64,
            if level.index == 0 {
                String::new()
            } else {
                format!(", 1/{}", format_scale(level.scale[0]))
            },
            if images.mask.is_some() { ", mask" } else { "" },
        );
    }
    let has_band_info = info.band_info.iter().any(|b| *b != BandInfo::default());
    if has_band_info {
        let _ = writeln!(s, "Bands:");
        for (i, band) in info.band_info.iter().enumerate() {
            let mut parts = Vec::new();
            if let Some(d) = &band.description {
                parts.push(d.clone());
            }
            match (band.wavelength_nm, band.fwhm_nm) {
                (Some(w), Some(f)) => parts.push(format!("{w} nm (FWHM {f} nm)")),
                (Some(w), None) => parts.push(format!("{w} nm")),
                (None, Some(f)) => parts.push(format!("FWHM {f} nm")),
                (None, None) => {}
            }
            if let Some(scale) = band.scale {
                parts.push(format!("scale {scale}"));
            }
            if let Some(offset) = band.offset {
                parts.push(format!("offset {offset}"));
            }
            if info.alpha_band == Some(i as u32) {
                parts.push("alpha".into());
            }
            let _ = writeln!(s, "  {:>3}: {}", i + 1, parts.join(", "));
        }
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn merge_ranges_should_join_close_ranges_and_keep_far_ones() {
        let merged = merge_ranges(vec![(1000, 10), (0, 100), (200, 50), (10_000_000, 5)]);
        assert_eq!(merged, [(0, 1010), (10_000_000, 5)]);
    }

    #[test]
    fn merge_ranges_should_cap_merged_size() {
        let mib = 1024 * 1024;
        let merged = merge_ranges(vec![(0, 3 * mib), (3 * mib, 2 * mib)]);
        assert_eq!(merged, [(0, 3 * mib), (3 * mib, 2 * mib)]);
    }

    #[test]
    fn merge_ranges_should_absorb_contained_and_duplicate_ranges() {
        let merged = merge_ranges(vec![(0, 100), (10, 20), (0, 100), (50, 100)]);
        assert_eq!(merged, [(0, 150)]);
    }

    #[test]
    fn slice_fetched_should_find_a_chunk_in_its_merged_read() {
        let merged = [(0, 10), (100, 5)];
        let fetched = vec![(0..10).collect::<Vec<u8>>(), vec![100, 101, 102, 103, 104]];
        assert_eq!(slice_fetched(&merged, &fetched, 3, 4), [3, 4, 5, 6]);
        assert_eq!(slice_fetched(&merged, &fetched, 101, 2), [101, 102]);
    }

    #[test]
    fn parallel_map_should_keep_order() {
        let items: Vec<u32> = (0..100).collect();
        assert_eq!(
            parallel_map(&items, |v| v * 2),
            items.iter().map(|v| v * 2).collect::<Vec<_>>()
        );
    }
}
