//! Turning a tile's (or strip's) bytes into samples: decompression,
//! predictors, byte order, and placing the samples into a band-sequential
//! `f32` tile with nodata as NaN.

use std::io::Read;
use std::sync::Arc;

use super::tiff::ByteOrder;
use super::{Compression, DataType};

/// How one chunk (a tile or a strip, holding every band or one band plane)
/// is stored.
#[derive(Clone, Debug)]
pub struct ChunkLayout {
    pub compression: Compression,
    /// 1 none, 2 horizontal differencing, 3 floating point.
    pub predictor: u16,
    pub order: ByteOrder,
    /// Bits per sample: 8, 16, 32 or 64, or 1 for masks.
    pub bits: u16,
    /// Ignored for 1-bit masks.
    pub data_type: DataType,
    /// Samples per pixel in this chunk: every band (chunky) or 1 (separate).
    pub samples: u32,
    /// `PhotometricInterpretation`, for JPEG's colour conversion.
    pub photometric: u16,
    /// `JPEGTables`: quantisation and Huffman tables shared by all tiles.
    pub jpeg_tables: Option<Arc<[u8]>>,
}

/// Decoded samples, pixel-interleaved within the chunk.
#[derive(Clone, Debug, PartialEq)]
pub enum Samples {
    U8(Vec<u8>),
    I8(Vec<i8>),
    U16(Vec<u16>),
    I16(Vec<i16>),
    U32(Vec<u32>),
    I32(Vec<i32>),
    F32(Vec<f32>),
    F64(Vec<f64>),
}

/// A decoded chunk.
#[derive(Clone, Debug)]
pub struct Chunk {
    pub samples: Samples,
    pub width: u32,
    pub rows: u32,
    /// Samples per pixel.
    pub channels: u32,
}

/// Decodes one chunk of `width` × `rows` pixels.
pub fn decode_chunk(
    layout: &ChunkLayout,
    bytes: &[u8],
    width: u32,
    rows: u32,
) -> Result<Chunk, String> {
    match layout.compression {
        Compression::Jpeg => return decode_jpeg(layout, bytes),
        Compression::Webp => return decode_webp(bytes),
        _ => {}
    }
    let bytes_per_sample = (layout.bits / 8) as usize;
    let expected = width as usize * rows as usize * layout.samples as usize * bytes_per_sample;
    let mut raw = decompress(layout.compression, bytes, expected)?;
    // Some writers store short final strips or pad tiles; be lenient.
    raw.resize(expected, 0);
    let row_samples = width as usize * layout.samples as usize;
    let samples = to_samples(layout, raw, row_samples)?;
    Ok(Chunk {
        samples,
        width,
        rows,
        channels: layout.samples,
    })
}

/// Decodes a mask chunk to one byte per pixel, 0 where masked out.
pub fn decode_mask(
    layout: &ChunkLayout,
    bytes: &[u8],
    width: u32,
    rows: u32,
) -> Result<Vec<u8>, String> {
    let (width, rows) = (width as usize, rows as usize);
    match layout.bits {
        1 => {
            let row_bytes = width.div_ceil(8);
            let mut raw = decompress(layout.compression, bytes, row_bytes * rows)?;
            raw.resize(row_bytes * rows, 0);
            Ok(unpack_bits(&raw, width, rows))
        }
        8 => {
            let mut raw = decompress(layout.compression, bytes, width * rows)?;
            raw.resize(width * rows, 0);
            Ok(raw)
        }
        bits => Err(format!("{bits}-bit masks are not supported")),
    }
}

/// Unpacks 1-bit rows (most significant bit first, rows padded to whole
/// bytes) into one byte per pixel.
pub fn unpack_bits(packed: &[u8], width: usize, rows: usize) -> Vec<u8> {
    let row_bytes = width.div_ceil(8);
    let mut out = Vec::with_capacity(width * rows);
    for row in packed.chunks(row_bytes).take(rows) {
        out.extend((0..width).map(|x| (row[x / 8] >> (7 - x % 8)) & 1));
    }
    out
}

/// Undoes a lossless compression. `expected` is the decompressed size, used
/// to size buffers.
pub fn decompress(
    compression: Compression,
    input: &[u8],
    expected: usize,
) -> Result<Vec<u8>, String> {
    match compression {
        Compression::None => Ok(input.to_vec()),
        Compression::Deflate => inflate(input, expected),
        Compression::Lzw => lzw(input, expected),
        Compression::PackBits => Ok(unpack_packbits(input, expected)),
        Compression::Zstd => zstd(input, expected),
        other => Err(format!("{other:?} is not a lossless byte codec")),
    }
}

fn inflate(input: &[u8], expected: usize) -> Result<Vec<u8>, String> {
    let mut out = Vec::with_capacity(expected);
    let result = flate2::read::ZlibDecoder::new(input).read_to_end(&mut out);
    match result {
        Ok(_) => Ok(out),
        // A truncated or padded stream that still gave everything is fine.
        Err(_) if out.len() >= expected => Ok(out),
        Err(e) => Err(format!("Deflate: {e}")),
    }
}

fn lzw(input: &[u8], expected: usize) -> Result<Vec<u8>, String> {
    let mut decoder = weezl::decode::Decoder::with_tiff_size_switch(weezl::BitOrder::Msb, 8);
    let mut out = Vec::with_capacity(expected);
    let result = decoder.into_vec(&mut out).decode(input);
    match result.status {
        // Streams without an end code stop with `NoProgress`; that's fine.
        Ok(_) => Ok(out),
        Err(_) if out.len() >= expected => Ok(out),
        Err(e) => Err(format!("LZW: {e}")),
    }
}

fn zstd(input: &[u8], expected: usize) -> Result<Vec<u8>, String> {
    let mut out = vec![0; expected];
    let mut decoder = ruzstd::decoding::FrameDecoder::new();
    match decoder.decode_all(input, &mut out) {
        Ok(written) => {
            out.truncate(written);
            Ok(out)
        }
        Err(e) => Err(format!("ZSTD: {e}")),
    }
}

/// PackBits run-length decoding, stopping at `expected` bytes.
pub fn unpack_packbits(input: &[u8], expected: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(expected);
    let mut i = 0;
    while i < input.len() && out.len() < expected {
        let n = input[i] as i8;
        i += 1;
        if n >= 0 {
            let count = n as usize + 1;
            let end = (i + count).min(input.len());
            out.extend_from_slice(&input[i..end]);
            i = end;
        } else if n != -128 {
            let count = 1 - n as isize;
            if let Some(&byte) = input.get(i) {
                out.extend(std::iter::repeat_n(byte, count as usize));
            }
            i += 1;
        }
    }
    out.truncate(expected);
    out
}

fn decode_jpeg(layout: &ChunkLayout, bytes: &[u8]) -> Result<Chunk, String> {
    // JPEG-in-TIFF tiles often leave out the tables, which live in
    // `JPEGTables` (a stream of SOI, tables, EOI): splice them in.
    let spliced;
    let stream = match &layout.jpeg_tables {
        Some(tables) if tables.len() > 4 && bytes.len() > 2 => {
            let mut s = Vec::with_capacity(tables.len() + bytes.len());
            s.extend_from_slice(&tables[..tables.len() - 2]);
            s.extend_from_slice(&bytes[2..]);
            spliced = s;
            &spliced[..]
        }
        _ => bytes,
    };
    let mut decoder = jpeg_decoder::Decoder::new(stream);
    let transform = match (layout.photometric, layout.samples) {
        (6, _) => Some(jpeg_decoder::ColorTransform::YCbCr),
        (2, 3) => Some(jpeg_decoder::ColorTransform::RGB),
        (_, 1) => Some(jpeg_decoder::ColorTransform::Grayscale),
        _ => None,
    };
    if let Some(transform) = transform {
        decoder.set_color_transform(transform);
    }
    let pixels = decoder.decode().map_err(|e| format!("JPEG: {e}"))?;
    let info = decoder.info().ok_or("JPEG: no image")?;
    let channels = match info.pixel_format {
        jpeg_decoder::PixelFormat::L8 => 1,
        jpeg_decoder::PixelFormat::RGB24 => 3,
        jpeg_decoder::PixelFormat::CMYK32 => 4,
        jpeg_decoder::PixelFormat::L16 => return Err("12/16-bit JPEG is not supported".into()),
    };
    Ok(Chunk {
        samples: Samples::U8(pixels),
        width: info.width as u32,
        rows: info.height as u32,
        channels,
    })
}

fn decode_webp(bytes: &[u8]) -> Result<Chunk, String> {
    let mut decoder = image_webp::WebPDecoder::new(std::io::Cursor::new(bytes))
        .map_err(|e| format!("WebP: {e}"))?;
    let (width, rows) = decoder.dimensions();
    let channels = if decoder.has_alpha() { 4 } else { 3 };
    let mut pixels = vec![0; width as usize * rows as usize * channels as usize];
    decoder
        .read_image(&mut pixels)
        .map_err(|e| format!("WebP: {e}"))?;
    Ok(Chunk {
        samples: Samples::U8(pixels),
        width,
        rows,
        channels,
    })
}

/// Integers that horizontal differencing works on.
trait Word: Copy {
    fn wrapping_add(self, other: Self) -> Self;
}

macro_rules! word {
    ($($t:ty),*) => {$(
        impl Word for $t {
            fn wrapping_add(self, other: Self) -> Self {
                <$t>::wrapping_add(self, other)
            }
        }
    )*};
}
word!(u8, u16, u32, u64);

/// Undoes horizontal differencing (predictor 2): each sample was stored as
/// the difference from the same band's sample in the pixel to its left.
fn undo_horizontal<T: Word>(words: &mut [T], row_samples: usize, samples: usize) {
    if row_samples == 0 {
        return;
    }
    for row in words.chunks_mut(row_samples) {
        for i in samples..row.len() {
            row[i] = row[i].wrapping_add(row[i - samples]);
        }
    }
}

/// Undoes the floating-point predictor (3) for one row of `row_samples`
/// samples of `size` bytes: byte-wise differencing, then the bytes were
/// split into planes, most significant first. Returns each sample's bytes in
/// big-endian order.
pub fn undo_float_predictor(row: &mut [u8], samples: usize, size: usize) -> Vec<u8> {
    for i in samples..row.len() {
        row[i] = row[i].wrapping_add(row[i - samples]);
    }
    let count = row.len() / size;
    let mut out = vec![0; count * size];
    for (k, value) in out.chunks_exact_mut(size).enumerate() {
        for (byte, v) in value.iter_mut().enumerate() {
            *v = row[byte * count + k];
        }
    }
    out
}

/// Turns raw decompressed bytes (file byte order) into typed samples,
/// undoing any predictor.
fn to_samples(
    layout: &ChunkLayout,
    mut raw: Vec<u8>,
    row_samples: usize,
) -> Result<Samples, String> {
    let samples = layout.samples as usize;
    let order = layout.order;
    if layout.predictor == 3 {
        let size = (layout.bits / 8) as usize;
        if !matches!(layout.data_type, DataType::F32 | DataType::F64) {
            return Err("floating-point predictor on integer samples".into());
        }
        let mut be = Vec::with_capacity(raw.len());
        for row in raw.chunks_mut(row_samples * size) {
            be.extend(undo_float_predictor(row, samples, size));
        }
        return Ok(match layout.data_type {
            DataType::F32 => Samples::F32(
                be.chunks_exact(4)
                    .map(|c| f32::from_be_bytes([c[0], c[1], c[2], c[3]]))
                    .collect(),
            ),
            _ => Samples::F64(
                be.chunks_exact(8)
                    .map(|c| f64::from_be_bytes(c.try_into().expect("8 bytes")))
                    .collect(),
            ),
        });
    }
    let horizontal = layout.predictor == 2;
    Ok(match layout.bits {
        8 => {
            if horizontal {
                undo_horizontal(&mut raw, row_samples, samples);
            }
            match layout.data_type {
                DataType::I8 => Samples::I8(raw.into_iter().map(|v| v as i8).collect()),
                _ => Samples::U8(raw),
            }
        }
        16 => {
            let mut words: Vec<u16> = match order {
                ByteOrder::Little => raw
                    .chunks_exact(2)
                    .map(|c| u16::from_le_bytes([c[0], c[1]]))
                    .collect(),
                ByteOrder::Big => raw
                    .chunks_exact(2)
                    .map(|c| u16::from_be_bytes([c[0], c[1]]))
                    .collect(),
            };
            if horizontal {
                undo_horizontal(&mut words, row_samples, samples);
            }
            match layout.data_type {
                DataType::I16 => Samples::I16(words.into_iter().map(|v| v as i16).collect()),
                _ => Samples::U16(words),
            }
        }
        32 => {
            let mut words: Vec<u32> = raw.chunks_exact(4).map(|c| order.u32(c)).collect();
            if horizontal {
                undo_horizontal(&mut words, row_samples, samples);
            }
            match layout.data_type {
                DataType::I32 => Samples::I32(words.into_iter().map(|v| v as i32).collect()),
                DataType::F32 => Samples::F32(words.into_iter().map(f32::from_bits).collect()),
                _ => Samples::U32(words),
            }
        }
        64 => {
            let mut words: Vec<u64> = raw.chunks_exact(8).map(|c| order.u64(c)).collect();
            if horizontal {
                undo_horizontal(&mut words, row_samples, samples);
            }
            Samples::F64(words.into_iter().map(f64::from_bits).collect())
        }
        bits => return Err(format!("{bits}-bit samples are not supported")),
    })
}

/// Sample types that convert to f32 for display.
trait Sample: Copy {
    fn to_f32(self) -> f32;
    fn to_f64(self) -> f64;
    /// The nodata value as this type would see it: f32 samples compare
    /// with the nodata value rounded to f32, as GDAL does.
    fn nodata_key(nodata: f64) -> f64 {
        nodata
    }
}

macro_rules! sample {
    ($($t:ty),*) => {$(
        impl Sample for $t {
            fn to_f32(self) -> f32 {
                self as f32
            }
            fn to_f64(self) -> f64 {
                self as f64
            }
        }
    )*};
}
sample!(u8, i8, u16, i16, u32, i32, f64);

impl Sample for f32 {
    fn to_f32(self) -> f32 {
        self
    }
    fn to_f64(self) -> f64 {
        self as f64
    }
    fn nodata_key(nodata: f64) -> f64 {
        nodata as f32 as f64
    }
}

/// Where a decoded chunk goes in the output tile.
#[derive(Clone, Copy, Debug)]
pub struct Placement {
    pub tile_width: u32,
    pub tile_height: u32,
    /// Bands in the output tile.
    pub bands: u32,
    /// The part of the tile inside the image; the rest stays NaN.
    pub valid: [u32; 2],
    /// The chunk's first row within the tile (strips stack up in a tile).
    pub row0: u32,
    /// The output band of the chunk's first channel.
    pub first_band: u32,
    pub nodata: Option<f64>,
}

/// Copies a decoded chunk into the band-sequential output tile, turning
/// nodata samples into NaN. Channels beyond the tile's bands are dropped.
pub fn place(chunk: &Chunk, out: &mut [f32], p: &Placement) {
    match &chunk.samples {
        Samples::U8(s) => place_typed(s, chunk, out, p),
        Samples::I8(s) => place_typed(s, chunk, out, p),
        Samples::U16(s) => place_typed(s, chunk, out, p),
        Samples::I16(s) => place_typed(s, chunk, out, p),
        Samples::U32(s) => place_typed(s, chunk, out, p),
        Samples::I32(s) => place_typed(s, chunk, out, p),
        Samples::F32(s) => place_typed(s, chunk, out, p),
        Samples::F64(s) => place_typed(s, chunk, out, p),
    }
}

fn place_typed<T: Sample>(samples: &[T], chunk: &Chunk, out: &mut [f32], p: &Placement) {
    let channels = chunk.channels as usize;
    let bands = (p.bands.saturating_sub(p.first_band) as usize).min(channels);
    let cols = chunk.width.min(p.valid[0]) as usize;
    let rows = chunk.rows.min(p.valid[1].saturating_sub(p.row0)) as usize;
    let (tw, th) = (p.tile_width as usize, p.tile_height as usize);
    let nodata = p.nodata.map(T::nodata_key);
    for c in 0..bands {
        let band_out = &mut out[(p.first_band as usize + c) * tw * th..][..tw * th];
        for y in 0..rows {
            let src_row = y * chunk.width as usize * channels;
            let dst = &mut band_out[(p.row0 as usize + y) * tw..][..cols];
            let src = &samples[src_row..];
            for (x, d) in dst.iter_mut().enumerate() {
                let v = src[x * channels + c];
                *d = match nodata {
                    Some(nd) if v.to_f64() == nd => f32::NAN,
                    _ => v.to_f32(),
                };
            }
        }
    }
}

/// Sets every band to NaN where a decoded mask chunk is 0.
pub fn apply_mask(mask: &[u8], mask_width: u32, mask_rows: u32, out: &mut [f32], p: &Placement) {
    let cols = mask_width.min(p.valid[0]) as usize;
    let rows = mask_rows.min(p.valid[1].saturating_sub(p.row0)) as usize;
    let (tw, th) = (p.tile_width as usize, p.tile_height as usize);
    for y in 0..rows {
        let mask_row = &mask[y * mask_width as usize..][..cols];
        for (x, &m) in mask_row.iter().enumerate() {
            if m == 0 {
                let at = (p.row0 as usize + y) * tw + x;
                for b in 0..p.bands as usize {
                    out[b * tw * th + at] = f32::NAN;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn layout(
        data_type: DataType,
        bits: u16,
        samples: u32,
        predictor: u16,
        order: ByteOrder,
    ) -> ChunkLayout {
        ChunkLayout {
            compression: Compression::None,
            predictor,
            order,
            bits,
            data_type,
            samples,
            photometric: 1,
            jpeg_tables: None,
        }
    }

    #[test]
    fn unpack_packbits_should_expand_runs_and_literals() {
        // The example from the TIFF 6 specification.
        let packed = [
            0xFE, 0xAA, 0x02, 0x80, 0x00, 0x2A, 0xFD, 0xAA, 0x03, 0x80, 0x00, 0x2A, 0x22, 0xF7,
            0xAA,
        ];
        let expected = [
            0xAA, 0xAA, 0xAA, 0x80, 0x00, 0x2A, 0xAA, 0xAA, 0xAA, 0xAA, 0x80, 0x00, 0x2A, 0x22,
            0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA,
        ];
        assert_eq!(unpack_packbits(&packed, 24), expected);
        assert_eq!(unpack_packbits(&packed, 5), &expected[..5]);
        // A no-op byte (-128) is skipped.
        assert_eq!(unpack_packbits(&[0x80, 0x00, 7], 1), [7]);
    }

    #[test]
    fn horizontal_predictor_should_restore_u16_chunky_rows() {
        // Two rows of 3 pixels × 2 samples, differenced per band.
        let values: [u16; 12] = [10, 100, 12, 90, 15, 95, 1, 65535, 3, 0, 2, 5];
        let mut diffs = values;
        for row in diffs.chunks_mut(6) {
            for i in (2..6).rev() {
                row[i] = row[i].wrapping_sub(row[i - 2]);
            }
        }
        for order in [ByteOrder::Little, ByteOrder::Big] {
            let raw: Vec<u8> = diffs
                .iter()
                .flat_map(|v| match order {
                    ByteOrder::Little => v.to_le_bytes(),
                    ByteOrder::Big => v.to_be_bytes(),
                })
                .collect();
            let l = layout(DataType::U16, 16, 2, 2, order);
            assert_eq!(
                to_samples(&l, raw, 6).unwrap(),
                Samples::U16(values.to_vec())
            );
        }
    }

    #[test]
    fn horizontal_predictor_should_wrap_signed_bytes() {
        let values: [i8; 4] = [-100, 100, -128, 127];
        let mut diffs: Vec<u8> = values.iter().map(|&v| v as u8).collect();
        for i in (1..4).rev() {
            diffs[i] = diffs[i].wrapping_sub(diffs[i - 1]);
        }
        let l = layout(DataType::I8, 8, 1, 2, ByteOrder::Little);
        assert_eq!(
            to_samples(&l, diffs, 4).unwrap(),
            Samples::I8(values.to_vec())
        );
    }

    /// Applies the floating-point predictor the way libtiff encodes it.
    fn encode_float_predictor(values: &[f32], samples: usize) -> Vec<u8> {
        let count = values.len();
        let mut planes = vec![0u8; count * 4];
        for (k, v) in values.iter().enumerate() {
            for (byte, b) in v.to_be_bytes().iter().enumerate() {
                planes[byte * count + k] = *b;
            }
        }
        for i in (samples..planes.len()).rev() {
            planes[i] = planes[i].wrapping_sub(planes[i - samples]);
        }
        planes
    }

    #[test]
    fn float_predictor_should_restore_f32_rows_in_either_byte_order() {
        let values = [1.5f32, -2.25, f32::NAN, 1e30, 0.0, -0.0];
        let mut raw = encode_float_predictor(&values[..3], 1);
        raw.extend(encode_float_predictor(&values[3..], 1));
        for order in [ByteOrder::Little, ByteOrder::Big] {
            let l = layout(DataType::F32, 32, 1, 3, order);
            let Samples::F32(decoded) = to_samples(&l, raw.clone(), 3).unwrap() else {
                panic!("expected f32");
            };
            let bits: Vec<u32> = decoded.iter().map(|v| v.to_bits()).collect();
            let expected: Vec<u32> = values.iter().map(|v| v.to_bits()).collect();
            assert_eq!(bits, expected);
        }
    }

    #[test]
    fn unpack_bits_should_read_msb_first_padded_rows() {
        let packed = [0b1010_0000, 0b0111_1111];
        assert_eq!(unpack_bits(&packed, 3, 2), [1, 0, 1, 0, 1, 1]);
    }

    #[test]
    fn place_should_convert_nodata_and_leave_outside_valid_as_nan() {
        let chunk = Chunk {
            samples: Samples::I16(vec![1, -5, 3, 4, 5, 6]),
            width: 3,
            rows: 2,
            channels: 1,
        };
        let mut out = vec![f32::NAN; 3 * 2 * 2];
        let p = Placement {
            tile_width: 3,
            tile_height: 2,
            bands: 2,
            valid: [2, 2],
            row0: 0,
            first_band: 1,
            nodata: Some(-5.0),
        };
        place(&chunk, &mut out, &p);
        let band1 = &out[6..];
        assert_eq!(band1[0], 1.0);
        assert!(band1[1].is_nan(), "nodata");
        assert!(band1[2].is_nan(), "outside the valid width");
        assert_eq!(&band1[3..5], [4.0, 5.0]);
        assert!(out[..6].iter().all(|v| v.is_nan()), "band 0 untouched");
    }

    #[test]
    fn place_should_compare_f32_nodata_after_rounding() {
        let nodata = -3.4028234663852886e38;
        let chunk = Chunk {
            samples: Samples::F32(vec![f32::MIN, 2.0]),
            width: 2,
            rows: 1,
            channels: 1,
        };
        let mut out = vec![0.0; 2];
        let p = Placement {
            tile_width: 2,
            tile_height: 1,
            bands: 1,
            valid: [2, 1],
            row0: 0,
            first_band: 0,
            nodata: Some(nodata),
        };
        place(&chunk, &mut out, &p);
        assert!(out[0].is_nan());
        assert_eq!(out[1], 2.0);
    }

    #[test]
    fn lossless_codecs_should_round_trip() {
        use std::io::Write;
        let data: Vec<u8> = (0..5000u32).map(|i| (i * 7 % 251) as u8).collect();
        let mut z = flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::default());
        z.write_all(&data).unwrap();
        let deflated = z.finish().unwrap();
        assert_eq!(
            decompress(Compression::Deflate, &deflated, data.len()).unwrap(),
            data
        );

        let lzw = weezl::encode::Encoder::with_tiff_size_switch(weezl::BitOrder::Msb, 8)
            .encode(&data)
            .unwrap();
        assert_eq!(
            decompress(Compression::Lzw, &lzw, data.len()).unwrap(),
            data
        );
    }
}
