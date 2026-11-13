//! TIFF and BigTIFF structure: the header, the chain of IFDs (image file
//! directories) and their tags' values.
//!
//! [`read_ifds`] reads a generous prefix of the file first, because GDAL's COGs
//! keep every IFD and most tag data at the start, then fetches whatever lies
//! outside it: IFDs further on in one read each, and all out-of-prefix tag
//! values (usually big `TileOffsets` arrays) together in one batch.

use std::collections::{BTreeMap, HashSet};

use super::CogError;
use crate::source::ByteSource;

// Baseline and extension tags.
pub const NEW_SUBFILE_TYPE: u16 = 254;
pub const IMAGE_WIDTH: u16 = 256;
pub const IMAGE_LENGTH: u16 = 257;
pub const BITS_PER_SAMPLE: u16 = 258;
pub const COMPRESSION: u16 = 259;
pub const PHOTOMETRIC: u16 = 262;
pub const STRIP_OFFSETS: u16 = 273;
pub const SAMPLES_PER_PIXEL: u16 = 277;
pub const ROWS_PER_STRIP: u16 = 278;
pub const STRIP_BYTE_COUNTS: u16 = 279;
pub const PLANAR_CONFIGURATION: u16 = 284;
pub const PREDICTOR: u16 = 317;
pub const TILE_WIDTH: u16 = 322;
pub const TILE_LENGTH: u16 = 323;
pub const TILE_OFFSETS: u16 = 324;
pub const TILE_BYTE_COUNTS: u16 = 325;
pub const EXTRA_SAMPLES: u16 = 338;
pub const SAMPLE_FORMAT: u16 = 339;
pub const JPEG_TABLES: u16 = 347;
// GeoTIFF.
pub const MODEL_PIXEL_SCALE: u16 = 33550;
pub const MODEL_TIEPOINT: u16 = 33922;
pub const MODEL_TRANSFORMATION: u16 = 34264;
pub const GEO_KEY_DIRECTORY: u16 = 34735;
pub const GEO_DOUBLE_PARAMS: u16 = 34736;
pub const GEO_ASCII_PARAMS: u16 = 34737;
// GDAL.
pub const GDAL_METADATA: u16 = 42112;
pub const GDAL_NODATA: u16 = 42113;

/// How much of the file to read up front.
pub const PREFIX_SIZE: u64 = 64 * 1024;

/// Guards against IFD loops and absurd files.
const MAX_IFDS: usize = 4096;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ByteOrder {
    Little,
    Big,
}

impl ByteOrder {
    pub fn u16(self, b: &[u8]) -> u16 {
        let b = [b[0], b[1]];
        match self {
            ByteOrder::Little => u16::from_le_bytes(b),
            ByteOrder::Big => u16::from_be_bytes(b),
        }
    }

    pub fn u32(self, b: &[u8]) -> u32 {
        let b = [b[0], b[1], b[2], b[3]];
        match self {
            ByteOrder::Little => u32::from_le_bytes(b),
            ByteOrder::Big => u32::from_be_bytes(b),
        }
    }

    pub fn u64(self, b: &[u8]) -> u64 {
        let b: [u8; 8] = b[..8].try_into().expect("8 bytes");
        match self {
            ByteOrder::Little => u64::from_le_bytes(b),
            ByteOrder::Big => u64::from_be_bytes(b),
        }
    }
}

/// The 8- or 16-byte file header.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Header {
    pub order: ByteOrder,
    pub big_tiff: bool,
    pub first_ifd: u64,
}

impl Header {
    /// Bytes of an IFD's entry count, each entry, and the next-IFD offset.
    fn sizes(&self) -> (u64, u64, u64) {
        if self.big_tiff {
            (8, 20, 8)
        } else {
            (2, 12, 4)
        }
    }
}

pub fn parse_header(bytes: &[u8]) -> Result<Header, CogError> {
    if bytes.len() < 8 {
        return Err(CogError::NotTiff(format!(
            "the file is only {} bytes long",
            bytes.len()
        )));
    }
    let order = match &bytes[..2] {
        b"II" => ByteOrder::Little,
        b"MM" => ByteOrder::Big,
        _ => {
            return Err(CogError::NotTiff(format!(
                "it starts with {:02x?} instead of \"II\" or \"MM\"",
                &bytes[..4]
            )));
        }
    };
    match order.u16(&bytes[2..]) {
        42 => Ok(Header {
            order,
            big_tiff: false,
            first_ifd: order.u32(&bytes[4..]) as u64,
        }),
        43 => {
            if bytes.len() < 16 {
                return Err(CogError::NotTiff("BigTIFF header is cut short".into()));
            }
            let offset_size = order.u16(&bytes[4..]);
            if offset_size != 8 {
                return Err(CogError::Invalid(format!(
                    "BigTIFF offset size {offset_size}, expected 8"
                )));
            }
            Ok(Header {
                order,
                big_tiff: true,
                first_ifd: order.u64(&bytes[8..]),
            })
        }
        version => Err(CogError::NotTiff(format!(
            "version {version}, expected 42 (TIFF) or 43 (BigTIFF)"
        ))),
    }
}

/// Size in bytes of one value of a TIFF field type, or `None` for an
/// unknown type.
pub fn type_size(field_type: u16) -> Option<u64> {
    Some(match field_type {
        1 | 2 | 6 | 7 => 1,   // BYTE, ASCII, SBYTE, UNDEFINED
        3 | 8 => 2,           // SHORT, SSHORT
        4 | 9 | 11 | 13 => 4, // LONG, SLONG, FLOAT, IFD
        5 | 10 | 12 => 8,     // RATIONAL, SRATIONAL, DOUBLE
        16..=18 => 8,         // LONG8, SLONG8, IFD8
        _ => return None,
    })
}

/// One tag's value: its field type, count and raw bytes in file byte order.
#[derive(Clone, Debug, PartialEq)]
pub struct Value {
    pub field_type: u16,
    pub count: u64,
    pub bytes: Vec<u8>,
}

/// A parsed IFD: its tags, with every value loaded.
#[derive(Clone, Debug)]
pub struct Ifd {
    pub order: ByteOrder,
    /// Where the IFD starts, for error messages.
    pub offset: u64,
    pub tags: BTreeMap<u16, Value>,
}

impl Ifd {
    pub fn has(&self, tag: u16) -> bool {
        self.tags.contains_key(&tag)
    }

    /// Every value of an integer tag (any unsigned or signed integer type).
    pub fn u64s(&self, tag: u16) -> Result<Option<Vec<u64>>, CogError> {
        let Some(value) = self.tags.get(&tag) else {
            return Ok(None);
        };
        let order = self.order;
        let b = &value.bytes;
        let values = match value.field_type {
            1 | 7 => b.iter().map(|&v| v as u64).collect(),
            6 => b.iter().map(|&v| v as i8 as i64 as u64).collect(),
            3 => b.chunks_exact(2).map(|c| order.u16(c) as u64).collect(),
            8 => b
                .chunks_exact(2)
                .map(|c| order.u16(c) as i16 as i64 as u64)
                .collect(),
            4 | 13 => b.chunks_exact(4).map(|c| order.u32(c) as u64).collect(),
            9 => b
                .chunks_exact(4)
                .map(|c| order.u32(c) as i32 as i64 as u64)
                .collect(),
            16..=18 => b.chunks_exact(8).map(|c| order.u64(c)).collect(),
            other => {
                return Err(CogError::Invalid(format!(
                    "tag {tag} has field type {other}, expected an integer"
                )));
            }
        };
        Ok(Some(values))
    }

    /// The first value of an integer tag.
    pub fn u64(&self, tag: u16) -> Result<Option<u64>, CogError> {
        Ok(self.u64s(tag)?.and_then(|v| v.first().copied()))
    }

    /// Every value of a numeric tag, as f64.
    pub fn f64s(&self, tag: u16) -> Result<Option<Vec<f64>>, CogError> {
        let Some(value) = self.tags.get(&tag) else {
            return Ok(None);
        };
        let order = self.order;
        let b = &value.bytes;
        let values = match value.field_type {
            11 => b
                .chunks_exact(4)
                .map(|c| f32::from_bits(order.u32(c)) as f64)
                .collect(),
            12 => b
                .chunks_exact(8)
                .map(|c| f64::from_bits(order.u64(c)))
                .collect(),
            5 => b
                .chunks_exact(8)
                .map(|c| order.u32(c) as f64 / order.u32(&c[4..]) as f64)
                .collect(),
            10 => b
                .chunks_exact(8)
                .map(|c| order.u32(c) as i32 as f64 / order.u32(&c[4..]) as i32 as f64)
                .collect(),
            _ => match self.u64s(tag)? {
                Some(ints) if matches!(value.field_type, 6 | 8 | 9 | 17) => {
                    ints.into_iter().map(|v| v as i64 as f64).collect()
                }
                Some(ints) => ints.into_iter().map(|v| v as f64).collect(),
                None => Vec::new(),
            },
        };
        Ok(Some(values))
    }

    /// An ASCII tag's text, without trailing NULs.
    pub fn ascii(&self, tag: u16) -> Option<String> {
        let value = self.tags.get(&tag)?;
        let text = String::from_utf8_lossy(&value.bytes);
        Some(text.trim_end_matches('\0').to_string())
    }

    /// A tag's raw bytes, e.g. `JPEGTables`.
    pub fn bytes(&self, tag: u16) -> Option<&[u8]> {
        self.tags.get(&tag).map(|v| v.bytes.as_slice())
    }
}

/// Bytes of the file read so far, so that IFDs and tag values inside them
/// don't need another request.
struct Cache<'a> {
    source: &'a dyn ByteSource,
    file_size: Option<u64>,
    chunks: Vec<(u64, Vec<u8>)>,
}

impl Cache<'_> {
    fn lookup(&self, offset: u64, length: u64) -> Option<&[u8]> {
        self.chunks.iter().find_map(|(start, bytes)| {
            let end = start + bytes.len() as u64;
            (offset >= *start && offset + length <= end)
                .then(|| &bytes[(offset - start) as usize..][..length as usize])
        })
    }

    fn check_in_file(&self, offset: u64, length: u64, what: &str) -> Result<(), CogError> {
        match self.file_size {
            Some(size) if offset.checked_add(length).is_none_or(|end| end > size) => {
                Err(CogError::Invalid(format!(
                    "{what} at offset {offset} ({length} bytes) is past the end of the \
                     {size}-byte file"
                )))
            }
            _ => Ok(()),
        }
    }

    /// Bytes at `offset`, read from the source unless already cached.
    fn get(&mut self, offset: u64, length: u64, what: &str) -> Result<Vec<u8>, CogError> {
        if let Some(bytes) = self.lookup(offset, length) {
            return Ok(bytes.to_vec());
        }
        self.check_in_file(offset, length, what)?;
        let bytes = self.source.read_range(offset, length)?;
        self.chunks.push((offset, bytes.clone()));
        Ok(bytes)
    }
}

/// An IFD entry whose value lies outside the IFD.
struct Pending {
    ifd: usize,
    tag: u16,
    offset: u64,
    length: u64,
}

/// Reads the header and every IFD in the chain, given the first bytes of the
/// file (as many as were read: `prefix` may be shorter than [`PREFIX_SIZE`]
/// for small files).
pub fn read_ifds(
    source: &dyn ByteSource,
    prefix: Vec<u8>,
    file_size: Option<u64>,
) -> Result<(Header, Vec<Ifd>), CogError> {
    let header = parse_header(&prefix)?;
    let (count_size, entry_size, next_size) = header.sizes();
    let order = header.order;
    let mut cache = Cache {
        source,
        file_size,
        chunks: vec![(0, prefix)],
    };
    let mut ifds = Vec::new();
    let mut pending = Vec::new();
    let mut seen = HashSet::new();
    let mut next = header.first_ifd;
    while next != 0 {
        if !seen.insert(next) || ifds.len() >= MAX_IFDS {
            return Err(CogError::Invalid(format!(
                "IFD chain loops at offset {next}"
            )));
        }
        let count_bytes = cache.get(next, count_size, "IFD")?;
        let count = if header.big_tiff {
            order.u64(&count_bytes)
        } else {
            order.u16(&count_bytes) as u64
        };
        if count > 10_000 {
            return Err(CogError::Invalid(format!(
                "IFD at offset {next} claims {count} entries"
            )));
        }
        let body = cache.get(
            next + count_size,
            count * entry_size + next_size,
            "IFD entries",
        )?;
        let mut ifd = Ifd {
            order,
            offset: next,
            tags: BTreeMap::new(),
        };
        for entry in body[..(count * entry_size) as usize].chunks_exact(entry_size as usize) {
            parse_entry(&header, entry, ifds.len(), &mut ifd, &mut pending, &cache);
        }
        let tail = &body[(count * entry_size) as usize..];
        next = if header.big_tiff {
            order.u64(tail)
        } else {
            order.u32(tail) as u64
        };
        ifds.push(ifd);
    }
    if ifds.is_empty() {
        return Err(CogError::Invalid("the file has no images (no IFDs)".into()));
    }
    load_pending(&mut cache, &mut ifds, pending)?;
    Ok((header, ifds))
}

/// Parses one IFD entry into `ifd`, or into `pending` if its value lies
/// outside the bytes read so far. Unknown field types are skipped.
fn parse_entry(
    header: &Header,
    entry: &[u8],
    ifd_index: usize,
    ifd: &mut Ifd,
    pending: &mut Vec<Pending>,
    cache: &Cache,
) {
    let order = header.order;
    let tag = order.u16(entry);
    let field_type = order.u16(&entry[2..]);
    let Some(size) = type_size(field_type) else {
        return;
    };
    let (count, value_field) = if header.big_tiff {
        (order.u64(&entry[4..]), &entry[12..20])
    } else {
        (order.u32(&entry[4..]) as u64, &entry[8..12])
    };
    let Some(length) = count.checked_mul(size) else {
        return;
    };
    let value = |bytes: Vec<u8>| Value {
        field_type,
        count,
        bytes,
    };
    if length <= value_field.len() as u64 {
        ifd.tags
            .insert(tag, value(value_field[..length as usize].to_vec()));
        return;
    }
    let offset = if header.big_tiff {
        order.u64(value_field)
    } else {
        order.u32(value_field) as u64
    };
    match cache.lookup(offset, length) {
        Some(bytes) => {
            ifd.tags.insert(tag, value(bytes.to_vec()));
        }
        None => {
            // Placeholder; filled in by `load_pending`.
            ifd.tags.insert(tag, value(Vec::new()));
            pending.push(Pending {
                ifd: ifd_index,
                tag,
                offset,
                length,
            });
        }
    }
}

/// Fetches all out-of-prefix tag values in one batch of reads.
fn load_pending(
    cache: &mut Cache,
    ifds: &mut [Ifd],
    pending: Vec<Pending>,
) -> Result<(), CogError> {
    if pending.is_empty() {
        return Ok(());
    }
    for p in &pending {
        cache.check_in_file(p.offset, p.length, &format!("the value of tag {}", p.tag))?;
    }
    let ranges: Vec<(u64, u64)> = pending.iter().map(|p| (p.offset, p.length)).collect();
    let values = cache.source.read_ranges(&ranges)?;
    for (p, bytes) in pending.into_iter().zip(values) {
        if let Some(value) = ifds[p.ifd].tags.get_mut(&p.tag) {
            value.bytes = bytes;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_header_should_read_classic_and_big_tiff() {
        let classic = [b'I', b'I', 42, 0, 8, 0, 0, 0];
        assert_eq!(
            parse_header(&classic).unwrap(),
            Header {
                order: ByteOrder::Little,
                big_tiff: false,
                first_ifd: 8
            }
        );
        let big = [b'M', b'M', 0, 43, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 16];
        assert_eq!(
            parse_header(&big).unwrap(),
            Header {
                order: ByteOrder::Big,
                big_tiff: true,
                first_ifd: 16
            }
        );
    }

    #[test]
    fn parse_header_should_reject_other_files() {
        assert!(matches!(
            parse_header(b"\x89PNG\r\n\x1a\n"),
            Err(CogError::NotTiff(_))
        ));
        assert!(matches!(parse_header(b"II"), Err(CogError::NotTiff(_))));
    }

    #[test]
    fn ifd_values_should_decode_in_either_byte_order() {
        for order in [ByteOrder::Little, ByteOrder::Big] {
            let mut tags = BTreeMap::new();
            let shorts: Vec<u8> = [1u16, 300]
                .iter()
                .flat_map(|v| match order {
                    ByteOrder::Little => v.to_le_bytes(),
                    ByteOrder::Big => v.to_be_bytes(),
                })
                .collect();
            tags.insert(
                1,
                Value {
                    field_type: 3,
                    count: 2,
                    bytes: shorts,
                },
            );
            let double = match order {
                ByteOrder::Little => 2.5f64.to_le_bytes(),
                ByteOrder::Big => 2.5f64.to_be_bytes(),
            };
            tags.insert(
                2,
                Value {
                    field_type: 12,
                    count: 1,
                    bytes: double.to_vec(),
                },
            );
            tags.insert(
                3,
                Value {
                    field_type: 2,
                    count: 4,
                    bytes: b"abc\0".to_vec(),
                },
            );
            let ifd = Ifd {
                order,
                offset: 0,
                tags,
            };
            assert_eq!(ifd.u64s(1).unwrap(), Some(vec![1, 300]));
            assert_eq!(ifd.f64s(1).unwrap(), Some(vec![1.0, 300.0]));
            assert_eq!(ifd.f64s(2).unwrap(), Some(vec![2.5]));
            assert_eq!(ifd.ascii(3).as_deref(), Some("abc"));
            assert!(ifd.u64s(2).is_err());
            assert_eq!(ifd.u64(9).unwrap(), None);
        }
    }
}
