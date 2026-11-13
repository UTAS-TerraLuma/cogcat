//! Where COG bytes come from: a local file, an HTTP(S) server that supports
//! range requests, or cloud object storage (`s3://`, `gs://`, `az://`).
//!
//! All reads block. They run on the loader's worker threads, never the UI's.

use std::sync::Arc;

#[derive(Debug, thiserror::Error)]
pub enum SourceError {
    #[error("{0}")]
    Io(#[from] std::io::Error),
    #[error("{0}")]
    Http(String),
    #[error("{0}")]
    ObjectStore(String),
    #[error("unsupported location {0:?}")]
    Unsupported(String),
    #[error("asked for {expected} bytes at offset {offset} but received {received}")]
    ShortRead {
        offset: u64,
        expected: u64,
        received: u64,
    },
}

/// Random access to the bytes of one file.
pub trait ByteSource: Send + Sync {
    /// Reads `length` bytes at `offset`. A read past the end is an error.
    fn read_range(&self, offset: u64, length: u64) -> Result<Vec<u8>, SourceError>;

    /// Reads several ranges, in the same order. Remote sources run them in
    /// parallel.
    fn read_ranges(&self, ranges: &[(u64, u64)]) -> Result<Vec<Vec<u8>>, SourceError> {
        ranges
            .iter()
            .map(|&(offset, length)| self.read_range(offset, length))
            .collect()
    }

    /// The file's size, if the source can tell.
    fn size(&self) -> Result<Option<u64>, SourceError>;
}

/// Opens `location`: a file path, `file://`, `http(s)://`, or with the `cloud`
/// feature `s3://`, `gs://` or `az://` (also `abfs(s)://`).
pub fn open_source(location: &str) -> Result<Arc<dyn ByteSource>, SourceError> {
    let _ = location;
    todo!("agent A")
}
