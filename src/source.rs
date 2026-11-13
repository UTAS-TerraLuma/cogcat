//! Where COG bytes come from: a local file, an HTTP(S) server that supports
//! range requests, or cloud object storage (`s3://`, `gs://`, `az://`).
//!
//! All reads block. They run on the loader's worker threads, never the UI's.

use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::Duration;

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

/// Upper bound on concurrent requests in one `read_ranges` call: enough to
/// hide latency without hammering servers.
pub const MAX_PARALLEL_REQUESTS: usize = 8;

/// Opens `location`: a file path, `file://`, `http(s)://`, or with the `cloud`
/// feature `s3://`, `gs://` or `az://` (also `abfs(s)://`).
pub fn open_source(location: &str) -> Result<Arc<dyn ByteSource>, SourceError> {
    let location = location.trim();
    let Some(scheme) = scheme(location) else {
        return Ok(Arc::new(LocalSource::open(Path::new(location))?));
    };
    match scheme.to_ascii_lowercase().as_str() {
        "http" | "https" => Ok(Arc::new(HttpSource::new(location))),
        "file" => {
            let path = url::Url::parse(location)
                .ok()
                .and_then(|url| url.to_file_path().ok())
                .ok_or_else(|| SourceError::Unsupported(location.into()))?;
            Ok(Arc::new(LocalSource::open(&path)?))
        }
        "s3" | "s3a" | "gs" | "gcs" | "az" | "azure" | "adl" | "abfs" | "abfss" => {
            open_cloud(location)
        }
        other => Err(SourceError::Http(format!(
            "can't open {location:?}: unknown scheme \"{other}://\" (expected a file path, \
             http(s)://, s3://, gs:// or az://)"
        ))),
    }
}

#[cfg(feature = "cloud")]
fn open_cloud(location: &str) -> Result<Arc<dyn ByteSource>, SourceError> {
    Ok(Arc::new(cloud::ObjectStoreSource::open(location)?))
}

#[cfg(not(feature = "cloud"))]
fn open_cloud(location: &str) -> Result<Arc<dyn ByteSource>, SourceError> {
    Err(SourceError::Unsupported(format!(
        "{location} (this build of cogcat has no cloud storage support; it was built \
         without the `cloud` feature)"
    )))
}

/// The scheme of a URL-like location (`s3` in `s3://...`), or `None` for a
/// plain path. Windows drive letters (`C:\...`) are paths.
fn scheme(location: &str) -> Option<&str> {
    let (scheme, _) = location.split_once("://")?;
    let valid = scheme.len() >= 2
        && scheme.starts_with(|c: char| c.is_ascii_alphabetic())
        && scheme
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '+' | '-' | '.'));
    valid.then_some(scheme)
}

/// Runs `read` over `ranges` on up to [`MAX_PARALLEL_REQUESTS`] threads,
/// keeping the order. Stops handing out work after the first error.
fn read_parallel<F>(ranges: &[(u64, u64)], read: F) -> Result<Vec<Vec<u8>>, SourceError>
where
    F: Fn(u64, u64) -> Result<Vec<u8>, SourceError> + Sync,
{
    let threads = ranges.len().min(MAX_PARALLEL_REQUESTS);
    if threads <= 1 {
        return ranges.iter().map(|&(o, l)| read(o, l)).collect();
    }
    let next = AtomicUsize::new(0);
    let failed = std::sync::atomic::AtomicBool::new(false);
    let mut done: Vec<(usize, Result<Vec<u8>, SourceError>)> = std::thread::scope(|scope| {
        let workers: Vec<_> = (0..threads)
            .map(|_| {
                scope.spawn(|| {
                    let mut done = Vec::new();
                    while !failed.load(Ordering::Relaxed) {
                        let i = next.fetch_add(1, Ordering::Relaxed);
                        let Some(&(offset, length)) = ranges.get(i) else {
                            break;
                        };
                        let result = read(offset, length);
                        if result.is_err() {
                            failed.store(true, Ordering::Relaxed);
                        }
                        done.push((i, result));
                    }
                    done
                })
            })
            .collect();
        workers
            .into_iter()
            .flat_map(|w| w.join().expect("range request thread panicked"))
            .collect()
    });
    done.sort_by_key(|(i, _)| *i);
    if let Some(pos) = done.iter().position(|(_, r)| r.is_err()) {
        return Err(done.swap_remove(pos).1.unwrap_err());
    }
    Ok(done.into_iter().map(|(_, r)| r.expect("checked")).collect())
}

// ---------------------------------------------------------------- local files

/// A local file, read with positional reads so threads don't share a seek
/// position.
pub struct LocalSource {
    file: File,
    path: PathBuf,
    size: u64,
}

impl LocalSource {
    pub fn open(path: &Path) -> Result<LocalSource, SourceError> {
        let context = |e: std::io::Error| {
            SourceError::Io(std::io::Error::new(
                e.kind(),
                format!("couldn't open {}: {e}", path.display()),
            ))
        };
        let file = File::open(path).map_err(context)?;
        let metadata = file.metadata().map_err(context)?;
        if metadata.is_dir() {
            return Err(SourceError::Io(std::io::Error::other(format!(
                "{} is a directory, not a file",
                path.display()
            ))));
        }
        Ok(LocalSource {
            file,
            path: path.to_path_buf(),
            size: metadata.len(),
        })
    }
}

#[cfg(unix)]
fn read_exact_at(file: &File, buf: &mut [u8], offset: u64) -> std::io::Result<()> {
    std::os::unix::fs::FileExt::read_exact_at(file, buf, offset)
}

#[cfg(windows)]
fn read_exact_at(file: &File, mut buf: &mut [u8], mut offset: u64) -> std::io::Result<()> {
    use std::os::windows::fs::FileExt;
    while !buf.is_empty() {
        match file.seek_read(buf, offset) {
            Ok(0) => return Err(std::io::ErrorKind::UnexpectedEof.into()),
            Ok(n) => {
                buf = &mut buf[n..];
                offset += n as u64;
            }
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
            Err(e) => return Err(e),
        }
    }
    Ok(())
}

impl ByteSource for LocalSource {
    fn read_range(&self, offset: u64, length: u64) -> Result<Vec<u8>, SourceError> {
        if offset.saturating_add(length) > self.size {
            return Err(SourceError::ShortRead {
                offset,
                expected: length,
                received: self.size.saturating_sub(offset).min(length),
            });
        }
        let mut buf = vec![0; length as usize];
        read_exact_at(&self.file, &mut buf, offset).map_err(|e| {
            SourceError::Io(std::io::Error::new(
                e.kind(),
                format!(
                    "couldn't read {length} bytes at offset {offset} of {}: {e}",
                    self.path.display()
                ),
            ))
        })?;
        Ok(buf)
    }

    fn size(&self) -> Result<Option<u64>, SourceError> {
        Ok(Some(self.size))
    }
}

// ---------------------------------------------------------------- HTTP(S)

/// A file on an HTTP(S) server that supports range requests.
pub struct HttpSource {
    url: String,
    agent: ureq::Agent,
    /// Learned from a `Content-Range` reply or a HEAD request.
    size: OnceLock<Option<u64>>,
}

/// One agent for every HTTP source, so they share pooled connections.
fn http_agent() -> ureq::Agent {
    static AGENT: OnceLock<ureq::Agent> = OnceLock::new();
    AGENT
        .get_or_init(|| {
            let tls = ureq::tls::TlsConfig::builder()
                .root_certs(root_certs())
                .build();
            ureq::Agent::config_builder()
                .tls_config(tls)
                .timeout_connect(Some(Duration::from_secs(20)))
                .timeout_recv_response(Some(Duration::from_secs(60)))
                .timeout_recv_body(Some(Duration::from_secs(120)))
                .max_idle_connections_per_host(2 * MAX_PARALLEL_REQUESTS)
                .user_agent(concat!("cogcat/", env!("CARGO_PKG_VERSION")))
                .build()
                .new_agent()
        })
        .clone()
}

/// The operating system's trusted certificates, so HTTPS works behind
/// proxies that re-sign TLS with a locally trusted CA (common on university
/// and corporate networks). Falls back to the bundled Mozilla roots.
fn root_certs() -> ureq::tls::RootCerts {
    let native = rustls_native_certs::load_native_certs();
    if native.certs.is_empty() {
        return ureq::tls::RootCerts::WebPki;
    }
    let certs: Vec<ureq::tls::Certificate<'static>> = native
        .certs
        .iter()
        .map(|c| ureq::tls::Certificate::from_der(c.as_ref()).to_owned())
        .collect();
    ureq::tls::RootCerts::new_with_certs(&certs)
}

/// Total size from a `Content-Range: bytes a-b/total` header.
fn content_range_total(value: &str) -> Option<u64> {
    value.rsplit_once('/')?.1.trim().parse().ok()
}

/// A user-facing message for a failed HTTP request.
fn http_error_message(url: &str, error: &ureq::Error) -> String {
    match error {
        ureq::Error::StatusCode(code) => {
            let hint = match code {
                401 | 403 => {
                    " (access denied: the file may be private, or a signed URL may have expired)"
                }
                404 => " (not found)",
                416 => " (range not satisfiable: the file is shorter than expected)",
                _ => "",
            };
            format!("HTTP {code} from {url}{hint}")
        }
        other => format!("couldn't fetch {url}: {other}"),
    }
}

/// Whether a failed request is worth one retry.
fn is_transient(error: &ureq::Error) -> bool {
    match error {
        ureq::Error::StatusCode(code) => matches!(code, 500 | 502 | 503 | 504),
        ureq::Error::Io(_) | ureq::Error::Timeout(_) | ureq::Error::ConnectionFailed => true,
        _ => false,
    }
}

impl HttpSource {
    pub fn new(url: &str) -> HttpSource {
        HttpSource {
            url: url.to_string(),
            agent: http_agent(),
            size: OnceLock::new(),
        }
    }

    fn get_range(&self, offset: u64, length: u64) -> Result<Vec<u8>, SourceError> {
        if length == 0 {
            return Ok(Vec::new());
        }
        // One retry for dropped connections, timeouts and 5xx replies.
        match self.try_get_range(offset, length) {
            Err(Attempt::Transient(_)) => self.try_get_range(offset, length),
            other => other,
        }
        .map_err(|a| a.into_error(&self.url))
    }

    fn try_get_range(&self, offset: u64, length: u64) -> Result<Vec<u8>, Attempt> {
        let last_byte = offset + length - 1;
        let mut response = self
            .agent
            .get(&self.url)
            .header("Range", format!("bytes={offset}-{last_byte}"))
            .call()
            .map_err(|e| Attempt::from_ureq(e, &self.url))?;
        // A server that ignores Range sends the whole file with 200, which
        // for a big COG means downloading all of it. Refuse instead.
        let status = response.status().as_u16();
        if status != 206 {
            return Err(Attempt::Failed(SourceError::Http(format!(
                "{} replied {status} instead of 206 Partial Content: the server doesn't \
                 support range requests, which streaming a COG needs",
                self.url
            ))));
        }
        if let Some(total) = response
            .headers()
            .get("content-range")
            .and_then(|v| v.to_str().ok())
            .and_then(content_range_total)
        {
            let _ = self.size.set(Some(total));
        }
        // ureq errors once the body *reaches* the limit, so allow a byte of
        // slack; the length check below rejects anything oversized.
        let bytes = response
            .body_mut()
            .with_config()
            .limit(length + 1)
            .read_to_vec()
            .map_err(|e| Attempt::from_ureq(e, &self.url))?;
        if bytes.len() as u64 != length {
            return Err(Attempt::Failed(SourceError::ShortRead {
                offset,
                expected: length,
                received: bytes.len() as u64,
            }));
        }
        Ok(bytes)
    }

    fn fetch_size(&self) -> Result<Option<u64>, SourceError> {
        let response = self
            .agent
            .head(&self.url)
            .call()
            .map_err(|e| SourceError::Http(http_error_message(&self.url, &e)))?;
        let length = response
            .headers()
            .get("content-length")
            .and_then(|v| v.to_str().ok()?.trim().parse().ok());
        if length.is_some() {
            return Ok(length);
        }
        // No length from HEAD (or chunked): ask for one byte and read the
        // total from Content-Range.
        self.get_range(0, 1)?;
        Ok(self.size.get().copied().flatten())
    }
}

/// A failed request, and whether it's worth retrying.
enum Attempt {
    Transient(ureq::Error),
    Failed(SourceError),
}

impl Attempt {
    fn from_ureq(error: ureq::Error, url: &str) -> Attempt {
        if is_transient(&error) {
            Attempt::Transient(error)
        } else {
            Attempt::Failed(SourceError::Http(http_error_message(url, &error)))
        }
    }

    fn into_error(self, url: &str) -> SourceError {
        match self {
            Attempt::Transient(error) => SourceError::Http(http_error_message(url, &error)),
            Attempt::Failed(error) => error,
        }
    }
}

impl ByteSource for HttpSource {
    fn read_range(&self, offset: u64, length: u64) -> Result<Vec<u8>, SourceError> {
        self.get_range(offset, length)
    }

    fn read_ranges(&self, ranges: &[(u64, u64)]) -> Result<Vec<Vec<u8>>, SourceError> {
        read_parallel(ranges, |offset, length| self.get_range(offset, length))
    }

    fn size(&self) -> Result<Option<u64>, SourceError> {
        if let Some(size) = self.size.get() {
            return Ok(*size);
        }
        let size = self.fetch_size()?;
        Ok(*self.size.get_or_init(|| size))
    }
}

// ---------------------------------------------------------------- cloud storage

#[cfg(feature = "cloud")]
mod cloud {
    //! `s3://`, `gs://` and `az://` through `object_store`, on a shared tokio
    //! runtime that the blocking reads wait on.

    use std::future::Future;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Arc, OnceLock};

    use object_store::path::Path as StorePath;
    use object_store::{ObjectStore, ObjectStoreExt};

    use super::{SourceError, http_agent};

    pub struct ObjectStoreSource {
        store: Arc<dyn ObjectStore>,
        /// When requests are signed only because credentials happen to be
        /// in the environment, an unsigned store to retry with if they're
        /// refused: those credentials may belong to another account, while
        /// the bucket is public.
        unsigned: Option<Arc<dyn ObjectStore>>,
        /// Set once the unsigned store worked where the signed one didn't.
        use_unsigned: AtomicBool,
        path: StorePath,
        location: String,
        size: OnceLock<u64>,
    }

    fn runtime() -> &'static tokio::runtime::Runtime {
        static RUNTIME: OnceLock<tokio::runtime::Runtime> = OnceLock::new();
        RUNTIME.get_or_init(|| {
            tokio::runtime::Builder::new_multi_thread()
                .worker_threads(2)
                .thread_name("cogcat-object-store")
                .enable_all()
                .build()
                .expect("couldn't start the tokio runtime for cloud storage")
        })
    }

    fn env(name: &str) -> Option<String> {
        std::env::var(name).ok().filter(|v| !v.trim().is_empty())
    }

    fn any_env(names: &[&str]) -> bool {
        names.iter().any(|n| env(n).is_some())
    }

    /// GDAL-style `*_NO_SIGN_REQUEST=YES/NO`: `Some(true)` to force unsigned
    /// requests, `Some(false)` to force signing.
    fn no_sign_request(name: &str) -> Option<bool> {
        let value = env(name)?.to_ascii_lowercase();
        Some(matches!(value.as_str(), "yes" | "true" | "1" | "on"))
    }

    const AWS_CREDENTIALS: &[&str] = &[
        "AWS_ACCESS_KEY_ID",
        "AWS_WEB_IDENTITY_TOKEN_FILE",
        "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI",
        "AWS_CONTAINER_CREDENTIALS_FULL_URI",
    ];
    const GOOGLE_CREDENTIALS: &[&str] = &[
        "GOOGLE_SERVICE_ACCOUNT",
        "GOOGLE_SERVICE_ACCOUNT_PATH",
        "GOOGLE_SERVICE_ACCOUNT_KEY",
        "GOOGLE_APPLICATION_CREDENTIALS",
    ];
    const AZURE_CREDENTIALS: &[&str] = &[
        "AZURE_STORAGE_ACCOUNT_KEY",
        "AZURE_STORAGE_ACCESS_KEY",
        "AZURE_STORAGE_KEY",
        "AZURE_STORAGE_SAS_KEY",
        "AZURE_STORAGE_SAS_TOKEN",
        "AZURE_STORAGE_TOKEN",
        "AZURE_CLIENT_ID",
        "AZURE_FEDERATED_TOKEN_FILE",
        "AZURE_USE_AZURE_CLI",
        "AZURE_STORAGE_USE_EMULATOR",
    ];

    fn is_denied(e: &object_store::Error) -> bool {
        matches!(
            e,
            object_store::Error::PermissionDenied { .. }
                | object_store::Error::Unauthenticated { .. }
        )
    }

    impl ObjectStoreSource {
        pub fn open(location: &str) -> Result<ObjectStoreSource, SourceError> {
            let url = url::Url::parse(location)
                .map_err(|e| SourceError::ObjectStore(format!("can't parse {location:?}: {e}")))?;
            let bucket = url.host_str().unwrap_or_default().to_string();
            if bucket.is_empty() {
                return Err(SourceError::ObjectStore(format!(
                    "{location} has no bucket or container name"
                )));
            }
            let path = StorePath::from_url_path(url.path()).map_err(|e| {
                SourceError::ObjectStore(format!("bad object path in {location}: {e}"))
            })?;
            let scheme = url.scheme();
            let (no_sign_var, credentials, credential_files) = match scheme {
                "s3" | "s3a" => ("AWS_NO_SIGN_REQUEST", AWS_CREDENTIALS, false),
                "gs" | "gcs" => (
                    "GS_NO_SIGN_REQUEST",
                    GOOGLE_CREDENTIALS,
                    gcloud_adc_exists(),
                ),
                _ => ("AZURE_NO_SIGN_REQUEST", AZURE_CREDENTIALS, false),
            };
            // Sign only with credentials, so public buckets open without an
            // account; `*_NO_SIGN_REQUEST=YES/NO` (as in GDAL) overrides.
            let forced = no_sign_request(no_sign_var);
            let skip = forced.unwrap_or(!(any_env(credentials) || credential_files));
            let region = matches!(scheme, "s3" | "s3a")
                .then(|| s3_region(&bucket))
                .flatten();
            let build = |skip: bool| match scheme {
                "s3" | "s3a" => build_s3(&bucket, region.clone(), skip),
                "gs" | "gcs" => build_gcs(&bucket, skip),
                _ => build_azure(location, skip),
            };
            let store = build(skip).map_err(|e| {
                SourceError::ObjectStore(format!("couldn't set up access to {location}: {e}"))
            })?;
            let unsigned = (!skip && forced.is_none())
                .then(|| build(true).ok())
                .flatten();
            Ok(ObjectStoreSource {
                store,
                unsigned,
                use_unsigned: AtomicBool::new(false),
                path,
                location: location.to_string(),
                size: OnceLock::new(),
            })
        }

        /// Runs one request, retrying unsigned if signed requests are
        /// refused and an unsigned store exists.
        fn run<T, F, Fut>(&self, op: F) -> Result<T, SourceError>
        where
            F: Fn(Arc<dyn ObjectStore>, StorePath) -> Fut,
            Fut: Future<Output = object_store::Result<T>>,
        {
            let unsigned_now = self.use_unsigned.load(Ordering::Relaxed);
            let store = match (&self.unsigned, unsigned_now) {
                (Some(unsigned), true) => unsigned.clone(),
                _ => self.store.clone(),
            };
            match runtime().block_on(op(store, self.path.clone())) {
                Err(e) if is_denied(&e) && !unsigned_now => {
                    if let Some(unsigned) = &self.unsigned
                        && let Ok(value) =
                            runtime().block_on(op(unsigned.clone(), self.path.clone()))
                    {
                        self.use_unsigned.store(true, Ordering::Relaxed);
                        return Ok(value);
                    }
                    Err(self.error(e))
                }
                result => result.map_err(|e| self.error(e)),
            }
        }

        fn error(&self, e: object_store::Error) -> SourceError {
            let hint = match &e {
                object_store::Error::NotFound { .. } => " (not found)",
                object_store::Error::PermissionDenied { .. }
                | object_store::Error::Unauthenticated { .. } => {
                    " (access denied: set credentials in the environment, e.g. \
                     AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY, \
                     GOOGLE_APPLICATION_CREDENTIALS or AZURE_STORAGE_ACCOUNT_KEY)"
                }
                _ => "",
            };
            SourceError::ObjectStore(format!("{}: {e}{hint}", self.location))
        }
    }

    /// The region to use for an S3 bucket: from the environment (`None`,
    /// as object_store reads it), or else asked of S3 itself.
    fn s3_region(bucket: &str) -> Option<String> {
        let configured = any_env(&[
            "AWS_REGION",
            "AWS_DEFAULT_REGION",
            "AWS_ENDPOINT",
            "AWS_ENDPOINT_URL",
        ]);
        (!configured).then(|| bucket_region(bucket).unwrap_or_else(|| "us-east-1".into()))
    }

    fn build_s3(
        bucket: &str,
        region: Option<String>,
        skip_signature: bool,
    ) -> object_store::Result<Arc<dyn ObjectStore>> {
        let mut builder = object_store::aws::AmazonS3Builder::from_env()
            .with_bucket_name(bucket)
            .with_skip_signature(skip_signature);
        if let Some(region) = region {
            builder = builder.with_region(region);
        }
        Ok(Arc::new(builder.build()?))
    }

    /// Finds an S3 bucket's region from the `x-amz-bucket-region` header,
    /// which S3 sends even when it refuses the request.
    fn bucket_region(bucket: &str) -> Option<String> {
        let url = if bucket.contains('.') {
            format!("https://s3.amazonaws.com/{bucket}")
        } else {
            format!("https://{bucket}.s3.amazonaws.com")
        };
        let response = http_agent()
            .head(&url)
            .config()
            .http_status_as_error(false)
            .max_redirects(0)
            .build()
            .call()
            .ok()?;
        let region = response
            .headers()
            .get("x-amz-bucket-region")?
            .to_str()
            .ok()?;
        Some(region.to_string())
    }

    fn build_gcs(bucket: &str, skip_signature: bool) -> object_store::Result<Arc<dyn ObjectStore>> {
        let builder = object_store::gcp::GoogleCloudStorageBuilder::from_env()
            .with_bucket_name(bucket)
            .with_skip_signature(skip_signature);
        Ok(Arc::new(builder.build()?))
    }

    /// Whether `gcloud auth application-default login` left credentials.
    fn gcloud_adc_exists() -> bool {
        let dir = if cfg!(windows) {
            env("APPDATA").map(|d| std::path::PathBuf::from(d).join("gcloud"))
        } else {
            env("HOME").map(|h| std::path::PathBuf::from(h).join(".config/gcloud"))
        };
        dir.is_some_and(|d| d.join("application_default_credentials.json").is_file())
    }

    fn build_azure(
        location: &str,
        skip_signature: bool,
    ) -> object_store::Result<Arc<dyn ObjectStore>> {
        let builder = object_store::azure::MicrosoftAzureBuilder::from_env()
            .with_url(location)
            .with_skip_signature(skip_signature);
        Ok(Arc::new(builder.build()?))
    }

    impl super::ByteSource for ObjectStoreSource {
        fn read_range(&self, offset: u64, length: u64) -> Result<Vec<u8>, SourceError> {
            if length == 0 {
                return Ok(Vec::new());
            }
            let bytes = self.run(|store, path| async move {
                store.get_range(&path, offset..offset + length).await
            })?;
            check_length(offset, length, bytes.len())?;
            Ok(bytes.to_vec())
        }

        fn read_ranges(&self, ranges: &[(u64, u64)]) -> Result<Vec<Vec<u8>>, SourceError> {
            let wanted: Vec<_> = ranges
                .iter()
                .filter(|(_, length)| *length > 0)
                .map(|&(offset, length)| offset..offset + length)
                .collect();
            let mut fetched = self
                .run(|store, path| {
                    let wanted = wanted.clone();
                    async move { store.get_ranges(&path, &wanted).await }
                })?
                .into_iter();
            ranges
                .iter()
                .map(|&(offset, length)| {
                    if length == 0 {
                        return Ok(Vec::new());
                    }
                    let bytes = fetched.next().unwrap_or_default();
                    check_length(offset, length, bytes.len())?;
                    Ok(bytes.to_vec())
                })
                .collect()
        }

        fn size(&self) -> Result<Option<u64>, SourceError> {
            if let Some(size) = self.size.get() {
                return Ok(Some(*size));
            }
            let meta = self.run(|store, path| async move { store.head(&path).await })?;
            Ok(Some(*self.size.get_or_init(|| meta.size)))
        }
    }

    fn check_length(offset: u64, expected: u64, received: usize) -> Result<(), SourceError> {
        if received as u64 == expected {
            Ok(())
        } else {
            Err(SourceError::ShortRead {
                offset,
                expected,
                received: received as u64,
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scheme_should_tell_urls_from_paths() {
        assert_eq!(scheme("https://example.com/a.tif"), Some("https"));
        assert_eq!(scheme("s3://bucket/key.tif"), Some("s3"));
        assert_eq!(scheme("/data/a.tif"), None);
        assert_eq!(scheme(r"C:\data\a.tif"), None);
        assert_eq!(scheme("C://data/a.tif"), None);
        assert_eq!(scheme("weird name://x"), None);
    }

    #[test]
    fn content_range_total_should_read_the_size() {
        assert_eq!(content_range_total("bytes 0-99/12345"), Some(12345));
        assert_eq!(content_range_total("bytes 0-99/*"), None);
    }

    #[test]
    fn open_source_should_reject_unknown_schemes() {
        assert!(open_source("ftp://example.com/a.tif").is_err());
    }

    #[test]
    fn local_source_should_read_ranges_and_refuse_reads_past_the_end() {
        let path = std::env::temp_dir().join(format!("cogcat-source-{}.bin", std::process::id()));
        std::fs::write(&path, (0..100u8).collect::<Vec<_>>()).unwrap();
        let source = open_source(path.to_str().unwrap()).unwrap();
        assert_eq!(source.size().unwrap(), Some(100));
        assert_eq!(source.read_range(10, 3).unwrap(), [10, 11, 12]);
        assert_eq!(
            source.read_ranges(&[(0, 2), (98, 2)]).unwrap(),
            [vec![0, 1], vec![98, 99]]
        );
        assert!(matches!(
            source.read_range(95, 10),
            Err(SourceError::ShortRead { received: 5, .. })
        ));
        let url = url::Url::from_file_path(&path).unwrap();
        assert_eq!(
            open_source(url.as_str()).unwrap().size().unwrap(),
            Some(100)
        );
        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn local_source_should_name_missing_files() {
        let Err(error) = open_source("/no/such/file.tif") else {
            panic!("opened a missing file");
        };
        assert!(error.to_string().contains("/no/such/file.tif"), "{error}");
    }

    #[test]
    fn read_parallel_should_keep_order_and_report_errors() {
        let ranges: Vec<(u64, u64)> = (0..40).map(|i| (i, 1)).collect();
        let read = |offset: u64, _| Ok(vec![offset as u8]);
        let out = read_parallel(&ranges, read).unwrap();
        assert_eq!(out, (0..40u8).map(|i| vec![i]).collect::<Vec<_>>());
        let failing = |offset: u64, _| {
            if offset == 7 {
                Err(SourceError::Http("boom".into()))
            } else {
                Ok(vec![])
            }
        };
        assert!(read_parallel(&ranges, failing).is_err());
    }
}
