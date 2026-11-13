//! Cogcat, a Cloud Optimized GeoTIFF viewer, as a library shared by the
//! `cogcat-viewer` window and the `cogcat` command line.

pub mod cog;
pub mod render;
pub mod source;
pub mod stretch;
#[cfg(windows)]
mod windows_dpi;
