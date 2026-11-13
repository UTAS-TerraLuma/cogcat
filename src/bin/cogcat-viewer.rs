// Release builds are Windows GUI programs, so launching one doesn't open a
// console window. Debug builds keep the console for logs and panics.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use clap::Parser;

/// The Cogcat Cloud Optimized GeoTIFF viewer.
#[derive(Parser)]
#[command(version)]
struct Args {
    /// A COG path or URL (http(s)://, s3://, gs://, az://) to open on startup.
    source: Option<String>,
    #[command(flatten)]
    display: cogcat::DisplayArgs,
}

fn main() -> eframe::Result {
    let args = Args::parse();
    cogcat::run_viewer(args.source, args.display)
}
