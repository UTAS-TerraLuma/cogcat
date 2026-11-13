//! Cogcat, a Cloud Optimized GeoTIFF viewer, as a library shared by the
//! `cogcat-viewer` window and the `cogcat` command line.

mod app;
pub mod cog;
pub mod composite;
pub mod loader;
pub mod render;
pub mod source;
mod spectral;
pub mod stretch;
pub mod view;
#[cfg(windows)]
mod windows_dpi;

use std::sync::Arc;

use eframe::{egui, egui_wgpu};

use crate::render::Colormap;
use crate::stretch::StretchMode;

/// How the viewer first shows a file. Shared by `cogcat open` and
/// `cogcat-viewer`, which `cogcat open` passes them on to.
#[derive(clap::Args, Clone, Debug, Default, PartialEq)]
pub struct DisplayArgs {
    /// Bands for red, green and blue, counting from 1, e.g. `--bands 8,4,3`.
    #[arg(long, value_parser = parse_list::<u32, 3>, value_name = "R,G,B")]
    pub bands: Option<[u32; 3]>,
    /// Show one band (counting from 1) alone, through `--colormap`.
    #[arg(long, conflicts_with = "bands", value_name = "BAND")]
    pub band: Option<u32>,
    /// Colour map for a single band.
    #[arg(long, value_enum)]
    pub colormap: Option<ColormapArg>,
    /// Contrast stretch.
    #[arg(long, value_enum)]
    pub stretch: Option<StretchArg>,
    /// Low and high percentiles for `--stretch percentile`, e.g. `2,98`.
    #[arg(long, value_parser = parse_list::<f32, 2>, value_name = "LOW,HIGH")]
    pub percentiles: Option<[f32; 2]>,
    /// Gamma applied after the stretch.
    #[arg(long)]
    pub gamma: Option<f32>,
    /// Outline tiles and show what's loading.
    #[arg(long)]
    pub debug_tiles: bool,
}

#[derive(clap::ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
pub enum StretchArg {
    /// Straight linear from the minimum to the maximum.
    Minmax,
    /// Linear between two percentiles (`--percentiles`, default 2,98).
    Percentile,
    /// ENVI's optimized linear stretch.
    Optimized,
    /// Mean ± 2 standard deviations.
    Stddev,
    /// No stretch: 0–255 for 8-bit data, else the type's full range.
    None,
}

#[derive(clap::ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
pub enum ColormapArg {
    Gray,
    Viridis,
    Magma,
    Turbo,
    Terrain,
}

impl From<ColormapArg> for Colormap {
    fn from(arg: ColormapArg) -> Self {
        match arg {
            ColormapArg::Gray => Colormap::Gray,
            ColormapArg::Viridis => Colormap::Viridis,
            ColormapArg::Magma => Colormap::Magma,
            ColormapArg::Turbo => Colormap::Turbo,
            ColormapArg::Terrain => Colormap::Terrain,
        }
    }
}

impl DisplayArgs {
    /// The stretch mode asked for, if any; `Some(None)` means no stretch.
    pub fn stretch_mode(&self) -> Option<Option<StretchMode>> {
        Some(match self.stretch? {
            StretchArg::Minmax => Some(StretchMode::MinMax),
            StretchArg::Percentile => {
                let [low, high] = self.percentiles.unwrap_or([2.0, 98.0]);
                Some(StretchMode::Percentile { low, high })
            }
            StretchArg::Optimized => Some(StretchMode::OPTIMIZED_LINEAR),
            StretchArg::Stddev => Some(StretchMode::StdDev { n: 2.0 }),
            StretchArg::None => None,
        })
    }

    /// These settings as command-line arguments, for starting the viewer.
    pub fn to_args(&self) -> Vec<String> {
        let mut args = Vec::new();
        let join = |values: &[u32]| {
            values
                .iter()
                .map(u32::to_string)
                .collect::<Vec<_>>()
                .join(",")
        };
        if let Some(bands) = &self.bands {
            args.push(format!("--bands={}", join(bands)));
        }
        if let Some(band) = self.band {
            args.push(format!("--band={band}"));
        }
        if let Some(colormap) = self.colormap {
            args.push(format!("--colormap={}", value_name(colormap)));
        }
        if let Some(stretch) = self.stretch {
            args.push(format!("--stretch={}", value_name(stretch)));
        }
        if let Some(percentiles) = &self.percentiles {
            let text: Vec<String> = percentiles.iter().map(f32::to_string).collect();
            args.push(format!("--percentiles={}", text.join(",")));
        }
        if let Some(gamma) = self.gamma {
            args.push(format!("--gamma={gamma}"));
        }
        if self.debug_tiles {
            args.push("--debug-tiles".into());
        }
        args
    }
}

/// Parses `N` comma-separated values, e.g. `8,4,3`.
fn parse_list<T: std::str::FromStr, const N: usize>(text: &str) -> Result<[T; N], String> {
    let values: Vec<T> = text
        .split(',')
        .map(|part| {
            part.trim()
                .parse()
                .map_err(|_| format!("{part:?} isn't a number"))
        })
        .collect::<Result<_, _>>()?;
    values
        .try_into()
        .map_err(|_| format!("expected {N} comma-separated values"))
}

fn value_name(value: impl clap::ValueEnum) -> String {
    value
        .to_possible_value()
        .expect("no skipped values")
        .get_name()
        .to_owned()
}

/// Opens the viewer window and blocks until it's closed. `initial_source` is
/// a COG path or URL to open on startup.
pub fn run_viewer(initial_source: Option<String>, display: DisplayArgs) -> eframe::Result {
    let mut options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("Cogcat")
            .with_inner_size([1400.0, 900.0]),
        ..Default::default()
    };
    // egui asks for 8192 pixel textures at most; big screens and saved images
    // can use what the GPU allows.
    if let egui_wgpu::WgpuSetup::CreateNew(setup) = &mut options.wgpu_options.wgpu_setup {
        let egui_descriptor = setup.device_descriptor.clone();
        setup.device_descriptor = Arc::new(move |adapter| {
            let mut descriptor = egui_descriptor(adapter);
            descriptor.required_limits.max_texture_dimension_2d =
                adapter.limits().max_texture_dimension_2d;
            descriptor
        });
    }
    eframe::run_native(
        "Cogcat",
        options,
        Box::new(|cc| Ok(Box::new(app::ViewerApp::new(cc, initial_source, display)?))),
    )
}

#[cfg(test)]
mod tests {
    use clap::Parser;

    use super::*;

    #[derive(Parser)]
    struct Cli {
        #[command(flatten)]
        display: DisplayArgs,
    }

    #[test]
    fn display_args_should_round_trip_through_the_command_line() {
        let display = DisplayArgs {
            bands: Some([8, 4, 3]),
            band: None,
            colormap: Some(ColormapArg::Viridis),
            stretch: Some(StretchArg::Percentile),
            percentiles: Some([1.0, 99.5]),
            gamma: Some(1.2),
            debug_tiles: true,
        };
        let args = std::iter::once("x".to_owned()).chain(display.to_args());
        assert_eq!(Cli::parse_from(args).display, display);
    }

    #[test]
    fn percentile_stretch_should_default_to_2_and_98() {
        let display = DisplayArgs {
            stretch: Some(StretchArg::Percentile),
            ..DisplayArgs::default()
        };
        assert_eq!(
            display.stretch_mode(),
            Some(Some(StretchMode::Percentile {
                low: 2.0,
                high: 98.0
            }))
        );
    }
}
