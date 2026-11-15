//! The spectral viewer: every band's value under the cursor, plotted against
//! wavelength (or band number), with spectra pinned by clicking the map.

use eframe::egui;
use egui_plot::{Legend, Line, LineStyle, Plot, Points, Span, VLine};

use crate::cog::CogInfo;
use crate::composite;

/// The bands under one pixel.
#[derive(Clone, Debug, PartialEq)]
pub struct Sample {
    /// Full-resolution column and row.
    pub pixel: [u32; 2],
    /// The level the values were read from (0 is full resolution).
    pub level: u8,
    /// One value per band, NaN where there's no data.
    pub values: Vec<f32>,
}

struct Pinned {
    sample: Sample,
    color: egui::Color32,
}

pub struct SpectralView {
    pinned: Vec<Pinned>,
    next_color: usize,
    /// Plot against wavelength when every data band has one.
    by_wavelength: bool,
    /// Apply each band's scale and offset (e.g. to reflectance).
    apply_scale: bool,
}

const PIN_COLORS: [egui::Color32; 8] = [
    egui::Color32::from_rgb(230, 159, 0),
    egui::Color32::from_rgb(86, 180, 233),
    egui::Color32::from_rgb(0, 158, 115),
    egui::Color32::from_rgb(240, 228, 66),
    egui::Color32::from_rgb(0, 114, 178),
    egui::Color32::from_rgb(213, 94, 0),
    egui::Color32::from_rgb(204, 121, 167),
    egui::Color32::from_rgb(170, 170, 170),
];

impl Default for SpectralView {
    fn default() -> Self {
        Self {
            pinned: Vec::new(),
            next_color: 0,
            by_wavelength: true,
            apply_scale: false,
        }
    }
}

/// A band's position on the x axis and value, for plotting.
struct Axis {
    xs: Vec<Option<f64>>,
    /// Whether `xs` are wavelengths (else band numbers).
    by_wavelength: bool,
    label: &'static str,
    scales: Vec<(f64, f64)>,
}

impl SpectralView {
    pub fn pin(&mut self, sample: Sample) {
        let color = PIN_COLORS[self.next_color % PIN_COLORS.len()];
        self.next_color += 1;
        self.pinned.push(Pinned { sample, color });
    }

    pub fn clear(&mut self) {
        self.pinned.clear();
        self.next_color = 0;
    }

    pub fn pinned_pixels(&self) -> impl Iterator<Item = ([u32; 2], egui::Color32)> + '_ {
        self.pinned.iter().map(|pin| (pin.sample.pixel, pin.color))
    }

    fn axis(&self, info: &CogInfo) -> Axis {
        let data_bands = composite::data_band_count(info) as usize;
        let wavelengths = composite::band_wavelengths(info);
        let have_all = wavelengths[..data_bands].iter().all(Option::is_some);
        let (xs, label) = if self.by_wavelength && have_all {
            (wavelengths[..data_bands].to_vec(), "Wavelength (nm)")
        } else {
            ((1..=data_bands).map(|b| Some(b as f64)).collect(), "Band")
        };
        let scales = info.band_info[..data_bands]
            .iter()
            .map(|band| {
                if self.apply_scale {
                    (band.scale.unwrap_or(1.0), band.offset.unwrap_or(0.0))
                } else {
                    (1.0, 0.0)
                }
            })
            .collect();
        Axis {
            by_wavelength: label.starts_with("Wavelength"),
            xs,
            label,
            scales,
        }
    }

    /// Draws the panel: controls, then the plot.
    pub fn ui(
        &mut self,
        ui: &mut egui::Ui,
        info: &CogInfo,
        hover: Option<&Sample>,
        composite: Option<[u32; 3]>,
    ) {
        let has_wavelengths = composite::band_wavelengths(info)
            [..composite::data_band_count(info) as usize]
            .iter()
            .all(Option::is_some);
        let has_scale = info
            .band_info
            .iter()
            .any(|band| band.scale.is_some() || band.offset.is_some());

        ui.horizontal(|ui| {
            ui.strong("Spectrum");
            match hover {
                Some(sample) => ui.label(format!(
                    "pixel {}, {}{}",
                    sample.pixel[0],
                    sample.pixel[1],
                    if sample.level > 0 {
                        format!(" (overview {})", sample.level)
                    } else {
                        String::new()
                    }
                )),
                None => ui.weak("hover over the image; click to pin"),
            };
            ui.separator();
            ui.add_enabled(
                has_wavelengths,
                egui::Checkbox::new(&mut self.by_wavelength, "By wavelength"),
            );
            if has_scale {
                ui.checkbox(&mut self.apply_scale, "Apply scale/offset");
            }
            if !self.pinned.is_empty() {
                if ui.button("Clear pins").clicked() {
                    self.clear();
                }
                if ui
                    .button("Copy CSV")
                    .on_hover_text("Copy the pinned spectra as CSV")
                    .clicked()
                {
                    ui.ctx().copy_text(self.csv(info));
                }
            }
        });

        let axis = self.axis(info);
        let x_label = axis.label;
        Plot::new("spectrum")
            .legend(Legend::default())
            .x_axis_label(x_label)
            .y_axis_label("Value")
            .allow_scroll(false)
            .auto_bounds(egui::Vec2b::new(true, true))
            .show(ui, |plot| {
                if let Some(bands) = composite {
                    let colors = [
                        egui::Color32::from_rgb(220, 60, 60),
                        egui::Color32::from_rgb(60, 180, 60),
                        egui::Color32::from_rgb(70, 110, 230),
                    ];
                    for (channel, (band, color)) in bands.iter().zip(colors).enumerate() {
                        if let Some(Some(x)) = axis.xs.get(*band as usize) {
                            let name = ["R", "G", "B"][channel];
                            // Shade the band's width when it's known.
                            let fwhm = info.band_info.get(*band as usize).and_then(|b| b.fwhm_nm);
                            if let (true, Some(fwhm)) = (axis.by_wavelength, fwhm) {
                                plot.span(
                                    Span::new(name, x - fwhm / 2.0..=x + fwhm / 2.0)
                                        .fill(color.gamma_multiply(0.12))
                                        .border_width(0.0),
                                );
                            }
                            plot.vline(
                                VLine::new(name, *x)
                                    .color(color.gamma_multiply(0.6))
                                    .style(LineStyle::dashed_loose()),
                            );
                        }
                    }
                }
                for (index, pin) in self.pinned.iter().enumerate() {
                    let points = series(&axis, &pin.sample.values);
                    let name = format!(
                        "#{} ({}, {})",
                        index + 1,
                        pin.sample.pixel[0],
                        pin.sample.pixel[1]
                    );
                    plot.line(Line::new(name.clone(), points.clone()).color(pin.color));
                    plot.points(Points::new(name, points).color(pin.color).radius(2.5));
                }
                if let Some(sample) = hover {
                    let points = series(&axis, &sample.values);
                    let color = plot.ctx().global_style().visuals.strong_text_color();
                    plot.line(Line::new("Cursor", points.clone()).color(color).width(2.0));
                    plot.points(Points::new("Cursor", points).color(color).radius(3.0));
                }
            });
    }

    fn csv(&self, info: &CogInfo) -> String {
        let axis = self.axis(info);
        let mut out = String::from("band,x");
        for (index, pin) in self.pinned.iter().enumerate() {
            out += &format!(
                ",pin{}_{}_{}",
                index + 1,
                pin.sample.pixel[0],
                pin.sample.pixel[1]
            );
        }
        out.push('\n');
        for (band, x) in axis.xs.iter().enumerate() {
            out += &format!(
                "{},{}",
                band + 1,
                x.map_or(String::new(), |x| x.to_string())
            );
            for pin in &self.pinned {
                let (scale, offset) = axis.scales[band];
                let value = pin.sample.values.get(band).copied().unwrap_or(f32::NAN);
                let value = f64::from(value) * scale + offset;
                out += &if value.is_finite() {
                    format!(",{value}")
                } else {
                    ",".into()
                };
            }
            out.push('\n');
        }
        out
    }
}

/// Plot points for one sample, skipping bands with no x or no data.
fn series(axis: &Axis, values: &[f32]) -> Vec<[f64; 2]> {
    axis.xs
        .iter()
        .zip(values)
        .zip(&axis.scales)
        .filter_map(|((x, value), (scale, offset))| {
            let value = f64::from(*value) * scale + offset;
            value.is_finite().then_some([(*x)?, value])
        })
        .collect()
}
