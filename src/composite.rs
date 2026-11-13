//! Band composites: which band goes to red, green and blue, and presets
//! chosen from what the bands measure.
//!
//! A band's wavelength comes from the file when it says (GDAL metadata), or
//! is guessed from its name: Sentinel-2 (`B02`, `B8A`), Landsat 8/9 (`SR_B4`),
//! and plain names such as `red`, `nir` or `swir16` (Earth Search, MicaSense,
//! ...). Presets then pick the band nearest each wavelength they need.

use crate::cog::CogInfo;

/// A spectral region, with its nominal centre wavelength and how far from it
/// a band may be and still count.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Region {
    Coastal,
    Blue,
    Green,
    Red,
    RedEdge,
    Nir,
    Swir1,
    Swir2,
}

impl Region {
    /// (centre, tolerance) in nanometres.
    fn window(self) -> (f64, f64) {
        match self {
            Region::Coastal => (443.0, 25.0),
            Region::Blue => (485.0, 40.0),
            Region::Green => (555.0, 40.0),
            Region::Red => (660.0, 35.0),
            Region::RedEdge => (720.0, 30.0),
            Region::Nir => (840.0, 80.0),
            Region::Swir1 => (1610.0, 150.0),
            Region::Swir2 => (2200.0, 200.0),
        }
    }
}

/// Sentinel-2 MSI centre wavelengths (nm).
const SENTINEL2: [(&str, f64); 13] = [
    ("B01", 443.0),
    ("B02", 490.0),
    ("B03", 560.0),
    ("B04", 665.0),
    ("B05", 705.0),
    ("B06", 740.0),
    ("B07", 783.0),
    ("B08", 842.0),
    ("B8A", 865.0),
    ("B09", 945.0),
    ("B10", 1375.0),
    ("B11", 1610.0),
    ("B12", 2190.0),
];

/// Landsat 8/9 OLI centre wavelengths (nm), bands 1–7 and 9.
const LANDSAT_OLI: [(u32, f64); 8] = [
    (1, 443.0),
    (2, 482.0),
    (3, 562.0),
    (4, 655.0),
    (5, 865.0),
    (6, 1610.0),
    (7, 2200.0),
    (9, 1373.0),
];

/// Names that say what a band is, with a nominal wavelength. Longer names
/// first, so `rededge` wins over `red`.
const NAMED: [(&str, f64); 16] = [
    ("coastal", 443.0),
    ("rededge", 717.0),
    ("red_edge", 717.0),
    ("red edge", 717.0),
    ("swir16", 1610.0),
    ("swir22", 2200.0),
    ("swir1", 1610.0),
    ("swir2", 2200.0),
    ("nir08", 865.0),
    ("nir", 842.0),
    ("near infrared", 842.0),
    ("blue", 485.0),
    ("green", 555.0),
    ("red", 660.0),
    ("pan", 590.0),
    ("swir", 1610.0),
];

/// A band's centre wavelength in nanometres: from the file, else guessed from
/// its description.
pub fn band_wavelengths(info: &CogInfo) -> Vec<Option<f64>> {
    info.band_info
        .iter()
        .map(|band| {
            band.wavelength_nm
                .or_else(|| band.description.as_deref().and_then(wavelength_from_name))
        })
        .collect()
}

fn wavelength_from_name(name: &str) -> Option<f64> {
    let lower = name.trim().to_ascii_lowercase();
    let upper = lower.to_ascii_uppercase();
    // Sentinel-2: B02, B8A (also B2 for B02).
    for (band, wavelength) in SENTINEL2 {
        let short = band.replace("B0", "B");
        if upper == band || upper == short || upper.ends_with(&format!("_{band}")) {
            return Some(wavelength);
        }
    }
    // Landsat: SR_B4, B4 already matched as Sentinel-2's B4 (both red).
    if let Some(number) = upper
        .strip_prefix("SR_B")
        .and_then(|n| n.parse::<u32>().ok())
    {
        return LANDSAT_OLI
            .iter()
            .find(|(band, _)| *band == number)
            .map(|(_, wavelength)| *wavelength);
    }
    NAMED
        .iter()
        .find(|(word, _)| lower == *word || lower.starts_with(word) || lower.contains(word))
        .map(|(_, wavelength)| *wavelength)
}

/// The band nearest a region's centre, if any is within its tolerance.
fn find(wavelengths: &[Option<f64>], region: Region) -> Option<u32> {
    let (centre, tolerance) = region.window();
    wavelengths
        .iter()
        .enumerate()
        .filter_map(|(band, wavelength)| Some((band, ((*wavelength)? - centre).abs())))
        .filter(|(_, distance)| *distance <= tolerance)
        .min_by(|a, b| a.1.total_cmp(&b.1))
        .map(|(band, _)| band as u32)
}

#[derive(Clone, Debug, PartialEq)]
pub struct Preset {
    pub name: &'static str,
    pub bands: [u32; 3],
}

const RECIPES: [(&str, [Region; 3]); 8] = [
    ("True colour", [Region::Red, Region::Green, Region::Blue]),
    ("Colour infrared", [Region::Nir, Region::Red, Region::Green]),
    (
        "Short-wave infrared",
        [Region::Swir2, Region::Nir, Region::Red],
    ),
    ("Agriculture", [Region::Swir1, Region::Nir, Region::Blue]),
    ("Urban", [Region::Swir2, Region::Swir1, Region::Red]),
    ("Land / water", [Region::Nir, Region::Swir1, Region::Red]),
    ("Red edge", [Region::Nir, Region::RedEdge, Region::Green]),
    ("Bathymetric", [Region::Red, Region::Green, Region::Coastal]),
];

/// The presets this file has bands for. Always has at least one entry for a
/// file with any bands.
pub fn presets(info: &CogInfo) -> Vec<Preset> {
    let wavelengths = band_wavelengths(info);
    let mut out: Vec<Preset> = RECIPES
        .iter()
        .filter_map(|(name, regions)| {
            let bands = [
                find(&wavelengths, regions[0])?,
                find(&wavelengths, regions[1])?,
                find(&wavelengths, regions[2])?,
            ];
            // Two regions landing on one band isn't the composite it claims.
            (bands[0] != bands[1] && bands[1] != bands[2] && bands[0] != bands[2])
                .then_some(Preset { name, bands })
        })
        .collect();
    let data_bands = data_band_count(info);
    if out.is_empty() && data_bands >= 3 {
        out.push(Preset {
            name: "Bands 1, 2, 3",
            bands: [0, 1, 2],
        });
    }
    out
}

/// Bands that hold data, not alpha.
pub fn data_band_count(info: &CogInfo) -> u32 {
    if info.alpha_band.is_some() {
        info.bands - 1
    } else {
        info.bands
    }
}

/// The composite a file opens with: RGB as is, else true colour if it can be
/// found, else the first three bands. `None` for one band (draw it alone).
pub fn default_bands(info: &CogInfo) -> Option<[u32; 3]> {
    if info.is_rgb() {
        return Some([0, 1, 2]);
    }
    if data_band_count(info) < 3 {
        return None;
    }
    Some(
        presets(info)
            .first()
            .map_or([0, 1, 2], |preset| preset.bands),
    )
}

/// "B04 (665 nm)", "Band 3", ...
pub fn band_label(info: &CogInfo, band: u32) -> String {
    let wavelengths = band_wavelengths(info);
    let description = info
        .band_info
        .get(band as usize)
        .and_then(|b| b.description.clone())
        .filter(|d| !d.trim().is_empty());
    let wavelength = wavelengths.get(band as usize).copied().flatten();
    match (description, wavelength) {
        (Some(d), Some(w)) => format!("{} · {d} ({w:.0} nm)", band + 1),
        (Some(d), None) => format!("{} · {d}", band + 1),
        (None, Some(w)) => format!("Band {} ({w:.0} nm)", band + 1),
        (None, None) => format!("Band {}", band + 1),
    }
}

/// Whether to offer the spectral view: more than one data band, and not a
/// plain RGB image.
pub fn is_multispectral(info: &CogInfo) -> bool {
    !info.is_rgb() && data_band_count(info) >= 2
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cog::{BandInfo, Compression, DataType, Planar};

    fn info(names: &[&str]) -> CogInfo {
        CogInfo {
            width: 1,
            height: 1,
            bands: names.len() as u32,
            data_type: DataType::U16,
            compression: Compression::None,
            planar: Planar::Chunky,
            photometric: 1,
            alpha_band: None,
            nodata: None,
            has_mask: false,
            geo: None,
            crs: Default::default(),
            band_info: names
                .iter()
                .map(|name| BandInfo {
                    description: (!name.is_empty()).then(|| name.to_string()),
                    ..BandInfo::default()
                })
                .collect(),
            file_size: None,
        }
    }

    #[test]
    fn sentinel2_names_should_give_true_colour_and_colour_infrared() {
        let s2 = info(&[
            "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12",
        ]);
        let presets = presets(&s2);
        assert_eq!(
            presets[0],
            Preset {
                name: "True colour",
                bands: [2, 1, 0]
            }
        );
        assert_eq!(
            presets[1],
            Preset {
                name: "Colour infrared",
                bands: [6, 2, 1]
            }
        );
        assert!(
            presets
                .iter()
                .any(|p| p.name == "Short-wave infrared" && p.bands == [9, 6, 2])
        );
    }

    #[test]
    fn micasense_names_should_be_recognised() {
        let rededge = info(&["Blue", "Green", "Red", "NIR", "Red edge"]);
        assert_eq!(
            band_wavelengths(&rededge),
            vec![
                Some(485.0),
                Some(555.0),
                Some(660.0),
                Some(842.0),
                Some(717.0)
            ]
        );
        assert_eq!(default_bands(&rededge), Some([2, 1, 0]));
    }

    #[test]
    fn wavelengths_from_the_file_should_beat_names() {
        let mut file = info(&["red", "x", "y"]);
        file.band_info[0].wavelength_nm = Some(842.0);
        assert_eq!(band_wavelengths(&file)[0], Some(842.0));
    }

    #[test]
    fn unknown_bands_should_default_to_the_first_three() {
        assert_eq!(default_bands(&info(&["", "", "", ""])), Some([0, 1, 2]));
        assert_eq!(presets(&info(&["", "", ""]))[0].name, "Bands 1, 2, 3");
    }

    #[test]
    fn one_band_should_have_no_composite() {
        assert_eq!(default_bands(&info(&["elevation"])), None);
        assert!(!is_multispectral(&info(&["elevation"])));
    }

    #[test]
    fn rgb_images_should_not_be_multispectral() {
        let mut rgb = info(&["", "", ""]);
        rgb.data_type = DataType::U8;
        rgb.photometric = 2;
        assert!(!is_multispectral(&rgb));
        assert!(is_multispectral(&info(&["", "", ""])));
    }

    #[test]
    fn landsat_surface_reflectance_names_should_map_to_oli_bands() {
        assert_eq!(wavelength_from_name("SR_B5"), Some(865.0));
        assert_eq!(wavelength_from_name("swir16"), Some(1610.0));
        assert_eq!(wavelength_from_name("rededge1"), Some(717.0));
    }
}
