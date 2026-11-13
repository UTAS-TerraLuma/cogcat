//! GeoTIFF georeferencing: the affine transform and the CRS from GeoKeys.

/// Full-resolution pixel (col, row) to CRS (x, y), GDAL's order:
/// `x = c[0] + c[1] * col + c[2] * row`, `y = c[3] + c[4] * col + c[5] * row`.
/// Pixel (0, 0) is the top-left corner of the top-left pixel (PixelIsArea).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GeoTransform(pub [f64; 6]);

impl GeoTransform {
    pub fn pixel_to_crs(&self, col: f64, row: f64) -> [f64; 2] {
        let c = &self.0;
        [
            c[0] + c[1] * col + c[2] * row,
            c[3] + c[4] * col + c[5] * row,
        ]
    }

    /// The inverse, or `None` if the transform is degenerate.
    pub fn crs_to_pixel(&self, x: f64, y: f64) -> Option<[f64; 2]> {
        let c = &self.0;
        let det = c[1] * c[5] - c[2] * c[4];
        if det == 0.0 || !det.is_finite() {
            return None;
        }
        let (dx, dy) = (x - c[0], y - c[3]);
        Some([(c[5] * dx - c[2] * dy) / det, (c[1] * dy - c[4] * dx) / det])
    }
}

/// What the GeoKeys say about the coordinate reference system.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Crs {
    /// `ProjectedCSTypeGeoKey`, or `GeographicTypeGeoKey` for a geographic CRS.
    pub epsg: Option<u32>,
    pub geographic: bool,
    /// `GTCitationGeoKey` / `PCSCitationGeoKey`, e.g. "WGS 84 / UTM zone 55S".
    pub citation: Option<String>,
    /// Linear units name, e.g. "metre", from `ProjLinearUnitsGeoKey`.
    pub units: Option<String>,
}

impl Crs {
    /// e.g. "EPSG:32755 (WGS 84 / UTM zone 55S)".
    pub fn label(&self) -> String {
        match (self.epsg, &self.citation) {
            (Some(code), Some(name)) => format!("EPSG:{code} ({name})"),
            (Some(code), None) => format!("EPSG:{code}"),
            (None, Some(name)) => name.clone(),
            (None, None) => "unknown CRS".into(),
        }
    }
}

// GeoKey IDs.
const GT_MODEL_TYPE: u16 = 1024;
const GT_RASTER_TYPE: u16 = 1025;
const GT_CITATION: u16 = 1026;
const GEOGRAPHIC_TYPE: u16 = 2048;
const GEOG_CITATION: u16 = 2049;
const GEOG_ANGULAR_UNITS: u16 = 2054;
const PROJECTED_CS_TYPE: u16 = 3072;
const PCS_CITATION: u16 = 3073;
const PROJ_LINEAR_UNITS: u16 = 3076;

/// GeoKey value meaning "user-defined", i.e. no EPSG code.
const USER_DEFINED: u16 = 32767;

/// The GeoTIFF tags that georeference an image, as read from the IFD.
#[derive(Clone, Debug, Default)]
pub struct GeoTags {
    pub pixel_scale: Option<Vec<f64>>,
    pub tiepoints: Option<Vec<f64>>,
    pub transformation: Option<Vec<f64>>,
    pub key_directory: Option<Vec<u16>>,
    pub double_params: Option<Vec<f64>>,
    pub ascii_params: Option<String>,
}

/// One GeoKey's value.
#[derive(Clone, Debug, PartialEq)]
enum KeyValue {
    Short(u16),
    Doubles(Vec<f64>),
    Ascii(String),
}

/// Parses the `GeoKeyDirectory` into (key id, value) pairs. Malformed
/// entries are skipped.
fn geo_keys(tags: &GeoTags) -> Vec<(u16, KeyValue)> {
    let Some(dir) = tags.key_directory.as_deref() else {
        return Vec::new();
    };
    if dir.len() < 4 {
        return Vec::new();
    }
    let count = dir[3] as usize;
    let mut keys = Vec::new();
    for entry in dir[4..].as_chunks::<4>().0.iter().take(count) {
        let (id, location, n, index) = (entry[0], entry[1], entry[2] as usize, entry[3] as usize);
        let value = match location {
            0 => Some(KeyValue::Short(index as u16)),
            34736 => tags
                .double_params
                .as_ref()
                .and_then(|d| d.get(index..index + n))
                .map(|d| KeyValue::Doubles(d.to_vec())),
            34737 => tags
                .ascii_params
                .as_ref()
                .and_then(|a| a.get(index..index + n))
                .map(|a| KeyValue::Ascii(a.trim_end_matches(['|', '\0']).to_string())),
            34735 => dir.get(index).map(|&v| KeyValue::Short(v)),
            _ => None,
        };
        if let Some(value) = value {
            keys.push((id, value));
        }
    }
    keys
}

/// Whether the GeoKeys say `RasterPixelIsPoint`.
pub fn pixel_is_point(tags: &GeoTags) -> bool {
    geo_keys(tags)
        .iter()
        .any(|(id, v)| *id == GT_RASTER_TYPE && *v == KeyValue::Short(2))
}

/// The affine transform, from `ModelTransformation` or from `ModelPixelScale`
/// and one `ModelTiepoint`. With `RasterPixelIsPoint` the transform is
/// shifted by half a pixel so it describes pixel corners, as GDAL does.
pub fn geo_transform(tags: &GeoTags) -> Option<GeoTransform> {
    let mut gt = if let Some(m) = tags.transformation.as_deref().filter(|m| m.len() >= 16) {
        [m[3], m[0], m[1], m[7], m[4], m[5]]
    } else {
        let scale = tags.pixel_scale.as_deref().filter(|s| s.len() >= 2)?;
        let tie = tags.tiepoints.as_deref().filter(|t| t.len() >= 6)?;
        // Several tiepoints are ground control points, not an affine transform.
        if tie.len() > 6 && scale.iter().all(|&s| s == 0.0) {
            return None;
        }
        let (i, j, x, y) = (tie[0], tie[1], tie[3], tie[4]);
        let (sx, sy) = (scale[0], scale[1]);
        [x - i * sx, sx, 0.0, y + j * sy, 0.0, -sy]
    };
    if gt.iter().any(|v| !v.is_finite()) || (gt[1] == 0.0 && gt[2] == 0.0) {
        return None;
    }
    if pixel_is_point(tags) {
        gt[0] -= 0.5 * (gt[1] + gt[2]);
        gt[3] -= 0.5 * (gt[4] + gt[5]);
    }
    Some(GeoTransform(gt))
}

/// The CRS the GeoKeys describe.
pub fn crs(tags: &GeoTags) -> Crs {
    let keys = geo_keys(tags);
    let short = |key: u16| {
        keys.iter().find_map(|(id, v)| match v {
            KeyValue::Short(s) if *id == key => Some(*s),
            _ => None,
        })
    };
    let ascii = |key: u16| {
        keys.iter().find_map(|(id, v)| match v {
            KeyValue::Ascii(s) if *id == key && !s.is_empty() => Some(s.clone()),
            _ => None,
        })
    };
    let code = |key: u16| short(key).filter(|&c| c != 0 && c != USER_DEFINED);
    let projected_code = code(PROJECTED_CS_TYPE);
    let geographic_code = code(GEOGRAPHIC_TYPE);
    let geographic = match short(GT_MODEL_TYPE) {
        Some(2) => true,
        Some(1) => false,
        _ => projected_code.is_none() && geographic_code.is_some(),
    };
    let epsg = if geographic {
        geographic_code
    } else {
        projected_code.or(geographic_code.filter(|_| short(GT_MODEL_TYPE).is_none()))
    };
    let citation = if geographic {
        ascii(GEOG_CITATION).or_else(|| ascii(GT_CITATION))
    } else {
        ascii(PCS_CITATION).or_else(|| ascii(GT_CITATION))
    }
    .map(|c| tidy_citation(&c))
    .or_else(|| epsg.and_then(|code| well_known_name(code as u32)));
    let units = if geographic {
        short(GEOG_ANGULAR_UNITS)
            .and_then(unit_name)
            .or(Some("degree"))
    } else {
        short(PROJ_LINEAR_UNITS)
            .and_then(unit_name)
            .or(epsg.map(|_| "metre").filter(|_| is_metric_epsg(epsg)))
    };
    Crs {
        epsg: epsg.map(u32::from),
        geographic,
        citation,
        units: units.map(String::from),
    }
}

/// Some writers put ESRI-style "PCS Name = ..." citations; keep the name.
fn tidy_citation(citation: &str) -> String {
    let citation = citation.trim();
    match citation.strip_prefix("PCS Name = ") {
        Some(name) => name.split('|').next().unwrap_or(name).trim().to_string(),
        None => citation.to_string(),
    }
}

/// EPSG unit-of-measure codes.
fn unit_name(code: u16) -> Option<&'static str> {
    Some(match code {
        9001 => "metre",
        9002 => "foot",
        9003 => "US survey foot",
        9036 => "kilometre",
        9101 => "radian",
        9102 => "degree",
        9105 => "grad",
        9122 => "degree",
        _ => return None,
    })
}

/// Whether an EPSG code is one of the common metric projected CRSs (UTM,
/// web mercator), whose units can be assumed when the file doesn't say.
fn is_metric_epsg(epsg: Option<u16>) -> bool {
    matches!(
        epsg,
        Some(32601..=32660 | 32701..=32760 | 3857 | 3577 | 28348..=28358 | 7846..=7859)
    )
}

/// Names for a few CRSs that files often give no citation for.
fn well_known_name(epsg: u32) -> Option<String> {
    Some(match epsg {
        4326 => "WGS 84".into(),
        4283 => "GDA94".into(),
        7844 => "GDA2020".into(),
        3857 => "WGS 84 / Pseudo-Mercator".into(),
        32601..=32660 => format!("WGS 84 / UTM zone {}N", epsg - 32600),
        32701..=32760 => format!("WGS 84 / UTM zone {}S", epsg - 32700),
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn utm_tags() -> GeoTags {
        GeoTags {
            pixel_scale: Some(vec![10.0, 10.0, 0.0]),
            tiepoints: Some(vec![0.0, 0.0, 0.0, 500000.0, 5270000.0, 0.0]),
            key_directory: Some(vec![
                1, 1, 0, 4, //
                1024, 0, 1, 1, //
                1025, 0, 1, 1, //
                1026, 34737, 22, 0, //
                3072, 0, 1, 32755,
            ]),
            ascii_params: Some("WGS 84 / UTM zone 55S|".into()),
            ..Default::default()
        }
    }

    #[test]
    fn geo_transform_should_come_from_scale_and_tiepoint() {
        let gt = geo_transform(&utm_tags()).unwrap();
        assert_eq!(gt.0, [500000.0, 10.0, 0.0, 5270000.0, 0.0, -10.0]);
    }

    #[test]
    fn geo_transform_should_shift_pixel_is_point_by_half_a_pixel() {
        let mut tags = utm_tags();
        tags.key_directory.as_mut().unwrap()[11] = 2;
        let gt = geo_transform(&tags).unwrap();
        assert_eq!(gt.0, [499995.0, 10.0, 0.0, 5270005.0, 0.0, -10.0]);
    }

    #[test]
    fn geo_transform_should_prefer_model_transformation() {
        let mut tags = utm_tags();
        let mut m = vec![0.0; 16];
        (m[0], m[1], m[3]) = (8.0, -5.0, 100.0);
        (m[4], m[5], m[7]) = (-5.0, -8.0, 200.0);
        m[15] = 1.0;
        tags.transformation = Some(m);
        let gt = geo_transform(&tags).unwrap();
        assert_eq!(gt.0, [100.0, 8.0, -5.0, 200.0, -5.0, -8.0]);
    }

    #[test]
    fn crs_to_pixel_should_invert_pixel_to_crs() {
        let gt = GeoTransform([100.0, 8.0, -5.0, 200.0, -5.0, -8.0]);
        let [x, y] = gt.pixel_to_crs(12.5, -3.25);
        let [col, row] = gt.crs_to_pixel(x, y).unwrap();
        assert!((col - 12.5).abs() < 1e-9 && (row + 3.25).abs() < 1e-9);
        assert_eq!(
            GeoTransform([0.0, 1.0, 1.0, 0.0, 1.0, 1.0]).crs_to_pixel(1.0, 1.0),
            None
        );
    }

    #[test]
    fn crs_should_read_projected_code_and_citation() {
        let crs = crs(&utm_tags());
        assert_eq!(crs.epsg, Some(32755));
        assert!(!crs.geographic);
        assert_eq!(crs.label(), "EPSG:32755 (WGS 84 / UTM zone 55S)");
        assert_eq!(crs.units.as_deref(), Some("metre"));
    }

    #[test]
    fn crs_should_ignore_user_defined_codes() {
        let tags = GeoTags {
            key_directory: Some(vec![1, 1, 0, 2, 1024, 0, 1, 1, 3072, 0, 1, 32767]),
            ..Default::default()
        };
        let crs = crs(&tags);
        assert_eq!(crs.epsg, None);
        assert_eq!(crs.label(), "unknown CRS");
    }

    #[test]
    fn crs_should_read_geographic_codes() {
        let tags = GeoTags {
            key_directory: Some(vec![1, 1, 0, 2, 1024, 0, 1, 2, 2048, 0, 1, 4326]),
            ..Default::default()
        };
        let crs = crs(&tags);
        assert_eq!((crs.epsg, crs.geographic), (Some(4326), true));
        assert_eq!(crs.label(), "EPSG:4326 (WGS 84)");
        assert_eq!(crs.units.as_deref(), Some("degree"));
    }
}
