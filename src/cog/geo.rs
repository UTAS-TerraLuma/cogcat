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
