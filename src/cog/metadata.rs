//! GDAL's metadata tags: `GDAL_METADATA` (an XML list of items, per dataset
//! or per band) and `GDAL_NODATA` (a number as text).

use super::BandInfo;

/// One `<Item>` of `GDAL_METADATA`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct MetadataItem {
    pub name: String,
    /// The zero-based band, or `None` for dataset-level items.
    pub sample: Option<u32>,
    /// e.g. `description`, `scale`, `offset`.
    pub role: Option<String>,
    /// e.g. `IMAGERY`, `ENVI`; `None` for the default domain.
    pub domain: Option<String>,
    pub value: String,
}

/// Parses `GDAL_NODATA`: a number, or `nan` / `inf` in any case.
pub fn parse_nodata(text: &str) -> Option<f64> {
    let text = text.trim_matches(|c: char| c.is_whitespace() || c == '\0');
    match text.to_ascii_lowercase().as_str() {
        "nan" | "-nan" => Some(f64::NAN),
        "inf" | "+inf" | "infinity" => Some(f64::INFINITY),
        "-inf" | "-infinity" => Some(f64::NEG_INFINITY),
        _ => text.parse().ok(),
    }
}

/// Parses every `<Item>` of `GDAL_METADATA`. This is a small hand-written
/// parser for the flat XML GDAL writes; anything it doesn't understand is
/// skipped.
pub fn parse_items(xml: &str) -> Vec<MetadataItem> {
    let mut items = Vec::new();
    let mut rest = xml;
    while let Some(start) = rest.find("<Item") {
        rest = &rest[start + "<Item".len()..];
        let Some(tag_end) = rest.find('>') else { break };
        let attributes = &rest[..tag_end];
        let self_closing = attributes.ends_with('/');
        rest = &rest[tag_end + 1..];
        let value = if self_closing {
            String::new()
        } else {
            let Some(close) = rest.find("</Item>") else {
                break;
            };
            let text = unescape(&rest[..close]);
            rest = &rest[close + "</Item>".len()..];
            text
        };
        let mut item = MetadataItem {
            value,
            ..Default::default()
        };
        for (key, val) in parse_attributes(attributes.trim_end_matches('/')) {
            match key {
                "name" => item.name = val,
                "sample" => item.sample = val.trim().parse().ok(),
                "role" => item.role = Some(val),
                "domain" => item.domain = Some(val).filter(|d| !d.is_empty()),
                _ => {}
            }
        }
        if !item.name.is_empty() || item.role.is_some() {
            items.push(item);
        }
    }
    items
}

/// `key="value"` pairs (single or double quotes), values unescaped.
fn parse_attributes(text: &str) -> Vec<(&str, String)> {
    let mut pairs = Vec::new();
    let mut rest = text;
    while let Some(eq) = rest.find('=') {
        let key = rest[..eq].trim();
        let after = rest[eq + 1..].trim_start();
        let Some(quote) = after.chars().next().filter(|c| *c == '"' || *c == '\'') else {
            break;
        };
        let Some(end) = after[1..].find(quote) else {
            break;
        };
        pairs.push((key, unescape(&after[1..1 + end])));
        rest = &after[1 + end + 1..];
    }
    pairs
}

/// Replaces XML entities: the five named ones and numeric references.
pub fn unescape(text: &str) -> String {
    if !text.contains('&') {
        return text.to_string();
    }
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(amp) = rest.find('&') {
        out.push_str(&rest[..amp]);
        rest = &rest[amp..];
        let replaced = rest.find(';').and_then(|semi| {
            let entity = &rest[1..semi];
            let c = match entity {
                "amp" => Some('&'),
                "lt" => Some('<'),
                "gt" => Some('>'),
                "quot" => Some('"'),
                "apos" => Some('\''),
                _ => {
                    let code = if let Some(hex) =
                        entity.strip_prefix("#x").or(entity.strip_prefix("#X"))
                    {
                        u32::from_str_radix(hex, 16).ok()
                    } else {
                        entity.strip_prefix('#').and_then(|d| d.parse().ok())
                    };
                    code.and_then(char::from_u32)
                }
            };
            c.map(|c| (c, semi))
        });
        match replaced {
            Some((c, semi)) => {
                out.push(c);
                rest = &rest[semi + 1..];
            }
            None => {
                out.push('&');
                rest = &rest[1..];
            }
        }
    }
    out.push_str(rest);
    out
}

/// What the metadata says about each of `bands` bands.
pub fn band_info(items: &[MetadataItem], bands: u32) -> Vec<BandInfo> {
    let envi_wavelengths = envi_wavelengths(items);
    (0..bands)
        .map(|band| {
            let band_items: Vec<&MetadataItem> =
                items.iter().filter(|i| i.sample == Some(band)).collect();
            let find = |name: &str| {
                band_items
                    .iter()
                    .find(|i| i.name.eq_ignore_ascii_case(name))
                    .map(|i| i.value.trim())
            };
            let role = |role: &str| {
                band_items
                    .iter()
                    .find(|i| i.role.as_deref() == Some(role))
                    .map(|i| i.value.trim())
            };
            let description = role("description")
                .or_else(|| find("DESCRIPTION"))
                .filter(|d| !d.is_empty())
                .map(String::from);
            let central_um = find("CENTRAL_WAVELENGTH_UM").and_then(|v| v.parse::<f64>().ok());
            let wavelength = find("wavelength").and_then(|v| v.parse::<f64>().ok());
            let wavelength_nm = central_um
                .map(|um| um * 1000.0)
                .or_else(|| wavelength.and_then(|w| to_nanometres(w, find("wavelength_units"))))
                .or_else(|| envi_wavelengths.get(band as usize).copied().flatten())
                .filter(|w| w.is_finite() && *w > 0.0);
            BandInfo {
                description,
                wavelength_nm,
                scale: role("scale").and_then(|v| v.parse().ok()),
                offset: role("offset").and_then(|v| v.parse().ok()),
            }
        })
        .collect()
}

/// ENVI-style dataset-level `wavelength = {a, b, ...}` (with
/// `wavelength_units`), converted to nanometres.
fn envi_wavelengths(items: &[MetadataItem]) -> Vec<Option<f64>> {
    let dataset = |name: &str| {
        items
            .iter()
            .filter(|i| i.sample.is_none())
            .filter(|i| {
                i.domain
                    .as_deref()
                    .is_none_or(|d| d.eq_ignore_ascii_case("ENVI"))
            })
            .find(|i| i.name.eq_ignore_ascii_case(name))
            .map(|i| i.value.as_str())
    };
    let Some(list) = dataset("wavelength") else {
        return Vec::new();
    };
    let units = dataset("wavelength_units");
    list.trim()
        .trim_start_matches('{')
        .trim_end_matches('}')
        .split(',')
        .map(|v| v.trim().parse().ok().and_then(|w| to_nanometres(w, units)))
        .collect()
}

/// Converts a wavelength to nanometres, guessing µm for values below 100
/// when the units are missing or unknown.
pub fn to_nanometres(value: f64, units: Option<&str>) -> Option<f64> {
    let units = units.map(|u| u.trim().to_lowercase());
    let micro = matches!(
        units.as_deref(),
        Some(
            "um" | "µm"
                | "μm"
                | "micron"
                | "microns"
                | "micrometer"
                | "micrometers"
                | "micrometre"
                | "micrometres"
        )
    );
    let nano = matches!(
        units.as_deref(),
        Some("nm" | "nanometer" | "nanometers" | "nanometre" | "nanometres")
    );
    if micro || (!nano && value < 100.0) {
        Some(value * 1000.0)
    } else {
        Some(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const XML: &str = r#"<GDALMetadata>
  <Item name="OVR_RESAMPLING">AVERAGE</Item>
  <Item name="DESCRIPTION" sample="0" role="description">B02 &amp; &lt;blue&gt;</Item>
  <Item name="wavelength" sample="0">0.49</Item>
  <Item name="wavelength_units" sample="0">micrometers</Item>
  <Item name="SCALE" sample="0" role="scale">0.0001</Item>
  <Item name="OFFSET" sample="0" role="offset">-0.1</Item>
  <Item name="wavelength" sample="1">560</Item>
  <Item name="wavelength_units" sample="1">nm</Item>
  <Item name="CENTRAL_WAVELENGTH_UM" sample="2" domain="IMAGERY">0.665</Item>
  <Item name="WAVELENGTH" sample="3">0.842</Item>
  <Item name="empty" sample='4'/>
</GDALMetadata>"#;

    #[test]
    fn parse_items_should_read_attributes_and_entities() {
        let items = parse_items(XML);
        assert_eq!(items.len(), 11);
        assert_eq!(
            items[1],
            MetadataItem {
                name: "DESCRIPTION".into(),
                sample: Some(0),
                role: Some("description".into()),
                domain: None,
                value: "B02 & <blue>".into(),
            }
        );
        assert_eq!(items[8].domain.as_deref(), Some("IMAGERY"));
        assert_eq!((items[10].sample, items[10].value.as_str()), (Some(4), ""));
    }

    #[test]
    fn band_info_should_read_descriptions_scales_and_wavelengths() {
        let info = band_info(&parse_items(XML), 5);
        assert_eq!(info[0].description.as_deref(), Some("B02 & <blue>"));
        assert_eq!(info[0].scale, Some(0.0001));
        assert_eq!(info[0].offset, Some(-0.1));
        assert_eq!(info[0].wavelength_nm, Some(490.0));
        assert_eq!(info[1].wavelength_nm, Some(560.0));
        assert_eq!(info[2].wavelength_nm, Some(665.0));
        assert_eq!(info[3].wavelength_nm, Some(842.0));
        assert_eq!(info[4], BandInfo::default());
    }

    #[test]
    fn band_info_should_read_envi_wavelength_lists() {
        let xml = r#"<GDALMetadata>
  <Item name="wavelength" domain="ENVI">{ 450.5, 550,650.25 }</Item>
  <Item name="wavelength_units" domain="ENVI">Nanometers</Item>
</GDALMetadata>"#;
        let info = band_info(&parse_items(xml), 4);
        let wavelengths: Vec<_> = info.iter().map(|b| b.wavelength_nm).collect();
        assert_eq!(wavelengths, [Some(450.5), Some(550.0), Some(650.25), None]);
    }

    #[test]
    fn unescape_should_handle_numeric_and_broken_entities() {
        assert_eq!(
            unescape("a &#181;m &#xB5;m &bogus; & b"),
            "a µm µm &bogus; & b"
        );
    }

    #[test]
    fn parse_nodata_should_accept_numbers_and_nan() {
        assert_eq!(parse_nodata("-9999"), Some(-9999.0));
        assert_eq!(parse_nodata("0\0"), Some(0.0));
        assert_eq!(parse_nodata(" 1e-3 "), Some(0.001));
        assert!(parse_nodata("nan").unwrap().is_nan());
        assert_eq!(parse_nodata("-inf"), Some(f64::NEG_INFINITY));
        assert_eq!(parse_nodata("none"), None);
    }

    #[test]
    fn to_nanometres_should_guess_micrometres_for_small_values() {
        assert_eq!(to_nanometres(0.665, None), Some(665.0));
        assert_eq!(to_nanometres(665.0, None), Some(665.0));
        assert_eq!(to_nanometres(1.6, Some("Micrometers")), Some(1600.0));
        assert_eq!(to_nanometres(90.0, Some("nm")), Some(90.0));
    }
}
