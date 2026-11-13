//! Colour maps for single-band display.

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Colormap {
    #[default]
    Gray,
    Viridis,
    Magma,
    Turbo,
    Terrain,
}

impl Colormap {
    pub const ALL: [Colormap; 5] = [
        Colormap::Gray,
        Colormap::Viridis,
        Colormap::Magma,
        Colormap::Turbo,
        Colormap::Terrain,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Colormap::Gray => "Gray",
            Colormap::Viridis => "Viridis",
            Colormap::Magma => "Magma",
            Colormap::Turbo => "Turbo",
            Colormap::Terrain => "Terrain",
        }
    }

    /// 256 straight RGBA8 entries, 0 to 1.
    pub fn lut(self) -> [[u8; 4]; 256] {
        std::array::from_fn(|i| {
            let t = i as f32 / 255.0;
            let [r, g, b] = match self {
                Colormap::Gray => [t; 3],
                Colormap::Viridis => ramp(&VIRIDIS, t),
                Colormap::Magma => ramp(&MAGMA, t),
                Colormap::Turbo => turbo(t),
                Colormap::Terrain => ramp(&TERRAIN, t),
            };
            let byte = |c: f32| (c.clamp(0.0, 1.0) * 255.0).round() as u8;
            [byte(r), byte(g), byte(b), 255]
        })
    }
}

/// Matplotlib's viridis at 0, 0.1, ..., 1.
const VIRIDIS: [u32; 11] = [
    0x440154, 0x482475, 0x414487, 0x355f8d, 0x2a788e, 0x21918c, 0x22a884, 0x44bf70, 0x7ad151,
    0xbddf26, 0xfde725,
];

/// Matplotlib's magma at 0, 0.1, ..., 1.
const MAGMA: [u32; 11] = [
    0x000004, 0x140e36, 0x3b0f70, 0x641a80, 0x8c2981, 0xb73779, 0xde4968, 0xf7705c, 0xfe9f6d,
    0xfecf92, 0xfcfdbf,
];

/// Lowland green, through tan and brown, to snow: for elevation models that
/// are mostly land, unlike matplotlib's terrain, which starts in sea blue.
const TERRAIN: [u32; 6] = [0x1a6634, 0x4d9e3a, 0xc8c06a, 0x8c6239, 0xb0a090, 0xffffff];

/// `stops` (0xRRGGBB, evenly spaced from 0 to 1) at `t`, interpolated linearly.
fn ramp(stops: &[u32], t: f32) -> [f32; 3] {
    let channels = |hex: u32| [hex >> 16, hex >> 8, hex].map(|c| (c & 0xff) as f32 / 255.0);
    let position = t.clamp(0.0, 1.0) * (stops.len() - 1) as f32;
    let i = (position as usize).min(stops.len() - 2);
    let fraction = position - i as f32;
    let (a, b) = (channels(stops[i]), channels(stops[i + 1]));
    std::array::from_fn(|c| a[c] + (b[c] - a[c]) * fraction)
}

/// Google's polynomial approximation of Turbo
/// (<https://gist.github.com/mikhailov-work/0d177465a8151eb6ede1768d51d476c7>).
fn turbo(t: f32) -> [f32; 3] {
    let t = t.clamp(0.0, 1.0);
    let polynomial =
        |c: [f32; 6]| c[0] + t * (c[1] + t * (c[2] + t * (c[3] + t * (c[4] + t * c[5]))));
    [
        polynomial([
            0.135_721_38,
            4.615_392_6,
            -42.660_32,
            132.131_08,
            -152.942_39,
            59.286_38,
        ]),
        polynomial([
            0.091_402_61,
            2.194_188_4,
            4.842_966_6,
            -14.185_033,
            4.277_298_6,
            2.829_566,
        ]),
        polynomial([
            0.106_673_3,
            12.641_946,
            -60.582_05,
            110.362_77,
            -89.903_11,
            27.348_25,
        ]),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gray_should_ramp_from_black_to_white() {
        let lut = Colormap::Gray.lut();
        assert_eq!(lut[0], [0, 0, 0, 255]);
        assert_eq!(lut[128], [128, 128, 128, 255]);
        assert_eq!(lut[255], [255, 255, 255, 255]);
    }

    #[test]
    fn viridis_should_match_matplotlib_at_its_ends_and_middle() {
        let lut = Colormap::Viridis.lut();
        assert_eq!(lut[0], [0x44, 0x01, 0x54, 255]);
        assert_eq!(lut[255], [0xfd, 0xe7, 0x25, 255]);
        // Matplotlib's viridis(0.5) is #21918c.
        let middle = lut[128];
        let expected = [0x21, 0x91, 0x8c];
        assert!(
            (0..3).all(|c| middle[c].abs_diff(expected[c]) <= 3),
            "{middle:?}"
        );
    }

    #[test]
    fn magma_should_run_from_black_to_pale_yellow() {
        let lut = Colormap::Magma.lut();
        assert_eq!(lut[0], [0x00, 0x00, 0x04, 255]);
        assert_eq!(lut[255], [0xfc, 0xfd, 0xbf, 255]);
    }

    #[test]
    fn turbo_should_run_from_dark_blue_through_green_to_dark_red() {
        let lut = Colormap::Turbo.lut();
        // The polynomial starts near black and turns blue at once.
        let [r, g, b, _] = lut[25];
        assert!(b > r && b > g, "start {:?}", lut[25]);
        let [r, g, b, _] = lut[128];
        assert!(g > r && g > b, "middle {:?}", lut[128]);
        let [r, g, b, _] = lut[255];
        assert!(r > g && r > b && r < 160, "end {:?}", lut[255]);
    }

    #[test]
    fn every_map_should_be_opaque_and_change_smoothly() {
        for colormap in Colormap::ALL {
            let lut = colormap.lut();
            assert!(lut.iter().all(|entry| entry[3] == 255));
            for pair in lut.windows(2) {
                let step = (0..3)
                    .map(|c| pair[0][c].abs_diff(pair[1][c]))
                    .max()
                    .unwrap();
                // Turbo's blue rises steepest, by about 13 per entry.
                assert!(step <= 16, "{} jumps by {step}", colormap.name());
            }
        }
    }
}
