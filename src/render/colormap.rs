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
        todo!("agent B")
    }
}
