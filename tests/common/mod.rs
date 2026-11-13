//! Shared by the GPU tests.

use cogcat::render::{TileRenderer, headless_device};
use eframe::egui_wgpu::wgpu;

/// Microsoft's PCI vendor id, and WARP's device id.
const WARP: (u32, u32) = (0x1414, 0x8c);

/// A headless renderer, or `None` (with a note) to skip the test: when there's
/// no GPU, or the adapter is WARP, Windows' CPU renderer on GitHub's runners,
/// which passes but takes minutes. Linux CI runs these tests on lavapipe. Set
/// `COGCAT_TEST_ON_WARP=1` to run them on WARP anyway.
pub fn renderer() -> Option<TileRenderer> {
    let (device, queue) = match headless_device() {
        Ok(device) => device,
        Err(error) => {
            eprintln!("skipping GPU test: {error}");
            return None;
        }
    };
    let info = device.adapter_info();
    let is_warp = info.backend == wgpu::Backend::Dx12 && (info.vendor, info.device) == WARP;
    if is_warp && std::env::var_os("COGCAT_TEST_ON_WARP").is_none() {
        eprintln!(
            "skipping GPU test on WARP ({}); set COGCAT_TEST_ON_WARP=1 to run it",
            info.name
        );
        return None;
    }
    Some(TileRenderer::headless(device, queue))
}
