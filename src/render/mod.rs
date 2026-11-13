//! Drawing tiles with wgpu, and statistics on the GPU.
//!
//! The GPU holds one `R32Float` texture per (tile, band), for the bands the
//! current composite needs; the CPU cache in the loader holds every band.
//! Tiles are drawn into an offscreen texture that egui shows, as in Copcat.
//!
//! Each tile is one quad, drawn from its four corners in clip space (so it may
//! be rotated or sheared), reading its band textures texel by texel with
//! `textureLoad`: nearest neighbour, which keeps pixels crisp when zoomed in
//! and needs no sampler (`R32Float` isn't filterable everywhere). The stretch,
//! gamma and colour map run in the fragment shader (`tiles.wgsl`).

mod colormap;
mod stats;

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use eframe::egui;
use eframe::egui::mutex::RwLock;
use eframe::egui_wgpu::{self, wgpu};

pub use colormap::Colormap;
pub use stats::{StatsRequest, StatsResult};

use crate::cog::TileKey;

const COLOR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
const BAND_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R32Float;
const VERTICES_PER_TILE: u32 = 6;

/// How to turn band values into colours.
#[derive(Clone, Debug, PartialEq)]
pub struct DisplayStyle {
    /// The band for red, green and blue. With `colormap`, only `bands[0]` is used.
    pub bands: [u32; 3],
    /// Draw `bands[0]` alone through this colour map.
    pub colormap: Option<Colormap>,
    /// Per channel, the value that maps to 0 and to 1.
    pub low: [f32; 3],
    pub high: [f32; 3],
    /// Applied after stretching: `out = stretched ^ (1 / gamma)`.
    pub gamma: f32,
    /// Shown where no tile is drawn, and under transparent (NaN) pixels.
    /// Straight RGBA, 0–1, as written to the 8-bit target.
    pub background: [f32; 4],
}

/// One tile to draw this frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TileDraw {
    pub key: TileKey,
    /// The tile's valid area (`TileData::valid`) on screen, in clip space
    /// (x right, y up, −1 to 1): top-left, top-right, bottom-right,
    /// bottom-left of the image region. Not necessarily axis aligned.
    pub corners: [[f32; 2]; 4],
    /// The valid area's size in texels, which the corners map to.
    pub texels: [f32; 2],
    /// Blended over the tile's colour by its alpha (debug "tint by level").
    pub tint: [f32; 4],
}

/// An image read back from the GPU, straight RGBA8, rows top to bottom.
pub struct Image {
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
}

/// A GPU device with no window, for tests and offscreen rendering. Takes any
/// adapter, including a software one (such as Mesa's lavapipe), with the
/// adapter's own limits, so images can be as big as it can draw.
pub fn headless_device() -> Result<(wgpu::Device, wgpu::Queue), String> {
    let instance =
        wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
    let request = |force_fallback_adapter| {
        pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter,
            ..Default::default()
        }))
    };
    let adapter = request(false)
        .or_else(|_| request(true))
        .map_err(|e| format!("couldn't find a GPU: {e}"))?;
    pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("cogcat headless"),
        required_limits: adapter.limits(),
        ..Default::default()
    }))
    .map_err(|e| format!("couldn't start the GPU: {e}"))
}

/// An offscreen target for one view, and its egui texture.
///
/// Each view keeps its own uniforms and vertices: `queue.write_buffer` only
/// lands at the next submit, so views sharing them would rely on each
/// `render` submitting before the next one writes.
#[derive(Default)]
pub struct Viewport {
    /// Created on the first `render`, and again when the size changes.
    target: Option<RenderTarget>,
    /// The target's colour texture as egui knows it. Kept across rebuilt
    /// targets, so the id egui is drawing stays valid.
    texture_id: Option<egui::TextureId>,
    /// Created on the first `render`.
    buffers: Option<ViewportBuffers>,
}

/// The texture we draw into, recreated whenever the viewport size changes.
struct RenderTarget {
    size: [u32; 2],
    /// Kept, beside its view, to copy from in `capture`.
    color_texture: wgpu::Texture,
    color: wgpu::TextureView,
    /// Whether `Viewport::texture_id` points at this target's colour yet.
    registered: bool,
}

struct ViewportBuffers {
    uniforms: wgpu::Buffer,
    /// The uniforms and the colour map.
    bind_group: wgpu::BindGroup,
    /// Every drawn tile's quad, reused across frames and grown when needed.
    vertices: wgpu::Buffer,
    /// In vertices.
    capacity: usize,
}

/// Must match `Uniforms` in `tiles.wgsl` byte for byte.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct TileUniforms {
    low: [f32; 4],
    high: [f32; 4],
    background: [f32; 4],
    inverse_gamma: f32,
    /// A WGSL `u32` used as a bool, since `bool` can't go in a uniform.
    use_colormap: u32,
    _padding: [u32; 2],
}

/// Must match `VertexIn` in `tiles.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct TileVertex {
    /// Clip space.
    position: [f32; 2],
    /// Texels from the valid area's top-left corner.
    texcoord: [f32; 2],
    tint: [f32; 4],
}

/// One band of one tile on the GPU.
struct BandTexture {
    /// Kept so bind groups for statistics can be made from it.
    view: wgpu::TextureView,
    size: [u32; 2],
    /// The `frame` it was last drawn (or uploaded), for evicting the least
    /// recently used first.
    last_drawn: u64,
}

impl BandTexture {
    fn bytes(&self) -> u64 {
        u64::from(self.size[0]) * u64::from(self.size[1]) * 4
    }
}

pub struct TileRenderer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    /// Shows render targets as egui textures. `None` without a window.
    egui_renderer: Option<Arc<RwLock<egui_wgpu::Renderer>>>,
    pipeline: wgpu::RenderPipeline,
    /// Group 0: uniforms and the colour map.
    style_layout: wgpu::BindGroupLayout,
    /// Group 1: a tile's red, green and blue band textures.
    bands_layout: wgpu::BindGroupLayout,
    /// 256 × 1 RGBA8, the colour map in use.
    colormap_texture: wgpu::Texture,
    colormap_view: wgpu::TextureView,
    /// What `colormap_texture` holds, so it's only rewritten on a change.
    colormap: Option<Colormap>,
    textures: HashMap<(TileKey, u32), BandTexture>,
    /// Bind groups for each tile and band triple drawn, made once and kept
    /// until one of the tile's textures is replaced or freed.
    bind_groups: HashMap<(TileKey, [u32; 3]), wgpu::BindGroup>,
    resident_bytes: u64,
    /// Counts draws, to find the least recently drawn textures.
    frame: u64,
    stats: stats::Statistics,
    /// Reused each frame, to avoid allocating.
    scratch_vertices: Vec<TileVertex>,
    scratch_drawn: Vec<TileKey>,
}

impl TileRenderer {
    /// A renderer for the viewer window, whose images egui can show.
    pub fn new(render_state: &egui_wgpu::RenderState) -> Self {
        Self::with_device(
            render_state.device.clone(),
            render_state.queue.clone(),
            Some(render_state.renderer.clone()),
        )
    }

    /// A renderer without a window, for tests and offscreen use. Its `render`
    /// can't give a real egui texture; use `capture` to get images.
    pub fn headless(device: wgpu::Device, queue: wgpu::Queue) -> Self {
        Self::with_device(device, queue, None)
    }

    fn with_device(
        device: wgpu::Device,
        queue: wgpu::Queue,
        egui_renderer: Option<Arc<RwLock<egui_wgpu::Renderer>>>,
    ) -> Self {
        let texture_entry = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };
        let style_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("tile_style_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                texture_entry(1),
            ],
        });
        let bands_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("tile_bands_layout"),
            entries: &[texture_entry(0), texture_entry(1), texture_entry(2)],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("tile_pipeline_layout"),
            bind_group_layouts: &[Some(&style_layout), Some(&bands_layout)],
            immediate_size: 0,
        });
        let shader = device.create_shader_module(wgpu::include_wgsl!("tiles.wgsl"));
        let vertex_attributes =
            wgpu::vertex_attr_array![0 => Float32x2, 1 => Float32x2, 2 => Float32x4];
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("tile_pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[Some(wgpu::VertexBufferLayout {
                    array_stride: size_of::<TileVertex>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &vertex_attributes,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            // No culling: a flipped geotransform mirrors the quad.
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                // Opaque: later tiles replace earlier ones, and NaN pixels
                // are discarded rather than blended.
                targets: &[Some(wgpu::ColorTargetState {
                    format: COLOR_FORMAT,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            multiview_mask: None,
            cache: None,
        });

        let colormap_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("colormap"),
            size: wgpu::Extent3d {
                width: 256,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let colormap_view = colormap_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let stats = stats::Statistics::new(&device);
        let mut renderer = Self {
            device,
            queue,
            egui_renderer,
            pipeline,
            style_layout,
            bands_layout,
            colormap_texture,
            colormap_view,
            colormap: None,
            textures: HashMap::new(),
            bind_groups: HashMap::new(),
            resident_bytes: 0,
            frame: 0,
            stats,
            scratch_vertices: Vec::new(),
            scratch_drawn: Vec::new(),
        };
        // Never leave the colour map texture undefined.
        renderer.set_colormap(Colormap::Gray);
        renderer
    }

    /// The largest image `capture` can draw, each way.
    pub fn max_image_size(&self) -> u32 {
        self.device.limits().max_texture_dimension_2d
    }

    pub fn has_band(&self, key: TileKey, band: u32) -> bool {
        self.textures.contains_key(&(key, band))
    }

    /// How many (tile, band) textures are on the GPU.
    pub fn resident_textures(&self) -> usize {
        self.textures.len()
    }

    /// Uploads one band of a tile: `size` texels, row-major, NaN for no data.
    /// Replaces any texture already there.
    ///
    /// Panics if `data` doesn't hold exactly `size[0] × size[1]` values, or a
    /// side is 0 or bigger than the GPU allows.
    pub fn upload_band(&mut self, key: TileKey, band: u32, size: [u32; 2], data: &[f32]) {
        let [width, height] = size;
        assert_eq!(
            data.len() as u64,
            u64::from(width) * u64::from(height),
            "band {band} of tile {key} should have {width} × {height} values"
        );
        let extent = wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };
        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("tile_band"),
            size: extent,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: BAND_FORMAT,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        self.queue.write_texture(
            texture.as_image_copy(),
            bytemuck::cast_slice(data),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(width * 4),
                rows_per_image: None,
            },
            extent,
        );
        let band_texture = BandTexture {
            view: texture.create_view(&wgpu::TextureViewDescriptor::default()),
            size,
            // Count as just drawn, so it isn't evicted before its first frame.
            last_drawn: self.frame,
        };
        self.resident_bytes += band_texture.bytes();
        if let Some(previous) = self.textures.insert((key, band), band_texture) {
            self.resident_bytes -= previous.bytes();
            self.forget_bind_groups(key);
        }
    }

    /// Bytes of tile textures on the GPU.
    pub fn resident_bytes(&self) -> u64 {
        self.resident_bytes
    }

    /// Frees least recently drawn textures until at most `max_bytes` remain,
    /// never those of tiles in `keep`. Returns what was freed.
    pub fn evict_to_budget(
        &mut self,
        max_bytes: u64,
        keep: &HashSet<TileKey>,
    ) -> Vec<(TileKey, u32)> {
        if self.resident_bytes <= max_bytes {
            return Vec::new();
        }
        let mut candidates: Vec<(u64, TileKey, u32)> = self
            .textures
            .iter()
            .filter(|((key, _), _)| !keep.contains(key))
            .map(|(&(key, band), texture)| (texture.last_drawn, key, band))
            .collect();
        candidates.sort_unstable();

        let mut evicted = Vec::new();
        for (_, key, band) in candidates {
            if self.resident_bytes <= max_bytes {
                break;
            }
            if let Some(texture) = self.textures.remove(&(key, band)) {
                self.resident_bytes -= texture.bytes();
                self.forget_bind_groups(key);
                evicted.push((key, band));
            }
        }
        evicted
    }

    /// Frees every tile texture (when another file opens).
    pub fn clear(&mut self) {
        self.textures.clear();
        self.bind_groups.clear();
        self.resident_bytes = 0;
    }

    /// Drops the bind groups that use any of `key`'s textures, which would
    /// otherwise keep a replaced or evicted texture alive.
    fn forget_bind_groups(&mut self, key: TileKey) {
        self.bind_groups.retain(|(tile, _), _| *tile != key);
    }

    /// Draws `tiles` in order (later ones on top) into `viewport` at `size`
    /// physical pixels, skipping any whose bands for `style` aren't all
    /// uploaded. Returns the texture for egui to show.
    ///
    /// A headless renderer has no egui to show it in, so it returns a
    /// placeholder id; `capture` reads such images back.
    pub fn render(
        &mut self,
        viewport: &mut Viewport,
        size: [u32; 2],
        tiles: &[TileDraw],
        style: &DisplayStyle,
    ) -> egui::TextureId {
        let max = self.max_image_size();
        let size = size.map(|side| side.clamp(1, max));
        self.draw(viewport, size, tiles, style);
        self.register_texture(viewport)
    }

    /// Draws like `render`, then reads the image back, waiting for the GPU.
    /// Errors if `size` is 0 or bigger than `max_image_size` either way.
    pub fn capture(
        &mut self,
        viewport: &mut Viewport,
        size: [u32; 2],
        tiles: &[TileDraw],
        style: &DisplayStyle,
    ) -> Result<Image, String> {
        let max = self.max_image_size();
        if size.iter().any(|&side| side == 0 || side > max) {
            return Err(format!(
                "can't render {} × {} pixels: each side must be 1 to {max}",
                size[0], size[1]
            ));
        }
        self.draw(viewport, size, tiles, style);
        let target = viewport.target.as_ref().expect("draw creates the target");
        self.read_back(&target.color_texture, size)
    }

    /// Frees `viewport`'s target and its egui texture, for a view that closes.
    pub fn free_viewport(&mut self, viewport: &mut Viewport) {
        if let (Some(id), Some(egui_renderer)) = (viewport.texture_id, &self.egui_renderer) {
            egui_renderer.write().free_texture(&id);
        }
        *viewport = Viewport::default();
    }

    /// Starts computing histograms on the GPU. Returns at once; the result
    /// comes from `poll_stats` a frame or more later. Returns the request id.
    ///
    /// Each rectangle is clamped to its texture. A device without compute
    /// shaders gives empty histograms.
    pub fn start_stats(&mut self, request: StatsRequest) -> u64 {
        let textures = &self.textures;
        self.stats
            .start(&self.device, &self.queue, request, |key, band| {
                textures
                    .get(&(key, band))
                    .map(|texture| (&texture.view, texture.size))
            })
    }

    /// Finished statistics, without blocking. Call once per frame.
    pub fn poll_stats(&mut self) -> Vec<StatsResult> {
        self.stats.poll(&self.device)
    }

    /// How many statistics requests haven't been returned by `poll_stats` yet.
    pub fn pending_stats(&self) -> usize {
        self.stats.pending()
    }

    /// Copies a colour texture into memory, dropping the row padding the copy needs.
    fn read_back(
        &self,
        texture: &wgpu::Texture,
        [width, height]: [u32; 2],
    ) -> Result<Image, String> {
        let row_bytes = width * 4;
        let padded_row_bytes = row_bytes.next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("capture"),
            size: u64::from(padded_row_bytes) * u64::from(height),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("capture_encoder"),
            });
        encoder.copy_texture_to_buffer(
            texture.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_row_bytes),
                    rows_per_image: None,
                },
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
        self.queue.submit([encoder.finish()]);

        let slice = buffer.slice(..);
        // The callback runs inside `poll`, which blocks until the copy is done.
        slice.map_async(wgpu::MapMode::Read, |_| {});
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| format!("the GPU failed while reading back the image: {e}"))?;
        let mapped = slice
            .get_mapped_range()
            .map_err(|e| format!("couldn't read back the image: {e}"))?;
        let rgba = mapped
            .chunks_exact(padded_row_bytes as usize)
            .flat_map(|row| &row[..row_bytes as usize])
            .copied()
            .collect();
        Ok(Image {
            width,
            height,
            rgba,
        })
    }

    /// Points `viewport.texture_id` at its target's colour texture, which
    /// changes whenever the target is rebuilt.
    fn register_texture(&self, viewport: &mut Viewport) -> egui::TextureId {
        let Some(egui_renderer) = self.egui_renderer.as_ref() else {
            return egui::TextureId::default();
        };
        let target = viewport.target.as_mut().expect("draw creates the target");
        if let Some(id) = viewport.texture_id
            && target.registered
        {
            return id;
        }
        // The texture maps 1:1 to screen pixels, so no filtering is needed.
        let mut egui_renderer = egui_renderer.write();
        let id = match viewport.texture_id {
            Some(id) => {
                egui_renderer.update_egui_texture_from_wgpu_texture(
                    &self.device,
                    &target.color,
                    wgpu::FilterMode::Nearest,
                    id,
                );
                id
            }
            None => egui_renderer.register_native_texture(
                &self.device,
                &target.color,
                wgpu::FilterMode::Nearest,
            ),
        };
        target.registered = true;
        viewport.texture_id = Some(id);
        id
    }

    /// Rewrites the colour map texture if it holds another map.
    fn set_colormap(&mut self, colormap: Colormap) {
        if self.colormap == Some(colormap) {
            return;
        }
        let lut = colormap.lut();
        self.queue.write_texture(
            self.colormap_texture.as_image_copy(),
            lut.as_flattened(),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(256 * 4),
                rows_per_image: None,
            },
            wgpu::Extent3d {
                width: 256,
                height: 1,
                depth_or_array_layers: 1,
            },
        );
        self.colormap = Some(colormap);
    }

    /// Draws the `tiles` whose bands are uploaded into `viewport`'s target,
    /// rebuilt at `size` if needed.
    fn draw(
        &mut self,
        viewport: &mut Viewport,
        size: [u32; 2],
        tiles: &[TileDraw],
        style: &DisplayStyle,
    ) {
        // Take the target out, reuse or rebuild it, then put it back below.
        let target = match viewport.target.take() {
            Some(target) if target.size == size => target,
            _ => self.create_target(size),
        };
        self.frame += 1;
        if let Some(colormap) = style.colormap {
            self.set_colormap(colormap);
        }
        // With a colour map, the one band goes in all three slots.
        let bands = match style.colormap {
            Some(_) => [style.bands[0]; 3],
            None => style.bands,
        };

        // Quads (and bind groups) for the drawable tiles.
        let mut vertices = std::mem::take(&mut self.scratch_vertices);
        let mut drawn = std::mem::take(&mut self.scratch_drawn);
        vertices.clear();
        drawn.clear();
        for tile in tiles {
            let mut views = [None; 3];
            for (view, band) in views.iter_mut().zip(bands) {
                *view = self.textures.get(&(tile.key, band)).map(|t| &t.view);
            }
            let [Some(r), Some(g), Some(b)] = views else {
                continue;
            };
            self.bind_groups
                .entry((tile.key, bands))
                .or_insert_with(|| {
                    self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("tile_bands"),
                        layout: &self.bands_layout,
                        entries: &[
                            texture_binding(0, r),
                            texture_binding(1, g),
                            texture_binding(2, b),
                        ],
                    })
                });
            for band in bands {
                if let Some(texture) = self.textures.get_mut(&(tile.key, band)) {
                    texture.last_drawn = self.frame;
                }
            }
            vertices.extend(quad(tile));
            drawn.push(tile.key);
        }

        let buffers = self.viewport_buffers(viewport, vertices.len());
        let uniforms = uniforms_for(style);
        self.queue
            .write_buffer(&buffers.uniforms, 0, bytemuck::bytes_of(&uniforms));
        if !vertices.is_empty() {
            self.queue
                .write_buffer(&buffers.vertices, 0, bytemuck::cast_slice(&vertices));
        }

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("tile_encoder"),
            });
        {
            let [r, g, b, a] = style.background.map(f64::from);
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("tile_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &target.color,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color { r, g, b, a }),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
                multiview_mask: None,
            });
            if !drawn.is_empty() {
                pass.set_pipeline(&self.pipeline);
                pass.set_bind_group(0, &buffers.bind_group, &[]);
                pass.set_vertex_buffer(0, buffers.vertices.slice(..));
                for (i, key) in drawn.iter().enumerate() {
                    let first = i as u32 * VERTICES_PER_TILE;
                    pass.set_bind_group(1, &self.bind_groups[&(*key, bands)], &[]);
                    pass.draw(first..first + VERTICES_PER_TILE, 0..1);
                }
            }
        }
        // egui submits its own work later on the same queue, so this draw is
        // guaranteed to finish before egui samples the texture.
        self.queue.submit([encoder.finish()]);
        viewport.target = Some(target);
        self.scratch_vertices = vertices;
        self.scratch_drawn = drawn;
    }

    /// `viewport`'s buffers, made on first use, with room for `vertex_count`.
    fn viewport_buffers<'v>(
        &self,
        viewport: &'v mut Viewport,
        vertex_count: usize,
    ) -> &'v ViewportBuffers {
        let device = &self.device;
        let buffers = viewport.buffers.get_or_insert_with(|| {
            let uniforms = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("tile_uniforms"),
                size: size_of::<TileUniforms>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("tile_style"),
                layout: &self.style_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: uniforms.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&self.colormap_view),
                    },
                ],
            });
            ViewportBuffers {
                uniforms,
                bind_group,
                vertices: create_vertex_buffer(device, 0),
                capacity: 0,
            }
        });
        if buffers.capacity < vertex_count {
            // Doubling avoids reallocating every frame while the count creeps up.
            let capacity = vertex_count.next_power_of_two().max(64);
            buffers.vertices = create_vertex_buffer(device, capacity);
            buffers.capacity = capacity;
        }
        buffers
    }

    fn create_target(&self, size: [u32; 2]) -> RenderTarget {
        let color_texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("tile_color"),
            size: wgpu::Extent3d {
                width: size[0],
                height: size[1],
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: COLOR_FORMAT,
            // TEXTURE_BINDING so egui can show it, COPY_SRC so `capture` can read it.
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let color = color_texture.create_view(&wgpu::TextureViewDescriptor::default());
        RenderTarget {
            size,
            color_texture,
            color,
            registered: false,
        }
    }
}

/// Two triangles covering a tile, each corner carrying its texel coordinate.
fn quad(tile: &TileDraw) -> [TileVertex; VERTICES_PER_TILE as usize] {
    let [w, h] = tile.texels;
    let texcoords = [[0.0, 0.0], [w, 0.0], [w, h], [0.0, h]];
    let corner = |i: usize| TileVertex {
        position: tile.corners[i],
        texcoord: texcoords[i],
        tint: tile.tint,
    };
    // Top-left, top-right, bottom-right; top-left, bottom-right, bottom-left.
    [0, 1, 2, 0, 2, 3].map(corner)
}

fn uniforms_for(style: &DisplayStyle) -> TileUniforms {
    let [r, g, b] = style.low;
    let [hr, hg, hb] = style.high;
    // A gamma of 0 (or less, or NaN) would be meaningless; draw it as 1.
    let gamma = if style.gamma > 0.0 { style.gamma } else { 1.0 };
    TileUniforms {
        low: [r, g, b, 0.0],
        high: [hr, hg, hb, 1.0],
        background: style.background,
        inverse_gamma: 1.0 / gamma,
        use_colormap: u32::from(style.colormap.is_some()),
        _padding: [0; 2],
    }
}

fn texture_binding(binding: u32, view: &wgpu::TextureView) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: wgpu::BindingResource::TextureView(view),
    }
}

fn create_vertex_buffer(device: &wgpu::Device, vertices: usize) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("tile_vertices"),
        // A zero-sized buffer can't be bound, so keep at least one vertex.
        size: (vertices.max(1) * size_of::<TileVertex>()) as u64,
        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quad_should_map_corners_to_texel_corners() {
        let tile = TileDraw {
            key: TileKey {
                level: 0,
                x: 0,
                y: 0,
            },
            corners: [[-1.0, 1.0], [1.0, 1.0], [1.0, -1.0], [-1.0, -1.0]],
            texels: [100.0, 50.0],
            tint: [0.0; 4],
        };
        let vertices = quad(&tile);
        let bottom_right = vertices[2];
        assert_eq!(bottom_right.position, [1.0, -1.0]);
        assert_eq!(bottom_right.texcoord, [100.0, 50.0]);
        assert_eq!(vertices[5].texcoord, [0.0, 50.0]);
    }

    #[test]
    fn uniforms_should_treat_a_bad_gamma_as_one() {
        let style = DisplayStyle {
            bands: [0, 1, 2],
            colormap: None,
            low: [0.0; 3],
            high: [1.0; 3],
            gamma: 0.0,
            background: [0.0; 4],
        };
        assert_eq!(uniforms_for(&style).inverse_gamma, 1.0);
    }
}
