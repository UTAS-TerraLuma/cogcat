//! Histograms computed on the GPU: a min/max pass, then a histogram pass,
//! over chosen texel rectangles of uploaded (tile, band) textures.
//!
//! Both passes go in one submission, followed by a copy of the (small)
//! results into a buffer that's mapped without blocking; `poll` picks up
//! finished requests a frame or more later. See `stats.wgsl` for the passes.

use std::collections::HashMap;
use std::num::NonZeroU64;
use std::sync::Arc;
use std::sync::atomic::{AtomicU8, Ordering};

use eframe::egui_wgpu::wgpu;

use crate::cog::TileKey;
use crate::stretch::Histogram;

pub struct StatsRequest {
    /// The bands to compute, each giving one histogram.
    pub bands: Vec<u32>,
    /// Which texels of which tiles to include: `[x0, y0, x1, y1)` in texels.
    /// Every band in `bands` must be uploaded for these tiles; any that
    /// aren't are skipped.
    pub regions: Vec<(TileKey, [u32; 4])>,
    /// Histogram bins per band (e.g. 1024). Clamped to 1 to 4096.
    pub bins: u32,
}

pub struct StatsResult {
    pub id: u64,
    /// One per requested band, in order. `total` is 0 if no finite texel was seen.
    pub histograms: Vec<(u32, Histogram)>,
}

/// The most bins a histogram can have: the shader keeps a workgroup copy of
/// the counts, and 4096 `u32`s fill WebGPU's default 16 KiB of workgroup memory.
/// Must match `MAX_BINS` in `stats.wgsl`.
const MAX_BINS: u32 = 4096;

/// Texels per workgroup each way. Must match `BLOCK` in `stats.wgsl`.
const BLOCK: u32 = 64;

/// One dispatch's parameters. Must match `Params` in `stats.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct StatsParams {
    rect: [u32; 4],
    slot: u32,
    bins: u32,
    _padding: [u32; 2],
}

/// A readback buffer's state, set by its `map_async` callback.
const WAITING: u8 = 0;
const MAPPED: u8 = 1;
const FAILED: u8 = 2;

/// The compute pipelines and the requests in flight.
pub(super) struct Statistics {
    /// `None` when the device can't run them (no compute shaders, as on
    /// WebGL-level devices, or too little workgroup memory); every request
    /// then gives empty histograms.
    pipelines: Option<Pipelines>,
    pending: Vec<Pending>,
    /// Requests answered without the GPU, returned by the next `poll`.
    ready: Vec<StatsResult>,
    next_id: u64,
}

struct Pipelines {
    min_max: wgpu::ComputePipeline,
    histogram: wgpu::ComputePipeline,
    /// Parameters, min/max and counts.
    buffers_layout: wgpu::BindGroupLayout,
    /// The band texture of one dispatch.
    texture_layout: wgpu::BindGroupLayout,
    /// Bytes between dispatches' parameters, as dynamic offsets must be aligned.
    params_stride: u64,
}

struct Pending {
    id: u64,
    bands: Vec<u32>,
    bins: u32,
    /// Min/max keys (2 per band), then the counts (`bins` per band).
    readback: wgpu::Buffer,
    state: Arc<AtomicU8>,
}

/// One rectangle of one (tile, band) texture.
struct Job<'a> {
    view: &'a wgpu::TextureView,
    /// Which texture, to share its bind group between rectangles.
    texture: (TileKey, u32),
    params: StatsParams,
}

impl Statistics {
    pub(super) fn new(device: &wgpu::Device) -> Self {
        let limits = device.limits();
        let can_run = limits.max_compute_invocations_per_workgroup >= 256
            && limits.max_compute_workgroup_size_x >= 16
            && limits.max_compute_workgroup_size_y >= 16
            && limits.max_compute_workgroup_storage_size >= MAX_BINS * 4
            && limits.max_storage_buffers_per_shader_stage >= 2;
        Self {
            pipelines: can_run.then(|| Pipelines::new(device)),
            pending: Vec::new(),
            ready: Vec::new(),
            next_id: 1,
        }
    }

    /// Requests whose results haven't been returned by `poll` yet.
    pub(super) fn pending(&self) -> usize {
        self.pending.len() + self.ready.len()
    }

    /// Records and submits both passes for `request`. `texture` finds an
    /// uploaded band's view and size.
    pub(super) fn start<'a>(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        request: StatsRequest,
        texture: impl Fn(TileKey, u32) -> Option<(&'a wgpu::TextureView, [u32; 2])>,
    ) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        let bins = request.bins.clamp(1, MAX_BINS);
        let Some(pipelines) = self
            .pipelines
            .as_ref()
            .filter(|_| !request.bands.is_empty())
        else {
            self.ready.push(StatsResult {
                id,
                histograms: request
                    .bands
                    .iter()
                    .map(|&band| (band, empty_histogram(bins)))
                    .collect(),
            });
            return id;
        };

        // Every rectangle of every band that's uploaded and not empty.
        let mut jobs = Vec::new();
        for (slot, &band) in request.bands.iter().enumerate() {
            for &(key, [x0, y0, x1, y1]) in &request.regions {
                let Some((view, [width, height])) = texture(key, band) else {
                    continue;
                };
                let (x1, y1) = (x1.min(width), y1.min(height));
                if x0 >= x1 || y0 >= y1 {
                    continue;
                }
                jobs.push(Job {
                    view,
                    texture: (key, band),
                    params: StatsParams {
                        rect: [x0, y0, x1, y1],
                        slot: slot as u32,
                        bins,
                        _padding: [0; 2],
                    },
                });
            }
        }

        let slots = request.bands.len() as u64;
        let min_max_bytes = slots * 2 * 4;
        let counts_bytes = slots * u64::from(bins) * 4;
        let storage = |label, size| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let min_max = storage("stats_min_max", min_max_bytes);
        let counts = storage("stats_counts", counts_bytes);
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("stats_readback"),
            size: min_max_bytes + counts_bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        // Every dispatch's parameters in one buffer, at aligned offsets.
        let stride = pipelines.params_stride;
        let mut params_bytes = vec![0u8; (jobs.len().max(1) as u64 * stride) as usize];
        for (i, job) in jobs.iter().enumerate() {
            let at = i * stride as usize;
            params_bytes[at..at + size_of::<StatsParams>()]
                .copy_from_slice(bytemuck::bytes_of(&job.params));
        }
        let params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("stats_params"),
            size: params_bytes.len() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&params, 0, &params_bytes);

        let buffers = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("stats_buffers"),
            layout: &pipelines.buffers_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &params,
                        offset: 0,
                        size: NonZeroU64::new(size_of::<StatsParams>() as u64),
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: min_max.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: counts.as_entire_binding(),
                },
            ],
        });
        // One per texture, shared by its rectangles.
        let mut textures: HashMap<(TileKey, u32), wgpu::BindGroup> = HashMap::new();
        for job in &jobs {
            textures.entry(job.texture).or_insert_with(|| {
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("stats_texture"),
                    layout: &pipelines.texture_layout,
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(job.view),
                    }],
                })
            });
        }

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("stats_encoder"),
        });
        // Zero means "nothing seen yet" for the min/max keys as well as the counts.
        encoder.clear_buffer(&min_max, 0, None);
        encoder.clear_buffer(&counts, 0, None);
        for (label, pipeline) in [
            ("stats_min_max", &pipelines.min_max),
            ("stats_histogram", &pipelines.histogram),
        ] {
            // Separate passes, so every min/max is in before any bin is counted.
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(label),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            for (i, job) in jobs.iter().enumerate() {
                let [x0, y0, x1, y1] = job.params.rect;
                pass.set_bind_group(0, &buffers, &[(i as u64 * stride) as u32]);
                pass.set_bind_group(1, &textures[&job.texture], &[]);
                pass.dispatch_workgroups((x1 - x0).div_ceil(BLOCK), (y1 - y0).div_ceil(BLOCK), 1);
            }
        }
        encoder.copy_buffer_to_buffer(&min_max, 0, &readback, 0, min_max_bytes);
        encoder.copy_buffer_to_buffer(&counts, 0, &readback, min_max_bytes, counts_bytes);
        queue.submit([encoder.finish()]);

        // Mapped once the GPU is done, during some later `device.poll`.
        let state = Arc::new(AtomicU8::new(WAITING));
        let callback_state = state.clone();
        readback.map_async(wgpu::MapMode::Read, .., move |result| {
            let state = if result.is_ok() { MAPPED } else { FAILED };
            callback_state.store(state, Ordering::Release);
        });
        self.pending.push(Pending {
            id,
            bands: request.bands,
            bins,
            readback,
            state,
        });
        id
    }

    /// Finished requests, oldest first, without blocking.
    pub(super) fn poll(&mut self, device: &wgpu::Device) -> Vec<StatsResult> {
        if !self.pending.is_empty() {
            // Runs the map callbacks of any work that has finished. An error
            // here means the device is lost; the requests then never finish.
            let _ = device.poll(wgpu::PollType::Poll);
        }
        let mut finished = std::mem::take(&mut self.ready);
        let mut i = 0;
        while i < self.pending.len() {
            let state = self.pending[i].state.load(Ordering::Acquire);
            if state == WAITING {
                i += 1;
                continue;
            }
            let pending = self.pending.remove(i);
            finished.push(pending.finish(state == MAPPED));
        }
        finished
    }
}

impl Pending {
    /// Reads the mapped results into histograms, or gives empty ones if the
    /// mapping failed.
    fn finish(self, mapped: bool) -> StatsResult {
        let empty = || {
            self.bands
                .iter()
                .map(|&band| (band, empty_histogram(self.bins)))
                .collect()
        };
        let histograms = match mapped.then(|| self.readback.get_mapped_range(..)) {
            Some(Ok(view)) => {
                let words: &[u32] = bytemuck::cast_slice(&view);
                let (min_max, counts) = words.split_at(self.bands.len() * 2);
                self.bands
                    .iter()
                    .zip(min_max.chunks_exact(2))
                    .zip(counts.chunks_exact(self.bins as usize))
                    .map(|((&band, keys), counts)| (band, histogram(keys, counts)))
                    .collect()
            }
            _ => empty(),
        };
        self.readback.unmap();
        StatsResult {
            id: self.id,
            histograms,
        }
    }
}

/// A histogram from one band's min/max keys (`[max of !key, max of key]`,
/// as `stats.wgsl` leaves them) and counts.
fn histogram(keys: &[u32], counts: &[u32]) -> Histogram {
    let total = counts.iter().map(|&count| u64::from(count)).sum();
    if keys[1] == 0 || total == 0 {
        return empty_histogram(counts.len() as u32);
    }
    Histogram {
        min: from_order_key(!keys[0]),
        max: from_order_key(keys[1]),
        counts: counts.to_vec(),
        total,
    }
}

fn empty_histogram(bins: u32) -> Histogram {
    Histogram {
        min: 0.0,
        max: 0.0,
        counts: vec![0; bins as usize],
        total: 0,
    }
}

/// The float whose `order_key` in `stats.wgsl` is `key`.
fn from_order_key(key: u32) -> f32 {
    if key & 0x8000_0000 != 0 {
        f32::from_bits(key & 0x7fff_ffff)
    } else {
        f32::from_bits(!key)
    }
}

impl Pipelines {
    fn new(device: &wgpu::Device) -> Self {
        let storage_entry = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let buffers_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("stats_buffers_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        // Each dispatch picks its parameters by offset.
                        has_dynamic_offset: true,
                        min_binding_size: NonZeroU64::new(size_of::<StatsParams>() as u64),
                    },
                    count: None,
                },
                storage_entry(1),
                storage_entry(2),
            ],
        });
        let texture_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("stats_texture_layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            }],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("stats_pipeline_layout"),
            bind_group_layouts: &[Some(&buffers_layout), Some(&texture_layout)],
            immediate_size: 0,
        });
        let shader = device.create_shader_module(wgpu::include_wgsl!("stats.wgsl"));
        let pipeline = |entry_point| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry_point),
                layout: Some(&layout),
                module: &shader,
                entry_point: Some(entry_point),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            })
        };
        let alignment = u64::from(device.limits().min_uniform_buffer_offset_alignment);
        Self {
            min_max: pipeline("min_max_pass"),
            histogram: pipeline("histogram_pass"),
            buffers_layout,
            texture_layout,
            params_stride: (size_of::<StatsParams>() as u64).next_multiple_of(alignment),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `order_key` from `stats.wgsl`.
    fn order_key(value: f32) -> u32 {
        let bits = value.to_bits();
        if bits & 0x8000_0000 != 0 {
            !bits
        } else {
            bits | 0x8000_0000
        }
    }

    #[test]
    fn order_keys_should_sort_like_the_floats() {
        let values = [
            f32::MIN,
            -1e10,
            -1.0,
            -f32::MIN_POSITIVE,
            -0.0,
            0.0,
            1e-40,
            0.5,
            1.0,
            f32::MAX,
        ];
        for pair in values.windows(2) {
            assert!(order_key(pair[0]) <= order_key(pair[1]), "{pair:?}");
        }
        for value in values {
            assert_eq!(from_order_key(order_key(value)).to_bits(), value.to_bits());
            // 0 means "none yet" in the shader, for keys and inverted keys alike.
            assert!(order_key(value) != 0 && order_key(value) != u32::MAX);
        }
    }

    #[test]
    fn histogram_should_be_empty_when_nothing_was_seen() {
        let histogram = histogram(&[0, 0], &[0; 8]);
        assert_eq!(histogram, empty_histogram(8));
    }
}
