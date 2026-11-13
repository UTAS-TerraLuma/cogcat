// Band statistics in two passes over texel rectangles of R32Float textures:
// 1. `min_max`: the smallest and largest finite value per band slot,
// 2. `histogram`: counts in `bins` equal bins over that range, read straight
//    from pass 1's buffer, so the CPU never waits between the passes.
//
// Each dispatch covers one (tile, band) rectangle, with its parameters at a
// dynamic offset in one uniform buffer. A workgroup of 16 × 16 threads covers
// a block of BLOCK × BLOCK texels. The min/max pass reduces into workgroup
// memory first, so one global atomic per workgroup touches the shared buffer;
// the histogram pass counts straight into the global bins.
//
// No big workgroup arrays: on Windows, wgpu compiles shaders with FXC, which
// took 25 s to build a per-workgroup histogram of 4096 bins, and the result
// never finished on WARP.

// Must match `StatsParams` in stats.rs.
struct Params {
    // [x0, y0, x1, y1) in texels, already clamped to the texture.
    rect: vec4<u32>,
    // Which band of the request: indexes `min_max` and `counts`.
    slot: u32,
    bins: u32,
    _padding: vec2<u32>,
}

const WORKGROUP: u32 = 16u;
// Texels per workgroup each way: each thread reads (BLOCK / WORKGROUP)² texels.
const BLOCK: u32 = 64u;

@group(0) @binding(0) var<uniform> params: Params;
// Per slot: [max of !key, max of key], so 0 (as cleared) means "none yet"
// for both. `key` is the order-preserving bit pattern from `order_key`.
@group(0) @binding(1) var<storage, read_write> min_max: array<atomic<u32>>;
// `bins` counts per slot.
@group(0) @binding(2) var<storage, read_write> counts: array<atomic<u32>>;
@group(1) @binding(0) var band: texture_2d<f32>;

var<workgroup> local_min: atomic<u32>;
var<workgroup> local_max: atomic<u32>;

// Exponent all ones: infinite or NaN. By bits, since WGSL lets
// implementations assume floats are always finite.
fn is_finite(bits: u32) -> bool {
    return (bits & 0x7f800000u) != 0x7f800000u;
}

// An unsigned key that sorts like the float: negatives have every bit
// flipped (so more negative is smaller), positives just the sign bit (so
// they sort above all negatives). Finite values never give 0 or 0xffffffff.
fn order_key(bits: u32) -> u32 {
    if (bits & 0x80000000u) != 0u {
        return ~bits;
    }
    return bits | 0x80000000u;
}

fn from_key(key: u32) -> f32 {
    if (key & 0x80000000u) != 0u {
        return bitcast<f32>(key & 0x7fffffffu);
    }
    return bitcast<f32>(~key);
}

// The bin of `value` in `bins` bins from `low` to `low + 2 * half_range`.
// Must match `bin_of` in the tests.
fn bin_of(value: f32, low: f32, half_range: f32, bins: u32) -> u32 {
    if half_range <= 0.0 {
        // A constant band: everything in bin 0.
        return 0u;
    }
    let t = (value * 0.5 - low * 0.5) / half_range;
    // The last bin includes the maximum.
    return min(u32(max(t * f32(bins), 0.0)), bins - 1u);
}

// The texel this thread reads on step `i` of its block, and whether it's
// inside the rectangle.
fn texel_at(workgroup: vec3<u32>, local: vec3<u32>, i: u32) -> vec3<u32> {
    let steps = BLOCK / WORKGROUP;
    let offset = vec2<u32>(i % steps, i / steps) * WORKGROUP + local.xy;
    let texel = params.rect.xy + workgroup.xy * BLOCK + offset;
    let inside = all(texel < params.rect.zw);
    return vec3<u32>(texel, u32(inside));
}

@compute @workgroup_size(16, 16, 1)
fn min_max_pass(
    @builtin(workgroup_id) workgroup: vec3<u32>,
    @builtin(local_invocation_id) local: vec3<u32>,
    @builtin(local_invocation_index) index: u32,
) {
    // Workgroup memory starts zeroed, which means "none yet" for both.
    let steps = BLOCK / WORKGROUP;
    var smallest = 0u; // max of !key
    var largest = 0u;  // max of key
    for (var i = 0u; i < steps * steps; i++) {
        let texel = texel_at(workgroup, local, i);
        if texel.z != 0u {
            let bits = bitcast<u32>(textureLoad(band, texel.xy, 0).r);
            if is_finite(bits) {
                let key = order_key(bits);
                smallest = max(smallest, ~key);
                largest = max(largest, key);
            }
        }
    }
    if largest != 0u {
        atomicMax(&local_min, smallest);
        atomicMax(&local_max, largest);
    }
    workgroupBarrier();
    if index == 0u {
        let found = atomicLoad(&local_max);
        if found != 0u {
            atomicMax(&min_max[params.slot * 2u], atomicLoad(&local_min));
            atomicMax(&min_max[params.slot * 2u + 1u], found);
        }
    }
}

@compute @workgroup_size(16, 16, 1)
fn histogram_pass(
    @builtin(workgroup_id) workgroup: vec3<u32>,
    @builtin(local_invocation_id) local: vec3<u32>,
) {
    let largest_key = atomicLoad(&min_max[params.slot * 2u + 1u]);
    let smallest_key = ~atomicLoad(&min_max[params.slot * 2u]);
    // With no finite texel in any rectangle of this band, there's nothing to
    // count; the loop below finds none either way.
    let low = from_key(smallest_key);
    let high = from_key(largest_key);
    let bins = params.bins;
    // Halved, so the range of values near ±f32::MAX doesn't overflow.
    let half_range = high * 0.5 - low * 0.5;

    let base = params.slot * bins;
    let steps = BLOCK / WORKGROUP;
    for (var i = 0u; i < steps * steps; i++) {
        let texel = texel_at(workgroup, local, i);
        if texel.z != 0u {
            let value = textureLoad(band, texel.xy, 0).r;
            if is_finite(bitcast<u32>(value)) {
                atomicAdd(&counts[base + bin_of(value, low, half_range, bins)], 1u);
            }
        }
    }
}
