// Draws one tile's quad, reading its three band textures texel by texel,
// then stretching, applying gamma and (optionally) a colour map.

// Must match `TileUniforms` in mod.rs byte for byte.
struct Uniforms {
    // Per channel (r, g, b, unused): the values that map to 0 and to 1.
    low: vec4<f32>,
    high: vec4<f32>,
    // Straight RGBA, drawn where any needed band has no data.
    background: vec4<f32>,
    inverse_gamma: f32,
    // 1 to draw band 0 through the colour map, 0 for an RGB composite.
    use_colormap: u32,
    _padding: vec2<u32>,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
// 256 × 1, read with `textureLoad`.
@group(0) @binding(1) var colormap: texture_2d<f32>;

// R32Float isn't filterable, so these are only ever read with `textureLoad`.
@group(1) @binding(0) var band_r: texture_2d<f32>;
@group(1) @binding(1) var band_g: texture_2d<f32>;
@group(1) @binding(2) var band_b: texture_2d<f32>;

struct VertexIn {
    @location(0) position: vec2<f32>,
    // In texels: 0 at the valid area's top-left corner, `texels` at its bottom-right.
    @location(1) texcoord: vec2<f32>,
    @location(2) tint: vec4<f32>,
}

struct VertexOut {
    @builtin(position) position: vec4<f32>,
    @location(0) texcoord: vec2<f32>,
    @location(1) tint: vec4<f32>,
}

@vertex
fn vs_main(in: VertexIn) -> VertexOut {
    var out: VertexOut;
    out.position = vec4<f32>(in.position, 0.0, 1.0);
    out.texcoord = in.texcoord;
    out.tint = in.tint;
    return out;
}

// NaN by its bits: exponent all ones and a non-zero mantissa. WGSL lets
// implementations assume floats are never NaN, so `x != x` can be optimised away.
fn is_nan(x: f32) -> bool {
    let bits = bitcast<u32>(x);
    return (bits & 0x7f800000u) == 0x7f800000u && (bits & 0x007fffffu) != 0u;
}

// `value` stretched from `low..high` to 0..1. With `low == high` (a constant
// band), that value maps to 0.5, below it to 0, and above it to 1.
fn stretch(value: f32, low: f32, high: f32) -> f32 {
    if high == low {
        return select(select(0.0, 1.0, value > low), 0.5, value == low);
    }
    return clamp((value - low) / (high - low), 0.0, 1.0);
}

// `pow(t, inverse_gamma)` for t in 0..1, exact at 0 (where `pow` is undefined)
// and untouched at gamma 1.
fn apply_gamma(t: vec3<f32>) -> vec3<f32> {
    if uniforms.inverse_gamma == 1.0 {
        return t;
    }
    return select(pow(t, vec3<f32>(uniforms.inverse_gamma)), vec3<f32>(0.0), t <= vec3<f32>(0.0));
}

fn colormap_at(t: f32) -> vec3<f32> {
    // Interpolated between the two nearest entries.
    let position = t * 255.0;
    let i = u32(floor(position));
    let next = min(i + 1u, 255u);
    let a = textureLoad(colormap, vec2<u32>(i, 0u), 0).rgb;
    let b = textureLoad(colormap, vec2<u32>(next, 0u), 0).rgb;
    return mix(a, b, position - f32(i));
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    // Nearest neighbour: the texel the pixel centre falls in.
    let last = vec2<i32>(textureDimensions(band_r)) - vec2<i32>(1);
    let texel = clamp(vec2<i32>(floor(in.texcoord)), vec2<i32>(0), last);

    var rgb: vec3<f32>;
    if uniforms.use_colormap != 0u {
        let value = textureLoad(band_r, texel, 0).r;
        if is_nan(value) {
            // Transparent: whatever is underneath (or the background) shows.
            discard;
        }
        let t = stretch(value, uniforms.low.r, uniforms.high.r);
        rgb = colormap_at(apply_gamma(vec3<f32>(t)).r);
    } else {
        let value = vec3<f32>(
            textureLoad(band_r, texel, 0).r,
            textureLoad(band_g, texel, 0).r,
            textureLoad(band_b, texel, 0).r,
        );
        if is_nan(value.r) || is_nan(value.g) || is_nan(value.b) {
            discard;
        }
        let t = vec3<f32>(
            stretch(value.r, uniforms.low.r, uniforms.high.r),
            stretch(value.g, uniforms.low.g, uniforms.high.g),
            stretch(value.b, uniforms.low.b, uniforms.high.b),
        );
        rgb = apply_gamma(t);
    }
    return vec4<f32>(mix(rgb, in.tint.rgb, in.tint.a), 1.0);
}
