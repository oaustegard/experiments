# Corpus

86 WGSL shaders used to test the parser and minifier, fetched 2026-11-01 from
raw.githubusercontent.com and kept unmodified except as noted.

- `naga/` — 55 files from [gfx-rs/wgpu](https://github.com/gfx-rs/wgpu) `naga/tests/in/wgsl/`
  (MIT OR Apache-2.0). Includes naga extensions Tint does not accept (`int64`, `f16`,
  `ray-query`, binding arrays).
- `samples/` — 31 files from [webgpu/webgpu-samples](https://github.com/webgpu/webgpu-samples)
  `sample/*/` (BSD-3-Clause), named `<sample>_<file>.wgsl`. Changes:
  `cornell_{radiosity,rasterizer,raytracer}_full.wgsl` are `cornell_common.wgsl` followed by
  the original file, since the sample concatenates them at runtime;
  `cornell_tonemapper.wgsl` has its `{OUTPUT_FORMAT}` template placeholder replaced by
  `rgba8unorm`. `skinnedMesh/gltf.wgsl` was left out because it needs code from another file.
