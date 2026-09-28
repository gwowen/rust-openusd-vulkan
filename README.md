# rust-openusd-vulkan

A small viewer that builds an OpenUSD stage of a checkerboard cube, torus, and sphere, then draws them with Vulkan as they orbit a shared center on a sky-blue background.

The scene is authored in memory with the [`openusd`](https://crates.io/crates/openusd) crate. Each shape is a `Mesh` with a uniform black-and-white `primvars:displayColor` checkerboard. Motion is time-sampled `xformOp:translate`: three circular paths of radius 3, 120° apart, on planes tilted 0°, +55°, and −55°. One revolution takes about eight seconds. On startup the stage is also written to `scene.usda` in the working directory.

The window is [winit](https://crates.io/crates/winit). Rendering is [ash](https://crates.io/crates/ash) (Vulkan). Shaders are [Slang](https://shader-slang.org/), compiled to SPIR-V by `slangc` when the crate builds.

## Requirements

- Rust 1.85 or newer (edition 2024)
- A Vulkan 1.2 loader and a GPU that can present to a window
- `slangc` 2026.18 or compatible

`slangc` is resolved from `SLANG_DIR/bin`, then `PATH`, then `/opt/slang/bin/slangc`.

## Run

```bash
cargo run
```

Close the window or press Escape to quit.

`VK_LAYER_KHRONOS_validation` is enabled automatically when that layer is installed.

## License

MIT. See [LICENSE](LICENSE).
