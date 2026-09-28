use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    let slangc = find_slangc();
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR"));
    let shader = PathBuf::from("src/shader.slang");
    println!("cargo:rerun-if-changed={}", shader.display());
    println!("cargo:rerun-if-env-changed=SLANG_DIR");

    compile(
        &slangc,
        &shader,
        &out_dir,
        "vertexMain",
        "vertex",
        "shader.vert.spv",
    );
    compile(
        &slangc,
        &shader,
        &out_dir,
        "fragmentMain",
        "fragment",
        "shader.frag.spv",
    );
}

fn compile(slangc: &Path, shader: &Path, out_dir: &Path, entry: &str, stage: &str, output: &str) {
    let destination = out_dir.join(output);
    let result = Command::new(slangc)
        .arg(shader)
        .args([
            "-target",
            "spirv",
            "-profile",
            "spirv_1_4",
            "-matrix-layout-column-major",
            "-fvk-use-entrypoint-name",
            "-entry",
            entry,
            "-stage",
            stage,
            "-o",
        ])
        .arg(&destination)
        .output()
        .unwrap_or_else(|err| panic!("failed to run {}: {err}", slangc.display()));
    if !result.status.success() {
        panic!(
            "slangc {entry} failed:\n{}",
            String::from_utf8_lossy(&result.stderr)
        );
    }
    // Slang writes `OpSource Slang`, which SPIR-V 1.4 and current drivers reject.
    rewrite_source_language(&destination);
}

fn rewrite_source_language(path: &Path) {
    let bytes = std::fs::read(path).unwrap_or_else(|err| panic!("read {}: {err}", path.display()));
    if bytes.len() < 20 || bytes.len() % 4 != 0 {
        panic!("{} is not a SPIR-V module", path.display());
    }
    let mut words: Vec<u32> = bytes
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap()))
        .collect();
    if words[0] != 0x0723_0203 {
        panic!("{} is not little-endian SPIR-V", path.display());
    }
    let mut index = 5;
    let mut patched = false;
    while index < words.len() {
        let word_count = (words[index] >> 16) as usize;
        let opcode = words[index] & 0xffff;
        if word_count == 0 || index + word_count > words.len() {
            panic!("{} has a truncated SPIR-V instruction", path.display());
        }
        if opcode == 3 && word_count >= 3 {
            words[index + 1] = 0;
            patched = true;
            break;
        }
        index += word_count;
    }
    if !patched {
        panic!("{} has no OpSource instruction to rewrite", path.display());
    }
    let mut out = Vec::with_capacity(bytes.len());
    for word in words {
        out.extend_from_slice(&word.to_le_bytes());
    }
    std::fs::write(path, out).unwrap_or_else(|err| panic!("write {}: {err}", path.display()));
}

fn find_slangc() -> PathBuf {
    if let Ok(dir) = env::var("SLANG_DIR") {
        let candidate = PathBuf::from(dir).join("bin").join("slangc");
        if candidate.is_file() {
            return candidate;
        }
    }
    if let Some(path) = env::var_os("PATH") {
        for dir in env::split_paths(&path) {
            let candidate = dir.join("slangc");
            if candidate.is_file() {
                return candidate;
            }
        }
    }
    let fallback = PathBuf::from("/opt/slang/bin/slangc");
    if fallback.is_file() {
        return fallback;
    }
    panic!("slangc not found; set SLANG_DIR or put slangc on PATH");
}
