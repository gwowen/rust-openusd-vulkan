//! In-memory OpenUSD stage: checkerboard meshes and orbiting translates.

use anyhow::{Context, Result, bail};
use openusd::{gf, sdf, usd};

use crate::mesh::{self, MeshData, PALETTE};

pub const ORBIT_SAMPLES: f64 = 120.0;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub color: [f32; 3],
}

pub struct SceneObject {
    pub vertices: Vec<Vertex>,
    translate: String,
}

pub struct Scene {
    stage: usd::Stage,
    objects: Vec<SceneObject>,
}

impl Scene {
    pub fn create() -> Result<Self> {
        let stage = usd::Stage::builder().in_memory("scene.usda")?;
        stage.define_prim("/World")?.set_type_name("Xform")?;
        stage.set_default_prim("World")?;

        let bodies = [
            Body {
                path: "/World/Cube",
                phase: 0.0,
                inclination: 0.0,
                mesh: mesh::checker_cube(8, 1.25),
            },
            Body {
                path: "/World/Torus",
                phase: std::f64::consts::FRAC_PI_3 * 2.0,
                inclination: 55.0_f64.to_radians(),
                mesh: mesh::checker_torus(40, 20, 0.65, 0.22),
            },
            Body {
                path: "/World/Sphere",
                phase: std::f64::consts::FRAC_PI_3 * 4.0,
                inclination: -55.0_f64.to_radians(),
                mesh: mesh::checker_sphere(24, 32, 0.7),
            },
        ];

        let mut objects = Vec::with_capacity(bodies.len());
        for body in bodies {
            author_mesh(&stage, body.path, &body.mesh)?;
            author_orbit(&stage, body.path, body.phase, body.inclination)?;
            objects.push(SceneObject {
                vertices: read_mesh(&stage, body.path)?,
                translate: format!("{}.xformOp:translate", body.path),
            });
        }

        Ok(Self { stage, objects })
    }

    pub fn export_usda(&self, path: &str) -> Result<()> {
        self.stage.root_layer().export(path)?;
        Ok(())
    }

    pub fn objects(&self) -> &[SceneObject] {
        &self.objects
    }

    /// World-space translations at `time`, which loops across one authored orbit.
    pub fn translations(&self, time: f64) -> Result<Vec<[f32; 3]>> {
        let time = time.rem_euclid(ORBIT_SAMPLES);
        self.objects
            .iter()
            .map(|object| {
                let value = self
                    .stage
                    .attribute(&object.translate)?
                    .get_at::<[f64; 3]>(Some(usd::TimeCode::new(time)))?
                    .with_context(|| format!("missing {}", object.translate))?;
                Ok([value[0] as f32, value[1] as f32, value[2] as f32])
            })
            .collect()
    }
}

struct Body {
    path: &'static str,
    phase: f64,
    inclination: f64,
    mesh: MeshData,
}

fn author_mesh(stage: &usd::Stage, path: &str, mesh: &MeshData) -> Result<()> {
    stage.define_prim(path)?.set_type_name("Mesh")?;

    let points = mesh.points.iter().copied().map(vec3f).collect::<Vec<_>>();
    let normals = mesh.normals.iter().copied().map(vec3f).collect::<Vec<_>>();
    let colors = PALETTE.map(vec3f).to_vec();

    write_array(
        stage,
        &format!("{path}.points"),
        "point3f[]",
        sdf::Value::Vec3fVec(points),
        None,
    )?;
    write_array(
        stage,
        &format!("{path}.normals"),
        "normal3f[]",
        sdf::Value::Vec3fVec(normals),
        Some("vertex"),
    )?;
    write_array(
        stage,
        &format!("{path}.faceVertexCounts"),
        "int[]",
        sdf::Value::IntVec(mesh.face_vertex_counts.clone()),
        None,
    )?;
    write_array(
        stage,
        &format!("{path}.faceVertexIndices"),
        "int[]",
        sdf::Value::IntVec(mesh.face_vertex_indices.clone()),
        None,
    )?;
    write_array(
        stage,
        &format!("{path}.primvars:displayColor"),
        "color3f[]",
        sdf::Value::Vec3fVec(colors),
        Some("uniform"),
    )?;
    write_array(
        stage,
        &format!("{path}.primvars:displayColor:indices"),
        "int[]",
        sdf::Value::IntVec(mesh.color_indices.clone()),
        None,
    )?;

    stage
        .create_attribute(format!("{path}.subdivisionScheme"), "token")?
        .set_custom(false)?
        .set_variability(sdf::Variability::Uniform)?
        .set(sdf::Value::token("none"))?;
    Ok(())
}

fn author_orbit(stage: &usd::Stage, path: &str, phase: f64, inclination: f64) -> Result<()> {
    stage
        .create_attribute(format!("{path}.xformOpOrder"), "token[]")?
        .set_custom(false)?
        .set_variability(sdf::Variability::Uniform)?
        .set(sdf::Value::token_vec(["xformOp:translate"]))?;

    let mut translate = stage
        .create_attribute(format!("{path}.xformOp:translate"), "double3")?
        .set_custom(false)?;
    for sample in 0..=ORBIT_SAMPLES as i32 {
        let position = orbit_position(sample as f64, phase, inclination);
        translate = translate.set_at(position, Some(usd::TimeCode::new(sample as f64)))?;
    }
    Ok(())
}

fn orbit_position(sample: f64, phase: f64, inclination: f64) -> [f64; 3] {
    let theta = sample / ORBIT_SAMPLES * std::f64::consts::TAU + phase;
    let x = 3.0 * theta.cos();
    let z = 3.0 * theta.sin();
    let y = -z * inclination.sin();
    let z = z * inclination.cos();
    [x, y, z]
}

fn write_array(
    stage: &usd::Stage,
    path: &str,
    type_name: &str,
    value: sdf::Value,
    interpolation: Option<&'static str>,
) -> Result<()> {
    let mut attribute = stage.create_attribute(path, type_name)?.set_custom(false)?;
    if let Some(interpolation) = interpolation {
        attribute = attribute.set_metadata("interpolation", sdf::Value::token(interpolation))?;
    }
    attribute.set(value)?;
    Ok(())
}

fn read_mesh(stage: &usd::Stage, path: &str) -> Result<Vec<Vertex>> {
    let points = required::<Vec<gf::Vec3f>>(stage, &format!("{path}.points"))?;
    let normals = required::<Vec<gf::Vec3f>>(stage, &format!("{path}.normals"))?;
    let counts = required_ints(stage, &format!("{path}.faceVertexCounts"))?;
    let indices = required_ints(stage, &format!("{path}.faceVertexIndices"))?;
    let colors = required::<Vec<gf::Vec3f>>(stage, &format!("{path}.primvars:displayColor"))?;
    let color_indices = required_ints(stage, &format!("{path}.primvars:displayColor:indices"))?;

    if colors.len() != PALETTE.len() {
        bail!("{path} displayColor palette has {} entries", colors.len());
    }
    if color_indices.len() != counts.len() {
        bail!("{path} color indices do not match face count");
    }

    let mut vertices = Vec::new();
    let mut cursor = 0usize;
    for (face, count) in counts.iter().copied().enumerate() {
        let count = count as usize;
        let ids = indices
            .get(cursor..cursor + count)
            .with_context(|| format!("{path} face {face} is missing indices"))?;
        cursor += count;
        if count < 3 {
            continue;
        }
        let color_index = color_indices[face] as usize;
        let color = colors
            .get(color_index)
            .with_context(|| format!("{path} color index {color_index} is out of range"))?;
        let color = [color.x, color.y, color.z];
        for corner in 1..count - 1 {
            for id in [ids[0], ids[corner], ids[corner + 1]] {
                let index = id as usize;
                let point = points
                    .get(index)
                    .with_context(|| format!("{path} point {id}"))?;
                let normal = normals
                    .get(index)
                    .with_context(|| format!("{path} normal {id}"))?;
                vertices.push(Vertex {
                    position: [point.x, point.y, point.z],
                    normal: [normal.x, normal.y, normal.z],
                    color,
                });
            }
        }
    }
    Ok(vertices)
}

fn required_ints(stage: &usd::Stage, path: &str) -> Result<Vec<i32>> {
    match required::<sdf::Value>(stage, path)? {
        sdf::Value::IntVec(values) => Ok(values),
        other => bail!("{path} is {other:?}, expected int[]"),
    }
}

fn required<T>(stage: &usd::Stage, path: &str) -> Result<T>
where
    T: TryFrom<sdf::Value>,
    T::Error: Into<openusd::Error>,
{
    stage
        .attribute(path)?
        .get::<T>()?
        .with_context(|| format!("missing {path}"))
}

fn vec3f(value: [f32; 3]) -> gf::Vec3f {
    gf::vec3f(value[0], value[1], value[2])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn orbit_samples_are_authored_and_move() {
        let scene = Scene::create().unwrap();
        let start = scene.translations(0.0).unwrap();
        let later = scene.translations(30.0).unwrap();
        assert_eq!(start.len(), 3);
        assert_ne!(start, later);
        for position in start.iter().chain(&later) {
            let radius = (position[0] as f64)
                .hypot(position[1] as f64)
                .hypot(position[2] as f64);
            assert!(
                (radius - 3.0).abs() < 0.05,
                "radius {radius} at {position:?}"
            );
        }
    }
}
