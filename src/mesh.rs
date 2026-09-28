//! Parametric meshes authored into the USD stage.

#[derive(Clone, Debug)]
pub struct MeshData {
    pub points: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
    pub face_vertex_counts: Vec<i32>,
    pub face_vertex_indices: Vec<i32>,
    /// Palette index per face: 0 is the dark square, 1 is the light square.
    pub color_indices: Vec<i32>,
}

pub const PALETTE: [[f32; 3]; 2] = [[0.08, 0.08, 0.08], [0.93, 0.93, 0.93]];

pub fn checker_cube(cells: u32, size: f32) -> MeshData {
    let mut mesh = MeshData::default();
    let half = size * 0.5;
    let faces = [
        // origin, axis_u, axis_v, normal. u × v points along the outward normal.
        Face::new(
            [-half, -half, half],
            [size, 0.0, 0.0],
            [0.0, size, 0.0],
            [0.0, 0.0, 1.0],
        ),
        Face::new(
            [-half, half, -half],
            [size, 0.0, 0.0],
            [0.0, -size, 0.0],
            [0.0, 0.0, -1.0],
        ),
        Face::new(
            [half, -half, half],
            [0.0, 0.0, -size],
            [0.0, size, 0.0],
            [1.0, 0.0, 0.0],
        ),
        Face::new(
            [-half, -half, -half],
            [0.0, 0.0, size],
            [0.0, size, 0.0],
            [-1.0, 0.0, 0.0],
        ),
        Face::new(
            [-half, half, half],
            [size, 0.0, 0.0],
            [0.0, 0.0, -size],
            [0.0, 1.0, 0.0],
        ),
        Face::new(
            [-half, -half, -half],
            [size, 0.0, 0.0],
            [0.0, 0.0, size],
            [0.0, -1.0, 0.0],
        ),
    ];
    for face in faces {
        add_grid(&mut mesh, face, cells);
    }
    mesh
}

pub fn checker_sphere(stacks: u32, slices: u32, radius: f32) -> MeshData {
    let mut mesh = MeshData::default();
    let stacks = stacks as usize;
    let slices = slices as usize;
    for stack in 0..=stacks {
        let phi = stack as f32 / stacks as f32 * std::f32::consts::PI;
        for slice in 0..slices {
            let theta = slice as f32 / slices as f32 * std::f32::consts::TAU;
            let point = [
                radius * phi.sin() * theta.cos(),
                radius * phi.cos(),
                radius * phi.sin() * theta.sin(),
            ];
            mesh.normals.push(normalize(point));
            mesh.points.push(point);
        }
    }
    for stack in 0..stacks {
        for slice in 0..slices {
            let next = (slice + 1) % slices;
            let i0 = (stack * slices + slice) as i32;
            let i1 = (stack * slices + next) as i32;
            let i2 = ((stack + 1) * slices + next) as i32;
            let i3 = ((stack + 1) * slices + slice) as i32;
            mesh.push_quad(i0, i1, i2, i3, ((stack + slice) % 2) as i32);
        }
    }
    mesh
}

pub fn checker_torus(
    major_seg: u32,
    minor_seg: u32,
    major_radius: f32,
    minor_radius: f32,
) -> MeshData {
    let mut mesh = MeshData::default();
    let major_seg = major_seg as usize;
    let minor_seg = minor_seg as usize;
    for major in 0..major_seg {
        let theta = major as f32 / major_seg as f32 * std::f32::consts::TAU;
        let (ct, st) = (theta.cos(), theta.sin());
        for minor in 0..minor_seg {
            let phi = minor as f32 / minor_seg as f32 * std::f32::consts::TAU;
            let (cp, sp) = (phi.cos(), phi.sin());
            let ring = major_radius + minor_radius * cp;
            mesh.points.push([ring * ct, minor_radius * sp, ring * st]);
            mesh.normals.push([cp * ct, sp, cp * st]);
        }
    }
    for major in 0..major_seg {
        let major_next = (major + 1) % major_seg;
        for minor in 0..minor_seg {
            let minor_next = (minor + 1) % minor_seg;
            let i0 = (major * minor_seg + minor) as i32;
            let i1 = (major * minor_seg + minor_next) as i32;
            let i2 = (major_next * minor_seg + minor_next) as i32;
            let i3 = (major_next * minor_seg + minor) as i32;
            mesh.push_quad(i0, i1, i2, i3, ((major + minor) % 2) as i32);
        }
    }
    mesh
}

struct Face {
    origin: [f32; 3],
    axis_u: [f32; 3],
    axis_v: [f32; 3],
    normal: [f32; 3],
}

impl Face {
    fn new(origin: [f32; 3], axis_u: [f32; 3], axis_v: [f32; 3], normal: [f32; 3]) -> Self {
        Self {
            origin,
            axis_u,
            axis_v,
            normal,
        }
    }
}

fn add_grid(mesh: &mut MeshData, face: Face, cells: u32) {
    let cells = cells as usize;
    let base = mesh.points.len() as i32;
    let stride = cells as i32 + 1;
    for j in 0..=cells {
        let v = j as f32 / cells as f32;
        for i in 0..=cells {
            let u = i as f32 / cells as f32;
            mesh.points.push(add(
                face.origin,
                add(scale(face.axis_u, u), scale(face.axis_v, v)),
            ));
            mesh.normals.push(face.normal);
        }
    }
    for j in 0..cells {
        for i in 0..cells {
            let i0 = base + (j as i32) * stride + i as i32;
            mesh.push_quad(
                i0,
                i0 + 1,
                i0 + stride + 1,
                i0 + stride,
                ((i + j) % 2) as i32,
            );
        }
    }
}

impl Default for MeshData {
    fn default() -> Self {
        Self {
            points: Vec::new(),
            normals: Vec::new(),
            face_vertex_counts: Vec::new(),
            face_vertex_indices: Vec::new(),
            color_indices: Vec::new(),
        }
    }
}

impl MeshData {
    fn push_quad(&mut self, a: i32, b: i32, c: i32, d: i32, color: i32) {
        self.face_vertex_counts.push(4);
        self.face_vertex_indices.extend([a, b, c, d]);
        self.color_indices.push(color);
    }
}

fn add(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn scale(v: [f32; 3], s: f32) -> [f32; 3] {
    [v[0] * s, v[1] * s, v[2] * s]
}

fn normalize(v: [f32; 3]) -> [f32; 3] {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if len == 0.0 {
        [0.0, 1.0, 0.0]
    } else {
        scale(v, 1.0 / len)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn outward(mesh: &MeshData) -> bool {
        let mut cursor = 0;
        mesh.face_vertex_counts.iter().all(|count| {
            let ids = &mesh.face_vertex_indices[cursor..cursor + *count as usize];
            cursor += *count as usize;
            let a = mesh.points[ids[0] as usize];
            let b = mesh.points[ids[1] as usize];
            let c = mesh.points[ids[2] as usize];
            let edge_u = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
            let edge_v = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
            let normal = [
                edge_u[1] * edge_v[2] - edge_u[2] * edge_v[1],
                edge_u[2] * edge_v[0] - edge_u[0] * edge_v[2],
                edge_u[0] * edge_v[1] - edge_u[1] * edge_v[0],
            ];
            let length = normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2];
            if length < 1.0e-8 {
                return true;
            }
            let stored = mesh.normals[ids[0] as usize];
            normal[0] * stored[0] + normal[1] * stored[1] + normal[2] * stored[2] > 0.0
        })
    }

    #[test]
    fn shapes_are_checkerboards_with_outward_faces() {
        for mesh in [
            checker_cube(8, 1.25),
            checker_sphere(16, 24, 0.7),
            checker_torus(32, 16, 0.65, 0.22),
        ] {
            assert!(mesh.color_indices.iter().any(|c| *c == 0));
            assert!(mesh.color_indices.iter().any(|c| *c == 1));
            assert_eq!(mesh.color_indices.len(), mesh.face_vertex_counts.len());
            assert!(outward(&mesh));
        }
    }
}
