import numpy as np
import os
import argparse
import time
import taichi as ti
from PIL import Image
import gc
from scipy import ndimage

# 初始化 Taichi 运行时
ti.init(arch=ti.gpu, debug=False)


# 三角形渲染器核心类
@ti.data_oriented
class TriangleRenderer:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.depth_buffer = ti.field(dtype=ti.f32, shape=(height, width))
        self.color_buffer = ti.Vector.field(3, dtype=ti.f32, shape=(height, width))
        self.clear_buffers()

    @ti.kernel
    def clear_buffers(self):
        for i, j in self.depth_buffer:
            self.depth_buffer[i, j] = float("inf")
            self.color_buffer[i, j] = ti.Vector([0.0, 0.0, 0.0])

    @ti.kernel
    def render_triangles_with_texture(
        self,
        vertices_x: ti.types.ndarray(),
        vertices_y: ti.types.ndarray(),
        faces: ti.types.ndarray(),
        z_values: ti.types.ndarray(),
        colors_r: ti.types.ndarray(),
        colors_g: ti.types.ndarray(),
        colors_b: ti.types.ndarray(),
        texcoords_u: ti.types.ndarray(),
        texcoords_v: ti.types.ndarray(),
        face_texcoord_indices: ti.types.ndarray(),
        texture_data: ti.types.ndarray(),
        texture_width: ti.i32,
        texture_height: ti.i32,
        face_count: ti.i32,
        is_perspective: ti.i32,
        use_zbuffer: ti.i32,
        use_texture: ti.i32,
    ) -> ti.i32:
        valid_triangles = 0
        ti.loop_config(block_dim=128)
        for face_idx in range(face_count):
            i0, i1, i2 = faces[face_idx, 0], faces[face_idx, 1], faces[face_idx, 2]
            v1 = ti.Vector([vertices_x[i0], vertices_y[i0]], dt=ti.f32)
            v2 = ti.Vector([vertices_x[i1], vertices_y[i1]], dt=ti.f32)
            v3 = ti.Vector([vertices_x[i2], vertices_y[i2]], dt=ti.f32)
            z1, z2, z3 = z_values[i0], z_values[i1], z_values[i2]
            color = ti.Vector(
                [colors_r[face_idx], colors_g[face_idx], colors_b[face_idx]], dt=ti.f32
            )

            tc1_u, tc1_v, tc2_u, tc2_v, tc3_u, tc3_v = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            if use_texture == 1:
                ti0, ti1, ti2 = (
                    face_texcoord_indices[face_idx, 0],
                    face_texcoord_indices[face_idx, 1],
                    face_texcoord_indices[face_idx, 2],
                )
                tc1_u, tc1_v = texcoords_u[ti0], texcoords_v[ti0]
                tc2_u, tc2_v = texcoords_u[ti1], texcoords_v[ti1]
                tc3_u, tc3_v = texcoords_u[ti2], texcoords_v[ti2]

            min_x = ti.max(0, ti.floor(ti.min(ti.min(v1[0], v2[0]), v3[0])))
            min_y = ti.max(0, ti.floor(ti.min(ti.min(v1[1], v2[1]), v3[1])))
            max_x = ti.min(self.width, ti.ceil(ti.max(ti.max(v1[0], v2[0]), v3[0])))
            max_y = ti.min(self.height, ti.ceil(ti.max(ti.max(v1[1], v2[1]), v3[1])))

            area = 0.5 * ti.abs(
                (v2[0] - v1[0]) * (v3[1] - v1[1]) - (v3[0] - v1[0]) * (v2[1] - v1[1])
            )
            if area < 0.5 or max_x <= min_x or max_y <= min_y:
                center_x = (v1[0] + v2[0] + v3[0]) / 3.0
                center_y = (v1[1] + v2[1] + v3[1]) / 3.0
                center_z = (z1 + z2 + z3) / 3.0
                center_u = (tc1_u + tc2_u + tc3_u) / 3.0 if use_texture else 0.0
                center_v = (tc1_v + tc2_v + tc3_v) / 3.0 if use_texture else 0.0

                if 0 <= center_x < self.width and 0 <= center_y < self.height:
                    px, py = ti.i32(center_x), ti.i32(center_y)
                    if center_z != float("inf") and (
                        use_zbuffer == 0 or center_z < self.depth_buffer[py, px]
                    ):
                        ti.atomic_min(self.depth_buffer[py, px], center_z)
                        if self.depth_buffer[py, px] == center_z:
                            if use_texture == 1:
                                tex_x = (
                                    ti.i32(center_u * (texture_width - 1))
                                    % texture_width
                                )
                                tex_y = (
                                    ti.i32((1.0 - center_v) * (texture_height - 1))
                                    % texture_height
                                )
                                self.color_buffer[py, px] = ti.Vector(
                                    [
                                        texture_data[tex_y, tex_x, 0],
                                        texture_data[tex_y, tex_x, 1],
                                        texture_data[tex_y, tex_x, 2],
                                    ]
                                )
                            else:
                                self.color_buffer[py, px] = color
                            valid_triangles += 1
                continue

            pixels_filled = 0
            for y in range(ti.i32(min_y), ti.i32(max_y)):
                for x in range(ti.i32(min_x), ti.i32(max_x)):
                    p = ti.Vector([float(x) + 0.5, float(y) + 0.5], dt=ti.f32)
                    bary = barycentric_coordinates_ti(p, v1, v2, v3)
                    if (
                        bary[0] >= -1e-5
                        and bary[1] >= -1e-5
                        and bary[2] >= -1e-5
                        and bary[0] + bary[1] + bary[2] <= 1.01
                    ):
                        z = interpolate_z_ti(bary, z1, z2, z3, is_perspective)
                        if z != float("inf") and (
                            use_zbuffer == 0 or z < self.depth_buffer[y, x]
                        ):
                            ti.atomic_min(self.depth_buffer[y, x], z)
                            if self.depth_buffer[y, x] == z:
                                if use_texture == 1:
                                    u = (
                                        bary[0] * tc1_u
                                        + bary[1] * tc2_u
                                        + bary[2] * tc3_u
                                    )
                                    v = (
                                        bary[0] * tc1_v
                                        + bary[1] * tc2_v
                                        + bary[2] * tc3_v
                                    )
                                    tex_x = (
                                        ti.i32((u % 1.0) * (texture_width - 1))
                                        % texture_width
                                    )
                                    tex_y = (
                                        ti.i32((1.0 - (v % 1.0)) * (texture_height - 1))
                                        % texture_height
                                    )
                                    self.color_buffer[y, x] = ti.Vector(
                                        [
                                            texture_data[tex_y, tex_x, 0],
                                            texture_data[tex_y, tex_x, 1],
                                            texture_data[tex_y, tex_x, 2],
                                        ]
                                    )
                                else:
                                    self.color_buffer[y, x] = color
                                pixels_filled += 1
            if pixels_filled > 0:
                valid_triangles += 1
        return valid_triangles

    @ti.kernel
    def render_triangles(
        self,
        vertices_x: ti.types.ndarray(),
        vertices_y: ti.types.ndarray(),
        faces: ti.types.ndarray(),
        z_values: ti.types.ndarray(),
        colors_r: ti.types.ndarray(),
        colors_g: ti.types.ndarray(),
        colors_b: ti.types.ndarray(),
        face_count: ti.i32,
        is_perspective: ti.i32,
        use_zbuffer: ti.i32,
    ) -> ti.i32:
        valid_triangles = 0
        ti.loop_config(block_dim=128)
        for face_idx in range(face_count):
            i0, i1, i2 = faces[face_idx, 0], faces[face_idx, 1], faces[face_idx, 2]
            v1 = ti.Vector([vertices_x[i0], vertices_y[i0]], dt=ti.f32)
            v2 = ti.Vector([vertices_x[i1], vertices_y[i1]], dt=ti.f32)
            v3 = ti.Vector([vertices_x[i2], vertices_y[i2]], dt=ti.f32)
            z1, z2, z3 = z_values[i0], z_values[i1], z_values[i2]
            color = ti.Vector(
                [colors_r[face_idx], colors_g[face_idx], colors_b[face_idx]], dt=ti.f32
            )

            min_x = ti.max(0, ti.floor(ti.min(ti.min(v1[0], v2[0]), v3[0])))
            min_y = ti.max(0, ti.floor(ti.min(ti.min(v1[1], v2[1]), v3[1])))
            max_x = ti.min(self.width, ti.ceil(ti.max(ti.max(v1[0], v2[0]), v3[0])))
            max_y = ti.min(self.height, ti.ceil(ti.max(ti.max(v1[1], v2[1]), v3[1])))

            area = 0.5 * ti.abs(
                (v2[0] - v1[0]) * (v3[1] - v1[1]) - (v3[0] - v1[0]) * (v2[1] - v1[1])
            )
            if area < 0.5 or max_x <= min_x or max_y <= min_y:
                center_x = (v1[0] + v2[0] + v3[0]) / 3.0
                center_y = (v1[1] + v2[1] + v3[1]) / 3.0
                center_z = (z1 + z2 + z3) / 3.0
                if 0 <= center_x < self.width and 0 <= center_y < self.height:
                    px, py = ti.i32(center_x), ti.i32(center_y)
                    if center_z != float("inf") and (
                        use_zbuffer == 0 or center_z < self.depth_buffer[py, px]
                    ):
                        ti.atomic_min(self.depth_buffer[py, px], center_z)
                        if self.depth_buffer[py, px] == center_z:
                            self.color_buffer[py, px] = color
                            valid_triangles += 1
                continue

            pixels_filled = 0
            for y in range(ti.i32(min_y), ti.i32(max_y)):
                for x in range(ti.i32(min_x), ti.i32(max_x)):
                    p = ti.Vector([float(x) + 0.5, float(y) + 0.5], dt=ti.f32)
                    bary = barycentric_coordinates_ti(p, v1, v2, v3)
                    if (
                        bary[0] >= -1e-5
                        and bary[1] >= -1e-5
                        and bary[2] >= -1e-5
                        and bary[0] + bary[1] + bary[2] <= 1.01
                    ):
                        z = interpolate_z_ti(bary, z1, z2, z3, is_perspective)
                        if z != float("inf") and (
                            use_zbuffer == 0 or z < self.depth_buffer[y, x]
                        ):
                            ti.atomic_min(self.depth_buffer[y, x], z)
                            if self.depth_buffer[y, x] == z:
                                self.color_buffer[y, x] = color
                                pixels_filled += 1
            if pixels_filled > 0:
                valid_triangles += 1
        return valid_triangles

    def get_color_array(self):
        return self.color_buffer.to_numpy()

    def get_depth_array(self):
        return self.depth_buffer.to_numpy()


# Taichi 核心函数
@ti.func
def barycentric_coordinates_ti(p, v1, v2, v3):
    e1 = v2 - v1
    e2 = v3 - v1
    p_v1 = p - v1
    area = e1[0] * e2[1] - e1[1] * e2[0]
    result = ti.Vector([0.0, 0.0, 0.0], dt=ti.f32)
    if ti.abs(area) > 1e-10:
        area1 = (p_v1[0] * e2[1] - p_v1[1] * e2[0]) / area
        area2 = (e1[0] * p_v1[1] - e1[1] * p_v1[0]) / area
        area3 = 1.0 - area1 - area2
        result = ti.Vector([area3, area1, area2], dt=ti.f32)
    return result


@ti.func
def interpolate_z_ti(bary, z1, z2, z3, is_perspective):
    alpha, beta, gamma = bary[0], bary[1], bary[2]
    z = float("inf")
    if alpha + beta + gamma >= 0.99:
        if is_perspective == 0:
            z = alpha * z1 + beta * z2 + gamma * z3
        else:
            if ti.abs(z1) < 1e-10 or ti.abs(z2) < 1e-10 or ti.abs(z3) < 1e-10:
                z = alpha * z1 + beta * z2 + gamma * z3
            else:
                inv_z1, inv_z2, inv_z3 = 1.0 / z1, 1.0 / z2, 1.0 / z3
                inv_z = alpha * inv_z1 + beta * inv_z2 + gamma * inv_z3
                if ti.abs(inv_z) > 1e-10:
                    z = 1.0 / inv_z
    return -z


# 颜色和缓冲区处理函数
def get_face_color(face_index, colorize=False):
    if not colorize:
        return np.array([0.7, 0.7, 0.7], dtype=np.float32)
    np.random.seed(face_index)
    return np.array(
        [
            0.3 + np.random.random() * 0.4,
            0.3 + np.random.random() * 0.4,
            0.3 + np.random.random() * 0.4,
        ],
        dtype=np.float32,
    )


def apply_colormap_jet(depth_image):
    height, width = depth_image.shape
    result = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            val = depth_image[y, x]
            if np.isnan(val) or np.isinf(val):
                result[y, x] = [0, 0, 0]
                continue
            if val <= 0.25:
                t = val * 4.0
                r, g, b = int(0 + t * 0), int(0 + t * 255), int(128 + t * (255 - 128))
            elif val <= 0.5:
                t = (val - 0.25) * 4.0
                r, g, b = int(0 + t * 0), int(255 + t * 0), int(255 + t * (-255))
            elif val <= 0.75:
                t = (val - 0.5) * 4.0
                r, g, b = int(0 + t * 255), 255, int(0 + t * 0)
            else:
                t = (val - 0.75) * 4.0
                r, g, b = (
                    int(255 + t * (128 - 255)),
                    int(255 + t * (-255)),
                    int(0 + t * 0),
                )
            result[y, x] = [
                max(0, min(255, r)),
                max(0, min(255, g)),
                max(0, min(255, b)),
            ]
    return result


# 3D变换与投影函数
def ndc_to_pixel(ndc_coords, width, height):
    pixel_coords = np.zeros_like(ndc_coords)
    pixel_coords[:, 0] = (ndc_coords[:, 0] + 1) * width / 2
    pixel_coords[:, 1] = height - (ndc_coords[:, 1] + 1) * height / 2
    return pixel_coords


def rotate_model(vertices, angle):
    theta = np.radians(angle)
    rotation_matrix = np.array(
        [
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)],
        ],
        dtype=np.float32,
    )
    return vertices @ rotation_matrix


def orthographic_projection(vertices):
    return vertices[:, :2]


def perspective_projection(vertices, focal_length=2.0):
    projected_vertices = np.zeros((vertices.shape[0], 2), dtype=np.float32)
    z_values = vertices[:, 2]
    # 确保 Z 坐标不会太接近 0，避免除零
    z_offset = 1e-5  # 小的正偏移量
    safe_z = np.where(np.abs(z_values) < z_offset, -z_offset, z_values)
    mask = safe_z != 0  # 避免除以 0
    projected_vertices[mask, 0] = vertices[mask, 0] * focal_length / (-safe_z[mask])
    projected_vertices[mask, 1] = vertices[mask, 1] * focal_length / (-safe_z[mask])
    # 对于 Z=0 的顶点，设为默认值（例如无穷远）
    projected_vertices[~mask] = np.inf
    return projected_vertices


# 模型加载与处理
def load_obj_enhanced(
    filename, load_texcoords=True, load_normals=True, load_materials=True
):
    vertices, faces, texcoords, normals = (
        [],
        [],
        [] if load_texcoords else None,
        [] if load_normals else None,
    )
    face_texcoord_indices, face_normal_indices = [] if load_texcoords else None, (
        [] if load_normals else None
    )
    materials, material_indices, current_material, mtl_filename = (
        {} if load_materials else None,
        [] if load_materials else None,
        None,
        None,
    )

    with open(filename, "r") as f:
        for line in f:
            tokens = line.strip().split()
            if not tokens or line.startswith("#"):
                continue
            if tokens[0] == "v":
                vertices.append([float(x) for x in tokens[1:4]])
            elif tokens[0] == "vt" and load_texcoords:
                texcoords.append([float(tokens[1]), float(tokens[2])])
            elif tokens[0] == "vn" and load_normals:
                normals.append([float(x) for x in tokens[1:4]])
            elif tokens[0] == "f":
                face, ftc, fn = (
                    [],
                    [] if load_texcoords else None,
                    [] if load_normals else None,
                )
                for v in tokens[1:]:
                    parts = v.split("/")
                    face.append(int(parts[0]) - 1)
                    if load_texcoords and len(parts) > 1 and parts[1]:
                        ftc.append(int(parts[1]) - 1)
                    elif load_texcoords:
                        ftc.append(0)
                    if load_normals and len(parts) > 2 and parts[2]:
                        fn.append(int(parts[2]) - 1)
                    elif load_normals:
                        fn.append(0)
                if len(face) == 3:
                    faces.append(face)
                    if load_texcoords:
                        face_texcoord_indices.append(ftc)
                    if load_normals:
                        face_normal_indices.append(fn)
                    if load_materials:
                        material_indices.append(current_material)
                elif len(face) == 4:
                    faces.extend(
                        [[face[0], face[1], face[2]], [face[0], face[2], face[3]]]
                    )
                    if load_texcoords:
                        face_texcoord_indices.extend(
                            [[ftc[0], ftc[1], ftc[2]], [ftc[0], ftc[2], ftc[3]]]
                        )
                    if load_normals:
                        face_normal_indices.extend(
                            [[fn[0], fn[1], fn[2]], [fn[0], fn[2], fn[3]]]
                        )
                    if load_materials:
                        material_indices.extend([current_material, current_material])
            elif tokens[0] == "mtllib" and load_materials:
                mtl_filename = " ".join(tokens[1:])
                mtl_path = os.path.join(os.path.dirname(filename), mtl_filename)
                materials = load_mtl(mtl_path) if os.path.exists(mtl_path) else {}
            elif tokens[0] == "usemtl" and load_materials:
                current_material = " ".join(tokens[1:])

    if load_texcoords and not texcoords:
        texcoords = generate_default_texcoords(np.array(vertices))
        face_texcoord_indices = [[i for i in f] for f in faces]
    if load_normals and not normals:
        normals = generate_vertex_normals(np.array(vertices), np.array(faces))
        face_normal_indices = [[i for i in f] for f in faces]

    result = {
        "vertices": np.array(vertices, dtype=np.float32),
        "faces": np.array(faces, dtype=np.int32),
    }
    if load_texcoords:
        result.update(
            {
                "texcoords": np.array(texcoords, dtype=np.float32),
                "face_texcoord_indices": np.array(
                    face_texcoord_indices, dtype=np.int32
                ),
            }
        )
    if load_normals:
        result.update(
            {
                "normals": np.array(normals, dtype=np.float32),
                "face_normal_indices": np.array(face_normal_indices, dtype=np.int32),
            }
        )
    if load_materials:
        result.update(
            {
                "materials": materials,
                "material_indices": (
                    np.array(material_indices) if material_indices else None
                ),
            }
        )
    return result


def load_mtl(mtl_filename):
    materials, current_material = {}, None
    with open(mtl_filename, "r") as f:
        for line in f:
            tokens = line.strip().split()
            if not tokens:
                continue
            if tokens[0] == "newmtl":
                current_material = " ".join(tokens[1:])
                materials[current_material] = {
                    "Ka": [0.2, 0.2, 0.2],
                    "Kd": [0.8, 0.8, 0.8],
                    "Ks": [0.0, 0.0, 0.0],
                    "Ns": 10.0,
                    "d": 1.0,
                    "map_Kd": None,
                    "map_Bump": None,
                }
            elif tokens[0] == "Ka" and current_material:
                materials[current_material]["Ka"] = [float(x) for x in tokens[1:4]]
            elif tokens[0] == "Kd" and current_material:
                materials[current_material]["Kd"] = [float(x) for x in tokens[1:4]]
            elif tokens[0] == "Ks" and current_material:
                materials[current_material]["Ks"] = [float(x) for x in tokens[1:4]]
            elif tokens[0] == "Ns" and current_material:
                materials[current_material]["Ns"] = float(tokens[1])
            elif tokens[0] == "d" and current_material:
                materials[current_material]["d"] = float(tokens[1])
            elif tokens[0] == "map_Kd" and current_material:
                materials[current_material]["map_Kd"] = " ".join(tokens[1:])
            elif tokens[0] == "map_Bump" and current_material:
                materials[current_material]["map_Bump"] = " ".join(tokens[1:])
    return materials


def generate_default_texcoords(vertices):
    center = np.mean(vertices, axis=0)
    vertices_centered = vertices - center
    max_dist = np.max(np.linalg.norm(vertices_centered, axis=1))
    vertices_normalized = vertices_centered / max_dist
    texcoords = np.zeros((len(vertices), 2), dtype=np.float32)
    for i, v in enumerate(vertices_normalized):
        x, y, z = v
        u = 0.5 + np.arctan2(x, z) / (2 * np.pi)
        v = 0.5 - np.arcsin(y) / np.pi
        texcoords[i] = [u, v]
    return texcoords


def generate_vertex_normals(vertices, faces):
    vertex_normals = np.zeros_like(vertices, dtype=np.float32)
    for face in faces:
        if len(face) >= 3:
            v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
            normal = np.cross(v1 - v0, v2 - v0)
            norm = np.linalg.norm(normal)
            if norm > 1e-10:
                normal /= norm
            vertex_normals[face[0]] += normal
            vertex_normals[face[1]] += normal
            vertex_normals[face[2]] += normal
    for i in range(len(vertex_normals)):
        norm = np.linalg.norm(vertex_normals[i])
        if norm > 1e-10:
            vertex_normals[i] /= norm
    return vertex_normals


def load_texture(texture_path, default_color=[0.7, 0.7, 0.7]):
    try:
        if os.path.exists(texture_path):
            img = Image.open(texture_path).convert("RGBA")
            return np.array(img, dtype=np.float32) / 255.0
    except Exception as e:
        print(f"纹理加载错误: {e}")
    return np.array(
        [[[default_color[0], default_color[1], default_color[2], 1.0]]],
        dtype=np.float32,
    )


def generate_procedural_texture(
    texture_type="checkerboard",
    size=512,
    color1=[0.8, 0.8, 0.8],
    color2=[0.2, 0.2, 0.2],
):
    texture = np.zeros((size, size, 4), dtype=np.float32)
    if texture_type == "checkerboard":
        check_size = size // 8
        for i in range(size):
            for j in range(size):
                texture[i, j] = (
                    (color1 + [1.0])
                    if ((i // check_size) + (j // check_size)) % 2 == 0
                    else (color2 + [1.0])
                )
    elif texture_type == "gradient":
        for i in range(size):
            t = i / size
            for j in range(size):
                r = color1[0] * (1 - t) + color2[0] * t
                g = color1[1] * (1 - t) + color2[1] * t
                b = color1[2] * (1 - t) + color2[2] * t
                texture[i, j] = [r, g, b, 1.0]
    elif texture_type == "noise":
        from numpy.random import RandomState

        rng = RandomState(42)
        noise = rng.rand(size // 4, size // 4)
        noise_large = ndimage.zoom(noise, 4, order=1)
        for i in range(size):
            for j in range(size):
                t = noise_large[i, j]
                r = color1[0] * (1 - t) + color2[0] * t
                g = color1[1] * (1 - t) + color2[1] * t
                b = color1[2] * (1 - t) + color2[2] * t
                texture[i, j] = [r, g, b, 1.0]
    return texture


# 主渲染函数
def render_model_with_texture(
    vertices,
    faces,
    width,
    height,
    projection="orthographic",
    use_zbuffer=True,
    angle=0,
    focal_length=2.0,
    model_name="model",
    colorize=False,
    render_depth=True,
    output_dir="output",
    texture_data=None,
    use_texture=False,
    texcoords=None,
    face_texcoord_indices=None,
    normals=None,
    face_normal_indices=None,
    materials=None,
    material_indices=None,
    depth_range_percentile=(1, 99),
    use_lighting=True,
    light_model="phong",
    ambient=0.2,
    diffuse=0.6,
    specular=0.2,
    shininess=32.0,
    light_dir=None,
):
    start_time = time.time()
    os.makedirs(output_dir, exist_ok=True)
    color_dir = os.path.join(output_dir, "color")
    depth_dir = os.path.join(output_dir, "depth")
    os.makedirs(color_dir, exist_ok=True)
    if render_depth:
        os.makedirs(depth_dir, exist_ok=True)

    rotated_vertices = rotate_model(vertices.astype(np.float64), angle).astype(
        np.float32
    )
    # 标准化模型
    center = np.mean(rotated_vertices, axis=0)
    rotated_vertices -= center
    scale = np.max(np.abs(rotated_vertices))
    if scale > 0:  # 避免除以 0
        rotated_vertices /= scale * 0.8
    max_z = np.max(rotated_vertices[:, 2])
    min_z = np.min(rotated_vertices[:, 2])
    z_range = max_z - min_z
    z_offset = max_z + (
        z_range * 0.1 if projection == "orthographic" else z_range * 2.0
    )
    rotated_vertices[:, 2] -= z_offset

    is_perspective = projection == "perspective"
    projected_vertices = (
        orthographic_projection(rotated_vertices)
        if projection == "orthographic"
        else perspective_projection(rotated_vertices, focal_length)
    )
    pixel_vertices = ndc_to_pixel(projected_vertices, width, height)
    renderer = TriangleRenderer(width, height)

    print(
        f"渲染{model_name}模型 (角度: {angle}°, {'persp' if is_perspective else 'ortho'}投影, {'启用' if use_texture else '禁用'}纹理)"
    )

    valid_faces, face_colors, valid_face_texcoord_indices = (
        [],
        [],
        [] if face_texcoord_indices is not None else None,
    )

    for idx, face in enumerate(faces):
        if len(face) == 3:
            material = None
            if materials and material_indices is not None:
                mat_idx = material_indices[idx]
                if mat_idx in materials:  # 直接检查字符串是否在字典键中
                    material = materials[mat_idx]
            color = (
                np.array(material["Kd"], dtype=np.float32)
                if material
                else get_face_color(idx, colorize)
            )
            valid_faces.append(face)
            face_colors.append(color)
            if face_texcoord_indices is not None:
                valid_face_texcoord_indices.append(face_texcoord_indices[idx])

    if not valid_faces:
        print("没有有效的三角形可渲染！")
        return None, None

    valid_faces = np.array(valid_faces, dtype=np.int32)
    face_colors = np.array(face_colors, dtype=np.float32)
    if use_lighting and normals is not None and face_normal_indices is not None:
        # 设置默认光源方向
        if light_dir is None:
            light_dir = np.array([1.0, -1.0, 1.0], dtype=np.float32)
            light_dir = light_dir / np.linalg.norm(light_dir)

        # 计算包含光照的face_colors
        for idx, face in enumerate(valid_faces):
            # 获取面的法线
            fn_idx = face_normal_indices[idx]
            face_normal = np.mean(
                [normals[fn_idx[0]], normals[fn_idx[1]], normals[fn_idx[2]]], axis=0
            )
            face_normal = face_normal / np.linalg.norm(face_normal)

            # 基础颜色（来自材质或随机生成）
            base_color = face_colors[idx]

            # 计算光照
            # 1. 环境光
            ambient_component = ambient * base_color

            # 2. 漫反射
            diffuse_strength = max(0.0, np.dot(face_normal, -light_dir))
            diffuse_component = diffuse * diffuse_strength * base_color

            # 3. 镜面反射 (Phong)
            if light_model.lower() == "phong":
                view_dir = np.array([0.0, 0.0, 1.0])  # 假设相机在z轴正方向
                reflect_dir = (
                    light_dir - 2.0 * np.dot(light_dir, face_normal) * face_normal
                )
                spec_strength = max(0.0, np.dot(view_dir, reflect_dir)) ** shininess
                specular_component = specular * spec_strength * np.ones(3)
            # Blinn-Phong
            else:
                view_dir = np.array([0.0, 0.0, 1.0])
                halfway_dir = -light_dir + view_dir
                halfway_dir = halfway_dir / np.linalg.norm(halfway_dir)
                spec_strength = max(0.0, np.dot(face_normal, halfway_dir)) ** shininess
                specular_component = specular * spec_strength * np.ones(3)

            # 组合所有光照分量
            face_colors[idx] = (
                ambient_component + diffuse_component + specular_component
            )
            # 确保颜色在有效范围内
            face_colors[idx] = np.clip(face_colors[idx], 0.0, 1.0)
    if valid_face_texcoord_indices is not None:
        valid_face_texcoord_indices = np.array(
            valid_face_texcoord_indices, dtype=np.int32
        )

    vertices_x = pixel_vertices[:, 0].astype(np.float32)
    vertices_y = pixel_vertices[:, 1].astype(np.float32)
    z_values = rotated_vertices[:, 2].astype(np.float32)
    colors_r, colors_g, colors_b = (
        face_colors[:, 0].astype(np.float32),
        face_colors[:, 1].astype(np.float32),
        face_colors[:, 2].astype(np.float32),
    )

    texture_width, texture_height = 1, 1
    if use_texture and texture_data is not None:
        if len(texture_data.shape) == 3:
            texture_height, texture_width = texture_data.shape[:2]
            if texture_data.shape[2] == 3:
                rgba_texture = np.ones(
                    (texture_height, texture_width, 4), dtype=np.float32
                )
                rgba_texture[:, :, :3] = texture_data.astype(np.float32)
                texture_data = rgba_texture
    elif use_texture:
        texture_data = np.ones((1, 1, 4), dtype=np.float32)

    texcoords_u, texcoords_v = (
        (texcoords[:, 0].astype(np.float32), texcoords[:, 1].astype(np.float32))
        if use_texture and texcoords is not None
        else (None, None)
    )

    valid_triangle_count = (
        renderer.render_triangles_with_texture(
            vertices_x,
            vertices_y,
            valid_faces,
            z_values,
            colors_r,
            colors_g,
            colors_b,
            texcoords_u,
            texcoords_v,
            valid_face_texcoord_indices,
            texture_data,
            texture_width,
            texture_height,
            len(valid_faces),
            1 if is_perspective else 0,
            1 if use_zbuffer else 0,
            1 if use_texture else 0,
        )
        if use_texture
        else renderer.render_triangles(
            vertices_x,
            vertices_y,
            valid_faces,
            z_values,
            colors_r,
            colors_g,
            colors_b,
            len(valid_faces),
            1 if is_perspective else 0,
            1 if use_zbuffer else 0,
        )
    )

    color_buffer = renderer.get_color_array()
    depth_buffer = renderer.get_depth_array()
    render_time = time.time() - start_time
    print(
        f"渲染了 {valid_triangle_count}/{len(valid_faces)} 个有效三角形，用时: {render_time:.2f} 秒"
    )

    img_array = (color_buffer * 255).astype(np.uint8)
    img = Image.fromarray(img_array)
    color_path = os.path.join(color_dir, f"{model_name}.png")
    img.save(color_path)
    print(f"保存彩色图像: {color_path}")

    depth_path = None
    if use_zbuffer and render_depth:
        mask = np.isfinite(depth_buffer)
        valid_depths = depth_buffer[mask]
        if len(valid_depths) == 0:
            print("深度缓冲区中没有有效值，跳过深度图生成")
        else:
            min_depth = float(np.percentile(valid_depths, depth_range_percentile[0]))
            max_depth = float(np.percentile(valid_depths, depth_range_percentile[1]))
            if abs(max_depth - min_depth) < 1e-6:
                max_depth = min_depth + 1.0
            normalized_depth = np.zeros_like(depth_buffer)
            valid_mask = np.logical_and(
                np.isfinite(depth_buffer),
                np.logical_and(
                    depth_buffer >= min_depth - (max_depth - min_depth) * 0.1,
                    depth_buffer <= max_depth + (max_depth - min_depth) * 0.1,
                ),
            )
            depth_range = max_depth - min_depth
            if depth_range > 1e-10:
                normalized_depth[valid_mask] = np.clip(
                    (depth_buffer[valid_mask] - min_depth) / depth_range, 0.0, 1.0
                )
            normalized_depth = ndimage.median_filter(normalized_depth, size=2)
            depth_colored = apply_colormap_jet(1 - normalized_depth)
            depth_img = Image.fromarray(depth_colored)
            depth_path = os.path.join(depth_dir, f"{model_name}.png")
            depth_img.save(depth_path)
            print(f"保存深度图: {depth_path}")

    gc.collect()
    return color_path, depth_path


# 程序入口
def main():
    try:
        import numpy as np
        import taichi as ti
        from PIL import Image
        from scipy import ndimage
    except ImportError as e:
        raise ImportError(f"缺少必要依赖: {e}")

    parser = argparse.ArgumentParser(description="Taichi GPU加速三角形渲染器")
    parser.add_argument("--obj", type=str, required=True, help="OBJ文件路径")
    parser.add_argument("--output", type=str, required=True, help="输出文件名")
    parser.add_argument("--width", type=int, default=800, help="输出图像宽度")
    parser.add_argument("--height", type=int, default=800, help="输出图像高度")
    parser.add_argument(
        "--projection",
        type=str,
        default="perspective",
        choices=["perspective", "orthographic"],
        help="投影类型",
    )
    parser.add_argument("--angle", type=float, default=0, help="绕Y轴旋转角度")
    parser.add_argument("--no-zbuffer", action="store_true", help="禁用Z-buffer")
    parser.add_argument("--colorize", action="store_true", help="启用随机颜色")
    parser.add_argument("--no-depth", action="store_true", help="不生成深度图")
    parser.add_argument("--focal", type=float, default=2.0, help="透视投影焦距")
    parser.add_argument("--output-dir", type=str, default="output", help="输出目录")
    parser.add_argument("--texture", type=str, help="纹理图像路径")
    parser.add_argument(
        "--texture-type",
        type=str,
        default="checkerboard",
        choices=["checkerboard", "gradient", "noise"],
        help="程序化纹理类型",
    )
    parser.add_argument("--texture-size", type=int, default=512, help="程序化纹理大小")
    parser.add_argument("--no-texture", action="store_true", help="禁用纹理渲染")
    parser.add_argument("--no-materials", action="store_true", help="禁用材质加载")
    parser.add_argument("--depth-min", type=int, default=1, help="深度范围最小百分位")
    parser.add_argument("--depth-max", type=int, default=99, help="深度范围最大百分位")

    # 光照参数
    parser.add_argument("--no-lighting", action="store_true", help="禁用光照模型")
    parser.add_argument(
        "--light-model",
        type=str,
        default="phong",
        choices=["phong", "blinn-phong"],
        help="光照模型类型",
    )
    parser.add_argument("--ambient", type=float, default=0.2, help="环境光强度")
    parser.add_argument("--diffuse", type=float, default=0.6, help="漫反射强度")
    parser.add_argument("--specular", type=float, default=0.2, help="高光强度")
    parser.add_argument("--shininess", type=float, default=32.0, help="高光锐度")
    parser.add_argument(
        "--light-dir", type=str, default="1,-1,1", help="光源方向 (x,y,z格式)"
    )

    args = parser.parse_args()

    if not os.path.exists(args.obj):
        raise FileNotFoundError(f"OBJ 文件未找到: {args.obj}")
    print(f"加载模型: {args.obj}")
    model_data = load_obj_enhanced(
        args.obj,
        load_texcoords=not args.no_texture,
        load_normals=True,
        load_materials=not args.no_materials,
    )

    texture_data = None
    use_texture = not args.no_texture
    if use_texture:
        texture_data = (
            load_texture(args.texture)
            if args.texture
            else generate_procedural_texture(
                texture_type=args.texture_type, size=args.texture_size
            )
        )
        print(
            f"加载纹理: {args.texture}"
            if args.texture
            else f"生成{args.texture_type}程序化纹理"
        )

    light_dir = None
    if not args.no_lighting:
        try:
            light_dir = np.array(
                [float(x) for x in args.light_dir.split(",")], dtype=np.float32
            )
            light_dir = light_dir / np.linalg.norm(light_dir)
        except:
            print(f"无效的光源方向格式: {args.light_dir}，使用默认值")
            light_dir = np.array([1.0, -1.0, 1.0], dtype=np.float32)
            light_dir = light_dir / np.linalg.norm(light_dir)

    render_model_with_texture(
        vertices=model_data["vertices"],
        faces=model_data["faces"],
        width=args.width,
        height=args.height,
        projection=args.projection,
        use_zbuffer=not args.no_zbuffer,
        angle=args.angle,
        focal_length=args.focal,
        model_name=args.output,
        colorize=args.colorize,
        render_depth=not args.no_depth,
        output_dir=args.output_dir,
        texture_data=texture_data,
        use_texture=use_texture,
        texcoords=model_data.get("texcoords"),
        face_texcoord_indices=model_data.get("face_texcoord_indices"),
        normals=model_data.get("normals"),
        face_normal_indices=model_data.get("face_normal_indices"),
        materials=model_data.get("materials"),
        material_indices=model_data.get("material_indices"),
        depth_range_percentile=(args.depth_min, args.depth_max),
        use_lighting=not args.no_lighting,
        light_model=args.light_model,
        ambient=args.ambient,
        diffuse=args.diffuse,
        specular=args.specular,
        shininess=args.shininess,
        light_dir=light_dir,
    )
    print("渲染完成！")


if __name__ == "__main__":
    main()
