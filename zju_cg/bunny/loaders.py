import numpy as np
import os


def load_mtl(mtl_filename):
    """加载并解析 .mtl 文件"""
    materials = {}
    current_material = None
    if not os.path.exists(mtl_filename):
        print(f"警告: 材质文件未找到: {mtl_filename}")
        return materials

    try:
        with open(mtl_filename, "r") as f:
            for line in f:
                tokens = line.strip().split()
                if not tokens or line.startswith("#"):
                    continue
                if tokens[0] == "newmtl":
                    current_material = " ".join(tokens[1:])
                    # 设置默认值
                    materials[current_material] = {
                        "Ka": [0.2, 0.2, 0.2],  # Ambient
                        "Kd": [0.8, 0.8, 0.8],  # Diffuse
                        "Ks": [0.0, 0.0, 0.0],  # Specular
                        "Ns": 10.0,  # Shininess
                        "d": 1.0,  # Dissolve (alpha)
                        "map_Kd": None,  # Diffuse texture map
                        "map_Bump": None,  # Bump map (can be used for normals)
                        # Add other maps as needed (e.g., map_Ks, map_Ka, map_Ns)
                    }
                elif current_material:  # Ensure a material context exists
                    key = tokens[0]
                    values = tokens[1:]
                    try:
                        if key in ["Ka", "Kd", "Ks"]:
                            materials[current_material][key] = [
                                float(x) for x in values[:3]
                            ]
                        elif key in ["Ns", "d", "Ni"]:  # Ni = optical density
                            materials[current_material][key] = float(values[0])
                        elif key.startswith("map_"):
                            # Store the texture filename relative to the MTL file's directory
                            texture_path = " ".join(values)
                            materials[current_material][key] = os.path.join(
                                os.path.dirname(mtl_filename), texture_path
                            )
                        # Add more properties as needed (e.g., illum)
                    except (ValueError, IndexError) as e:
                        print(
                            f"警告: 解析材质 '{current_material}' 属性 '{key}' 时出错: {e} - 行: {line.strip()}"
                        )

    except Exception as e:
        print(f"错误: 无法加载或解析材质文件 {mtl_filename}: {e}")

    return materials


def generate_default_texcoords(vertices):
    """根据顶点位置生成默认的球面纹理坐标"""
    center = np.mean(vertices, axis=0)
    vertices_centered = vertices - center
    radii = np.linalg.norm(vertices_centered, axis=1)
    max_radius = np.max(radii)

    if max_radius < 1e-6:  # Avoid division by zero for single point or flat models
        print("警告: 模型半径接近零，无法生成球面纹理坐标。返回零坐标。")
        return np.zeros((len(vertices), 2), dtype=np.float32)

    # Normalize vectors (handle potential zero-length vectors after centering)
    # Add a small epsilon to avoid division by zero for vertices at the center
    vertices_normalized = vertices_centered / (radii[:, np.newaxis] + 1e-9)

    # Spherical coordinates (phi = azimuth, theta = inclination)
    # atan2(x, z) for azimuth (longitude), range [-pi, pi] -> u [0, 1]
    # acos(y) for inclination (latitude), range [0, pi] -> v [0, 1]
    phi = np.arctan2(vertices_normalized[:, 0], vertices_normalized[:, 2])  # x, z
    theta = np.arccos(np.clip(vertices_normalized[:, 1], -1.0, 1.0))  # y

    u = (phi / (2 * np.pi)) + 0.5
    v = theta / np.pi

    texcoords = np.stack((u, v), axis=-1).astype(np.float32)
    return texcoords


def generate_vertex_normals(vertices, faces):
    """通过平均面法线生成平滑的顶点法线"""
    vertex_normals = np.zeros_like(vertices, dtype=np.float32)
    face_normals = np.zeros((len(faces), 3), dtype=np.float32)

    # 计算每个三角面片的法线
    for i, face in enumerate(faces):
        # 确保面是有效的（至少3个顶点）
        if len(face) >= 3:
            v0 = vertices[face[0]]
            v1 = vertices[face[1]]
            v2 = vertices[face[2]]
            # 计算叉积得到面法线 (v1-v0) x (v2-v0)
            normal = np.cross(v1 - v0, v2 - v0)
            norm = np.linalg.norm(normal)
            if norm > 1e-10:
                face_normals[i] = normal / norm
            # else: face normal remains [0, 0, 0] for degenerate triangles

    # 将面法线累加到共享该法线的顶点上
    # 使用 numpy indexing 提高效率
    for i, face in enumerate(faces):
        if len(face) >= 3:
            # 使用 add.at 来安全地对重复索引进行累加
            np.add.at(vertex_normals, face[0], face_normals[i])
            np.add.at(vertex_normals, face[1], face_normals[i])
            np.add.at(vertex_normals, face[2], face_normals[i])

    # 归一化顶点法线
    norms = np.linalg.norm(vertex_normals, axis=1)
    # 避免除以零
    valid_norms_mask = norms > 1e-10
    vertex_normals[valid_norms_mask] /= norms[valid_norms_mask, np.newaxis]

    # 对于法线为零的顶点（可能未被任何有效面片引用），可以设置一个默认法线
    zero_norm_mask = ~valid_norms_mask
    if np.any(zero_norm_mask):
        print(f"警告: {np.sum(zero_norm_mask)} 个顶点的法线为零，设置为 [0, 1, 0]")
        vertex_normals[zero_norm_mask] = [0.0, 1.0, 0.0]  # Default up vector

    return vertex_normals


def load_obj_enhanced(
    filename, load_texcoords=True, load_normals=True, load_materials=True
):
    """
    增强版 OBJ 加载器，支持纹理坐标、法线、材质，并处理缺失数据和四边形。
    """
    vertices, faces = [], []
    texcoords = [] if load_texcoords else None
    normals = [] if load_normals else None
    face_vertex_indices = []
    face_texcoord_indices = [] if load_texcoords else None
    face_normal_indices = [] if load_normals else None

    materials = {} if load_materials else None
    material_indices = (
        [] if load_materials else None
    )  # Index of material per *output* face
    current_material_name = None  # Name of the active material
    mtl_filename = None

    obj_dir = os.path.dirname(filename)

    try:
        with open(filename, "r") as f:
            for line in f:
                tokens = line.strip().split()
                if not tokens or line.startswith("#"):
                    continue

                if tokens[0] == "v":
                    try:
                        vertices.append([float(x) for x in tokens[1:4]])
                    except (ValueError, IndexError):
                        print(f"警告: 忽略格式错误的顶点行: {line.strip()}")
                elif tokens[0] == "vt" and load_texcoords:
                    try:
                        # OBJ format can have 1, 2, or 3 components for texcoords (u, v, w)
                        # We primarily use u, v. Handle cases with fewer than 2.
                        u = float(tokens[1]) if len(tokens) > 1 else 0.0
                        v = float(tokens[2]) if len(tokens) > 2 else 0.0
                        # Optional w = float(tokens[3]) if len(tokens) > 3 else 0.0
                        texcoords.append([u, v])
                    except (ValueError, IndexError):
                        print(f"警告: 忽略格式错误的纹理坐标行: {line.strip()}")
                elif tokens[0] == "vn" and load_normals:
                    try:
                        normals.append([float(x) for x in tokens[1:4]])
                    except (ValueError, IndexError):
                        print(f"警告: 忽略格式错误的法线行: {line.strip()}")
                elif tokens[0] == "f":
                    # Parse face indices (v/vt/vn)
                    face_v, face_vt, face_vn = [], [], []
                    valid_face = True
                    for v_spec in tokens[1:]:
                        parts = v_spec.split("/")
                        try:
                            # Vertex index (always present)
                            v_idx = int(parts[0])
                            # OBJ indices are 1-based, convert to 0-based
                            face_v.append(
                                v_idx - 1 if v_idx > 0 else len(vertices) + v_idx
                            )

                            # Texture coordinate index (optional)
                            if load_texcoords:
                                if len(parts) > 1 and parts[1]:
                                    vt_idx = int(parts[1])
                                    face_vt.append(
                                        vt_idx - 1
                                        if vt_idx > 0
                                        else len(texcoords) + vt_idx
                                    )
                                else:
                                    face_vt.append(
                                        -1
                                    )  # Indicate missing texcoord index

                            # Normal index (optional)
                            if load_normals:
                                if len(parts) > 2 and parts[2]:
                                    vn_idx = int(parts[2])
                                    face_vn.append(
                                        vn_idx - 1
                                        if vn_idx > 0
                                        else len(normals) + vn_idx
                                    )
                                else:
                                    face_vn.append(-1)  # Indicate missing normal index
                        except (ValueError, IndexError):
                            print(
                                f"警告: 忽略格式错误的面定义部分 '{v_spec}' 在行: {line.strip()}"
                            )
                            valid_face = False
                            break
                    if not valid_face:
                        continue

                    # Triangulate polygons (common practice: fan triangulation from first vertex)
                    if len(face_v) >= 3:
                        for i in range(1, len(face_v) - 1):
                            # Add triangle (v0, vi, vi+1)
                            faces.append([face_v[0], face_v[i], face_v[i + 1]])
                            if load_texcoords:
                                face_texcoord_indices.append(
                                    [face_vt[0], face_vt[i], face_vt[i + 1]]
                                )
                            if load_normals:
                                face_normal_indices.append(
                                    [face_vn[0], face_vn[i], face_vn[i + 1]]
                                )
                            if load_materials:
                                material_indices.append(current_material_name)
                    else:
                        print(f"警告: 忽略少于3个顶点的面: {line.strip()}")

                elif tokens[0] == "mtllib" and load_materials:
                    mtl_filename = " ".join(tokens[1:])
                    mtl_path = os.path.join(obj_dir, mtl_filename)
                    materials.update(
                        load_mtl(mtl_path)
                    )  # Use update to merge if multiple mtllibs
                elif tokens[0] == "usemtl" and load_materials:
                    current_material_name = " ".join(tokens[1:])
                    if current_material_name not in materials:
                        print(
                            f"警告: 使用了未在 '{mtl_filename}' 中定义的材质 '{current_material_name}'"
                        )
                        # Optionally create a default material entry here
                        # materials[current_material_name] = { ... default values ... }

    except FileNotFoundError:
        print(f"错误: OBJ 文件未找到: {filename}")
        return None
    except Exception as e:
        print(f"错误: 读取 OBJ 文件时发生错误 {filename}: {e}")
        return None

    vertices_np = np.array(vertices, dtype=np.float32)
    faces_np = np.array(faces, dtype=np.int32)

    result = {
        "vertices": vertices_np,
        "faces": faces_np,
    }

    # --- Post-processing and Validation ---

    # Texture Coordinates
    if load_texcoords:
        if texcoords:
            texcoords_np = np.array(texcoords, dtype=np.float32)
            face_texcoord_indices_np = np.array(face_texcoord_indices, dtype=np.int32)
            # Check if any face is missing texcoord indices assigned during parsing
            if np.any(face_texcoord_indices_np == -1):
                print("警告: 部分面缺少纹理坐标索引。")
                # Option 1: Assign default (e.g., index 0) if texcoords exist
                # face_texcoord_indices_np[face_texcoord_indices_np == -1] = 0
                # Option 2: Fallback to generating default texcoords if too many are missing
                # Option 3: Leave as -1 and handle downstream (e.g., disable texture for that face)
                # Current approach: Rely on downstream check for valid indices per face
            result["texcoords"] = texcoords_np
            result["face_texcoord_indices"] = face_texcoord_indices_np
        elif vertices_np.size > 0:  # Only generate if vertices exist
            print("警告: OBJ 文件中未找到纹理坐标 (vt)，将生成默认球面映射。")
            texcoords_np = generate_default_texcoords(vertices_np)
            # Assign texcoord indices based on vertex indices for each face
            # This assumes one texcoord per vertex position
            face_texcoord_indices_np = faces_np.copy()
            result["texcoords"] = texcoords_np
            result["face_texcoord_indices"] = face_texcoord_indices_np
        else:  # No vertices, cannot generate texcoords
            result["texcoords"] = np.empty((0, 2), dtype=np.float32)
            result["face_texcoord_indices"] = np.empty((0, 3), dtype=np.int32)

    # Normals
    if load_normals:
        if normals:
            normals_np = np.array(normals, dtype=np.float32)
            # Normalize loaded normals just in case
            norms = np.linalg.norm(normals_np, axis=1)
            valid_norms_mask = norms > 1e-10
            normals_np[valid_norms_mask] /= norms[valid_norms_mask, np.newaxis]

            face_normal_indices_np = np.array(face_normal_indices, dtype=np.int32)
            if np.any(face_normal_indices_np == -1):
                print("警告: 部分面缺少法线索引。")
                # Handle missing indices similarly to texcoords
            result["normals"] = normals_np
            result["face_normal_indices"] = face_normal_indices_np
        elif (
            vertices_np.size > 0 and faces_np.size > 0
        ):  # Only generate if geometry exists
            print("警告: OBJ 文件中未找到顶点法线 (vn)，将生成平滑法线。")
            normals_np = generate_vertex_normals(vertices_np, faces_np)
            # Assign normal indices based on vertex indices
            face_normal_indices_np = faces_np.copy()
            result["normals"] = normals_np
            result["face_normal_indices"] = face_normal_indices_np
        else:  # No geometry, cannot generate normals
            result["normals"] = np.empty((0, 3), dtype=np.float32)
            result["face_normal_indices"] = np.empty((0, 3), dtype=np.int32)

    # Materials
    if load_materials:
        result["materials"] = materials
        # Ensure material_indices array has the correct length (one per output triangle face)
        if len(material_indices) != len(faces_np):
            print(
                f"警告: 材质索引数量 ({len(material_indices)}) 与三角面片数量 ({len(faces_np)}) 不匹配。可能由多边形细分引起。材质将不可靠。"
            )
            # Fallback: assign None or a default material index if needed downstream
            result["material_indices"] = [None] * len(
                faces_np
            )  # Or handle appropriately
        else:
            result["material_indices"] = (
                material_indices  # Store the list of material names
            )

    return result
