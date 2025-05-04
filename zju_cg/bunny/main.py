import numpy as np
import os
import argparse
import time
import taichi as ti
from PIL import Image
import gc
from scipy import ndimage  # For depth map median filter

# --- Import from local modules ---
from args_setup import setup_parser
from renderer import TriangleRenderer
from loaders import load_obj_enhanced
from texture_utils import load_texture, generate_procedural_texture
from transformations import (
    ndc_to_pixel,
    rotate_model,
    orthographic_projection,
    perspective_projection,
)
from camera import Camera
from color_utils import get_face_color, apply_colormap_jet
from lighting import calculate_light_position


# --- Taichi Initialization ---
# Initialize Taichi once when the script starts
# try:
#     ti.init(arch=ti.gpu, device_memory_fraction=0.8, log_level=ti.INFO)
#     print("Taichi initialized on GPU.")
# except Exception as e:
#     print(f"GPU initialization failed: {e}. Trying CPU...")
#     try:
#         ti.init(arch=ti.cpu, log_level=ti.INFO)
#         print("Taichi initialized on CPU.")
#     except Exception as e_cpu:
#         print(f"CPU initialization also failed: {e_cpu}")
#         exit(1)

# Just use cpu whatever for simplicity

ti.init(arch=ti.cpu, log_level=ti.INFO)  # Use CPU for simplicity


# --- Main Rendering Function ---
def render_model(
    # Model data
    vertices=None,
    faces=None,
    # Render settings
    width=1200,
    height=1200,
    # Camera settings
    camera=None,
    camera_type=None,
    camera_angle=None,
    # 然后是可选参数
    texcoords=None,
    face_texcoord_indices=None,
    normals=None,
    face_normal_indices=None,
    materials=None,
    material_indices=None,  # List of material names per face
    projection="perspective",
    use_zbuffer=True,
    angle=0,  # Rotation angle for model animation
    focal_length=2.0,
    # Output settings
    model_name="model",
    output_dir="output",
    colorize=False,
    render_depth=True,
    depth_range_percentile=(1, 99),
    # Texture settings
    texture_data=None,  # Pre-loaded numpy texture array
    use_texture=False,
    # Lighting settings
    use_lighting=True,
    light_model="blinn-phong",
    ambient=0.2,
    diffuse=0.6,
    specular=0.2,
    shininess=32.0,
    light_type="directional",
    light_dir=None,  # Directional light vector
    light_pos=None,  # Point light position
    light_attenuation=(1.0, 0.09, 0.032),
):
    """
    Renders the loaded model data with specified settings.
    Handles vertex processing, lighting, calling the Taichi renderer, and saving output.
    """
    start_time = time.time()
    os.makedirs(output_dir, exist_ok=True)
    color_dir = os.path.join(output_dir, "color")
    depth_dir = os.path.join(output_dir, "depth")
    os.makedirs(color_dir, exist_ok=True)
    if render_depth and use_zbuffer:
        os.makedirs(depth_dir, exist_ok=True)

    print("-" * 20)
    print(
        f"开始渲染: {model_name} (相机类型: {camera_type}, 角度: {camera_angle}°, 投影: {projection})"
    )
    print(
        f"设置: 纹理={'启用' if use_texture else '禁用'}, 光照={'启用' if use_lighting else '禁用'}, ZBuffer={'启用' if use_zbuffer else '禁用'}"
    )

    if vertices is None or faces is None or vertices.size == 0 or faces.size == 0:
        print("错误: 顶点或面数据无效或为空。无法渲染。")
        return None, None
    if camera is None:
        camera = Camera(
            look_from=np.array([0.0, 0.0, 3.0], dtype=np.float32),
            look_at=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            up=np.array([0.0, 1.0, 0.0], dtype=np.float32),
            aspect_ratio=width / height,
            fov=45.0,
        )

    # --- 1. Vertex Processing ---
    # a. Rotate (Model space)
    # rotated_vertices = rotate_model(vertices.astype(np.float64), angle).astype(
    #     np.float32
    # )  # Use higher precision for rotation # No longer needed

    # b. Normalize and Center (World space / View space preparation)
    # Move model to origin, scale to fit roughly within NDC cube later
    if angle != 0:
        model_vertices = rotate_model(vertices.astype(np.float64), angle).astype(
            np.float32
        )
    else:
        model_vertices = vertices.astype(np.float32)

    center = np.mean(vertices, axis=0)
    centered_vertices = vertices - center
    max_extent = np.max(np.abs(centered_vertices))
    if max_extent > 1e-6:
        world_vertices = centered_vertices / max_extent * 0.8
    else:
        world_vertices = centered_vertices

    # c. Position in front of camera (View space Z adjustment)
    # Ensure the model is in front of the near plane (z < 0)
    if camera_type and camera_angle != 0:
        if camera_type == "yaw":  # 左右摇头
            camera.yaw(camera_angle)
        elif camera_type == "pitch":  # 抬头低头
            camera.pitch(camera_angle)
        elif camera_type == "roll":  # 歪头
            camera.roll(camera_angle)
        elif camera_type == "orbit":  # 环绕物体
            camera.orbit(camera_angle)
        print(f"相机位置: {camera.look_from}, 观察点: {camera.look_at}")

    # d. 使用相机的变换矩阵将顶点从世界空间转换到NDC空间
    # 同时返回视图空间顶点坐标用于Z插值
    ndc_vertices, view_space_vertices = camera.transform_vertices(
        model_vertices, projection
    )

    # e. 视口变换 (NDC到像素空间)
    pixel_vertices = ndc_to_pixel(ndc_vertices, width, height)

    # 提取渲染所需的顶点数据
    vertices_x_ti = pixel_vertices[:, 0].astype(np.float32)
    vertices_y_ti = pixel_vertices[:, 1].astype(np.float32)
    # 使用视图空间的Z值用于深度插值
    z_values_ti = view_space_vertices[:, 2].astype(np.float32)

    # 输出渲染信息
    print(f"模型中心: {center}, 缩放因子: {0.8/max_extent:.3f}")
    print(f"视图空间 Z 范围: [{np.min(z_values_ti):.3f}, {np.max(z_values_ti):.3f}]")

    # 这里投影步骤已经在相机系统中完成，无需重复计算
    is_perspective = projection == "perspective"

    # --- 2. Prepare Data for Renderer ---
    renderer = TriangleRenderer(width, height)  # Create renderer instance

    # 请注意：这里只使用了一次Z值设置，而不是重复设置两次
    # Extract data needed by Taichi kernels - 使用相机已经计算好的数据
    vertices_x_ti = pixel_vertices[:, 0].astype(np.float32)
    vertices_y_ti = pixel_vertices[:, 1].astype(np.float32)
    # 使用视图空间z值进行深度插值
    z_values_ti = view_space_vertices[:, 2].astype(np.float32)

    n_faces = len(faces)
    face_colors_r = np.zeros(n_faces, dtype=np.float32)
    face_colors_g = np.zeros(n_faces, dtype=np.float32)
    face_colors_b = np.zeros(n_faces, dtype=np.float32)

    # --- 3. Face Processing (Color determination, Lighting) ---
    print("处理面片颜色和光照...")
    valid_material_map = isinstance(materials, dict)
    valid_material_indices = (
        isinstance(material_indices, list) and len(material_indices) == n_faces
    )

    # Normalize light direction if provided
    if use_lighting and light_type == "directional" and light_dir is not None:
        norm = np.linalg.norm(light_dir)
        if norm > 1e-6:
            light_direction_normalized = (
                -light_dir / norm
            )  # Direction *towards* the light source
        else:
            light_direction_normalized = np.array(
                [0.0, 0.0, -1.0], dtype=np.float32
            )  # Default fallback
    else:
        light_direction_normalized = np.array(
            [0.0, 0.0, -1.0], dtype=np.float32
        )  # Default for non-directional or missing

    # Camera view direction (simplified: assumes camera at origin looking down -Z)
    view_dir = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    # Pre-calculate light attenuation factors
    att_const, att_linear, att_quad = light_attenuation

    # Process each face
    for idx in range(n_faces):
        face = faces[idx]
        # Get base color (from random, material, or default)
        base_color = get_face_color(idx, colorize)  # Start with random/default

        # Try getting color from material if available
        if not colorize and valid_material_map and valid_material_indices:
            mat_name = material_indices[idx]
            if mat_name and mat_name in materials:
                material = materials[mat_name]
                if "Kd" in material:  # Use diffuse color from material
                    base_color = np.array(material["Kd"], dtype=np.float32)
                # Could also use Ka for ambient base color if needed

        # --- Apply Lighting (if enabled) ---
        final_color = base_color
        if use_lighting and normals is not None and face_normal_indices is not None:
            # Get vertex normals for the face
            try:
                n_indices = face_normal_indices[idx]
                # Ensure indices are valid
                if np.all(n_indices >= 0) and np.all(n_indices < len(normals)):
                    # Smooth shading: Average vertex normals (approximation at face center)
                    # For better quality, compute lighting per vertex and interpolate color
                    face_normal = np.mean(normals[n_indices], axis=0)
                    norm = np.linalg.norm(face_normal)
                    if norm > 1e-6:
                        face_normal /= norm
                    else:
                        face_normal = np.array([0.0, 0.0, 1.0])  # Default if degenerate
                else:
                    # print(f"警告: 面 {idx} 的法线索引无效: {n_indices}。使用默认法线。")
                    face_normal = np.array([0.0, 0.0, 1.0])  # Default normal
            except IndexError:
                # print(f"警告: 面 {idx} 的法线索引越界。使用默认法线。")
                face_normal = np.array([0.0, 0.0, 1.0])

            # Get face center in view space for point light calculations
            face_center_view = np.mean(view_space_vertices[face], axis=0)

            # --- Calculate Lighting Components ---
            # 1. Ambient Component
            ambient_comp = ambient * base_color

            # 2. Diffuse and Specular Components (depend on light type)
            diffuse_comp = np.zeros(3, dtype=np.float32)
            specular_comp = np.zeros(3, dtype=np.float32)
            current_light_dir = light_direction_normalized  # Default for directional
            attenuation = 1.0  # Default for directional

            if light_type == "point" and light_pos is not None:
                # Calculate direction from face center *to* light source
                light_vec = light_pos - face_center_view
                distance = np.linalg.norm(light_vec)
                if distance > 1e-6:
                    current_light_dir = light_vec / distance
                    # Calculate attenuation for point light
                    attenuation = 1.0 / (
                        att_const + att_linear * distance + att_quad * (distance**2)
                    )
                    attenuation = max(0.0, attenuation)  # Ensure non-negative
                else:
                    current_light_dir = np.array([0, 0, 1])  # Avoid division by zero
                    attenuation = 1.0  # Max intensity if at the exact point

            # Calculate Diffuse term (Lambertian)
            # N dot L (use max to clamp negative values)
            N_dot_L = max(0.0, np.dot(face_normal, current_light_dir))
            diffuse_comp = diffuse * N_dot_L * base_color

            # Calculate Specular term
            spec_strength = 0.0
            if N_dot_L > 0:  # Only calculate specular if light hits the surface
                if light_model.lower() == "phong":
                    # R = 2 * (N dot L) * N - L
                    reflect_dir = 2.0 * N_dot_L * face_normal - current_light_dir
                    # V dot R (V is view direction)
                    V_dot_R = max(0.0, np.dot(view_dir, reflect_dir))
                    spec_strength = np.power(V_dot_R, shininess)
                else:  # Default to Blinn-Phong
                    # H = normalize(L + V)
                    halfway_dir = current_light_dir + view_dir
                    norm_H = np.linalg.norm(halfway_dir)
                    if norm_H > 1e-6:
                        halfway_dir /= norm_H
                        # N dot H
                        N_dot_H = max(0.0, np.dot(face_normal, halfway_dir))
                        spec_strength = np.power(N_dot_H, shininess)

            # Use white specular color highlight by default
            specular_comp = specular * spec_strength * np.array([1.0, 1.0, 1.0])

            # Combine components with attenuation and clamp
            final_color = ambient_comp + attenuation * (diffuse_comp + specular_comp)
            final_color = np.clip(final_color, 0.0, 1.0)

        # Store final color for the face
        face_colors_r[idx] = final_color[0]
        face_colors_g[idx] = final_color[1]
        face_colors_b[idx] = final_color[2]

    # --- 4. Perform Rendering with Taichi ---
    print("调用 Taichi 渲染内核...")
    kernel_start_time = time.time()

    # Prepare texture data for the unified renderer function
    tex_args = {}
    if use_texture:
        # Check if necessary texture data components are available
        if (
            texcoords is not None
            and face_texcoord_indices is not None
            and texcoords.size > 0
            and face_texcoord_indices.size > 0
            and texture_data is not None
            and texture_data.size > 0
        ):
            tex_args = {
                "use_texture": True,
                "texcoords_u": texcoords[:, 0].astype(np.float32),
                "texcoords_v": texcoords[:, 1].astype(np.float32),
                "face_texcoord_indices": face_texcoord_indices.astype(np.int32),
                "texture_data": texture_data.astype(
                    np.float32
                ),  # Already loaded/generated
            }
        else:
            print("警告: 纹理已启用，但缺少纹理坐标、索引或数据。将禁用纹理渲染。")
            use_texture = False  # Force disable if data is missing
            tex_args["use_texture"] = False

    valid_triangle_count = renderer.render_triangles_unified(
        vertices_x=vertices_x_ti,
        vertices_y=vertices_y_ti,
        faces=faces.astype(np.int32),  # Ensure int32 for Taichi
        z_values=z_values_ti,  # View space Z
        colors_r=face_colors_r,  # Calculated face colors (R)
        colors_g=face_colors_g,  # Calculated face colors (G)
        colors_b=face_colors_b,  # Calculated face colors (B)
        face_count=n_faces,
        is_perspective=1 if is_perspective else 0,
        use_zbuffer=1 if use_zbuffer else 0,
        **tex_args,  # Pass texture arguments dictionary
    )
    ti.sync()  # Ensure kernel finishes before proceeding
    kernel_end_time = time.time()
    print(f"Taichi 内核执行时间: {kernel_end_time - kernel_start_time:.3f} 秒")

    # --- 5. Post-processing and Saving ---
    print("后处理和保存图像...")
    color_buffer = renderer.get_color_array()
    depth_buffer = renderer.get_depth_array() if use_zbuffer else None
    total_render_time = time.time() - start_time
    print(
        f"渲染了 {valid_triangle_count}/{n_faces} 个有效三角形"
        f" (总时间: {total_render_time:.2f} 秒)"
    )

    # Save color image
    # Convert float [0, 1] buffer to uint8 [0, 255]
    img_array = (np.clip(color_buffer, 0.0, 1.0) * 255).astype(np.uint8)
    img = Image.fromarray(img_array, "RGB")  # Assuming RGB format
    color_path = os.path.join(color_dir, f"{model_name}.png")
    try:
        img.save(color_path)
        print(f"保存彩色图像: {color_path}")
    except Exception as e:
        print(f"错误：无法保存彩色图像到 {color_path}: {e}")

    # Save depth image (if enabled and buffer exists)
    depth_path = None
    if use_zbuffer and render_depth and depth_buffer is not None:
        # Process depth buffer: normalize and colorize
        # Replace inf with NaN for percentile calculation
        finite_depth = np.where(np.isinf(depth_buffer), np.nan, depth_buffer)
        mask_valid = ~np.isnan(finite_depth)

        if np.any(mask_valid):
            valid_depths = finite_depth[mask_valid]
            # Use robust percentiles to determine normalization range
            min_depth = float(np.percentile(valid_depths, depth_range_percentile[0]))
            max_depth = float(np.percentile(valid_depths, depth_range_percentile[1]))

            # Avoid division by zero if range is too small
            if abs(max_depth - min_depth) < 1e-6:
                max_depth = min_depth + 1.0  # Create a small range

            depth_range = max_depth - min_depth
            print(
                f"深度范围 (百分位 {depth_range_percentile[0]}-{depth_range_percentile[1]}): [{min_depth:.3f}, {max_depth:.3f}]"
            )

            # Normalize depth values within the calculated range [0, 1]
            # Clamp values outside the percentile range
            normalized_depth = np.full_like(finite_depth, np.nan)  # Start with NaN
            normalized_depth[mask_valid] = (
                finite_depth[mask_valid] - min_depth
            ) / depth_range
            normalized_depth = np.clip(normalized_depth, 0.0, 1.0)  # Clip to [0, 1]

            # Fill NaN values (original infinities or outside range) with a default (e.g., 1.0 for farthest)
            normalized_depth = np.nan_to_num(normalized_depth, nan=1.0)

            # Optional: Apply median filter to reduce noise/aliasing in depth map
            filter_size = 2
            normalized_depth = ndimage.median_filter(normalized_depth, size=filter_size)

            # Apply colormap (invert: closer is hotter/brighter -> 1.0 - normalized)
            depth_colored = apply_colormap_jet(1.0 - normalized_depth)

            depth_img = Image.fromarray(depth_colored, "RGB")
            depth_path = os.path.join(depth_dir, f"{model_name}.png")
            try:
                depth_img.save(depth_path)
                print(f"保存深度图: {depth_path}")
            except Exception as e:
                print(f"错误：无法保存深度图像到 {depth_path}: {e}")
        else:
            print("深度缓冲区中没有有效值，跳过深度图生成。")

    # --- Cleanup ---
    del renderer  # Explicitly delete renderer object if needed
    gc.collect()  # Suggest garbage collection
    print("-" * 20)

    return color_path, depth_path


# --- Main Execution Block ---
if __name__ == "__main__":
    # --- Check Dependencies ---
    try:
        import numpy
        import taichi
        from PIL import Image
        from scipy import ndimage
    except ImportError as e:
        print(f"错误: 缺少必要的 Python 库: {e}")
        print("请运行: pip install numpy taichi Pillow scipy")
        exit(1)

    # --- Parse Arguments ---
    parser = setup_parser()
    args = parser.parse_args()

    # --- Validate Inputs ---
    if not os.path.exists(args.obj):
        print(f"错误: OBJ 文件未找到: {args.obj}")
        exit(1)

    # --- Load Model ---
    print(f"加载模型: {args.obj}")
    model_data = load_obj_enhanced(
        args.obj,
        load_texcoords=not args.no_texture,
        load_normals=not args.no_lighting,  # Normals needed for lighting
        load_materials=(
            not args.no_materials and not args.colorize
        ),  # Load materials if needed and not overridden by colorize
    )
    if model_data is None:
        print("模型加载失败。")
        exit(1)

    # --- Load or Generate Texture ---
    texture_data = None
    use_texture_flag = not args.no_texture
    if use_texture_flag:
        if args.texture:  # Prioritize loading texture from file
            texture_data = load_texture(args.texture)
            if (
                texture_data is None
                or texture_data.shape[0] <= 1
                or texture_data.shape[1] <= 1
            ):
                print("警告: 从文件加载纹理失败或纹理无效。尝试生成程序化纹理。")
                texture_data = (
                    None  # Reset texture data to trigger procedural generation
                )
        # If file texture wasn't loaded or provided, try generating procedural
        if texture_data is None:
            if (
                model_data.get("texcoords") is None
                or model_data.get("face_texcoord_indices") is None
            ):
                print("警告: 无法生成程序化纹理，因为模型缺少纹理坐标。已禁用纹理。")
                use_texture_flag = False
            else:
                print(
                    f"生成 {args.texture_size}x{args.texture_size} '{args.texture_type}' 程序化纹理..."
                )
                texture_data = generate_procedural_texture(
                    texture_type=args.texture_type, size=args.texture_size
                )

    # --- Setup Lighting ---
    light_dir_vec = None
    light_pos_vec = None
    light_atten_vec = [1.0, 0.09, 0.032]  # Default attenuation

    if not args.no_lighting:
        # Parse light direction
        try:
            light_dir_vec = np.array(
                [float(x) for x in args.light_dir.split(",")], dtype=np.float32
            )
            if light_dir_vec.shape != (3,):
                raise ValueError("需要3个分量")
        except Exception as e:
            print(
                f"无效的方向光方向格式 '{args.light_dir}': {e}. 使用默认值 [1, -1, 1]."
            )
            light_dir_vec = np.array([1.0, -1.0, 1.0], dtype=np.float32)

        # Parse base light position
        try:
            base_light_pos = np.array(
                [float(x) for x in args.light_pos.split(",")], dtype=np.float32
            )
            if base_light_pos.shape != (3,):
                raise ValueError("需要3个分量")
        except Exception as e:
            print(
                f"无效的点光源位置格式 '{args.light_pos}': {e}. 使用默认值 [0, 0, 3]."
            )
            base_light_pos = np.array([0.0, 0.0, 3.0], dtype=np.float32)

        # Parse light attenuation
        try:
            light_atten_vec = [float(x) for x in args.light_atten.split(",")]
            if len(light_atten_vec) != 3:
                raise ValueError("需要3个系数")
            if any(x < 0 for x in light_atten_vec):
                raise ValueError("衰减系数不能为负")
        except Exception as e:
            print(
                f"无效的衰减系数格式 '{args.light_atten}': {e}. 使用默认值 [1.0, 0.09, 0.032]."
            )
            light_atten_vec = [1.0, 0.09, 0.032]

        # Calculate animated light position if applicable (for the current frame)
        light_pos_vec = base_light_pos.copy()  # Start with base position
        if args.light_animation != "none" and args.total_frames > 1:
            # Ensure frame number is within bounds [0, total_frames - 1]
            current_frame = max(0, min(args.light_frame, args.total_frames - 1))
            # Normalized time t from 0 to 1
            t = (
                float(current_frame) / max(1, args.total_frames - 1)
                if args.total_frames > 1
                else 0.0
            )

            custom_params = None
            if args.light_animation == "custom":
                custom_params = {
                    "x_expr": args.custom_x_expr,
                    "y_expr": args.custom_y_expr,
                    "z_expr": args.custom_z_expr,
                }

            light_pos_vec = calculate_light_position(
                base_light_pos,
                args.light_animation,
                t,
                args.light_range,
                custom_params,
            )
            print(
                f"帧 {current_frame}/{args.total_frames-1} (t={t:.3f}): 计算光源位置 = {np.round(light_pos_vec, 3)}"
            )
        elif args.light_type == "point":
            print(f"点光源位置 (静态): {np.round(light_pos_vec, 3)}")
        elif args.light_type == "directional":
            print(f"方向光方向: {np.round(light_dir_vec, 3)}")

    animation_type = getattr(args, "animation_type", "model")
    camera_type = getattr(args, "camera_type", "orbit")
    frame = getattr(args, "frame", 0)
    camera_fov = getattr(args, "camera_fov", 45.0)

    # 解析相机位置
    try:
        camera_from = np.array(
            [float(x) for x in getattr(args, "camera_from", "0,0,3").split(",")],
            dtype=np.float32,
        )
    except:
        camera_from = np.array([0.0, 0.0, 3.0], dtype=np.float32)

    # 解析观察点
    try:
        camera_at = np.array(
            [float(x) for x in getattr(args, "camera_at", "0,0,0").split(",")],
            dtype=np.float32,
        )
    except:
        camera_at = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    # 解析上方向
    try:
        camera_up = np.array(
            [float(x) for x in getattr(args, "camera_up", "0,1,0").split(",")],
            dtype=np.float32,
        )
    except:
        camera_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    # 创建相机
    camera = Camera(
        look_from=camera_from,
        look_at=camera_at,
        up=camera_up,
        fov=camera_fov,
        aspect_ratio=args.width / args.height,
    )

    # 计算相机动画角度 (-180到180度)
    camera_angle = 0
    if animation_type == "camera" and args.total_frames > 1:
        current_frame = max(0, min(frame, args.total_frames - 1))
        # 根据当前帧计算角度范围
        if camera_type in ["yaw", "pitch", "roll"]:
            angle_range = 60.0  # 摇头、抬头和歪头使用±30度范围
            camera_angle = (
                current_frame / (args.total_frames - 1) * angle_range
            ) - angle_range / 2
        else:  # orbit环绕默认使用360度
            angle_range = 360.0
            camera_angle = current_frame / (args.total_frames - 1) * angle_range

        print(
            f"相机动画 '{camera_type}': 帧 {current_frame}/{args.total_frames-1}, 角度: {camera_angle:.1f}°"
        )

    # --- Perform Rendering ---
    render_model(
        # Model data unpacked from dictionary
        vertices=model_data.get("vertices"),
        faces=model_data.get("faces"),
        texcoords=model_data.get("texcoords"),
        face_texcoord_indices=model_data.get("face_texcoord_indices"),
        normals=model_data.get("normals"),
        face_normal_indices=model_data.get("face_normal_indices"),
        materials=model_data.get("materials"),
        material_indices=model_data.get("material_indices"),
        # Render settings from args
        width=args.width,
        height=args.height,
        camera=camera,
        camera_type=camera_type if animation_type == "camera" else None,
        camera_angle=camera_angle,
        projection=args.projection,
        use_zbuffer=not args.no_zbuffer,
        angle=args.angle,  # Current angle for model rotation
        focal_length=args.focal,
        # Output settings
        model_name=args.output,  # Base name for the output file
        output_dir=args.output_dir,
        colorize=args.colorize,
        render_depth=not args.no_depth,
        depth_range_percentile=(args.depth_min, args.depth_max),
        # Texture
        texture_data=texture_data,  # Pass loaded/generated texture
        use_texture=use_texture_flag,
        # Lighting
        use_lighting=not args.no_lighting,
        light_model=args.light_model,
        ambient=args.ambient,
        diffuse=args.diffuse,
        specular=args.specular,
        shininess=args.shininess,
        light_type=args.light_type,
        light_dir=light_dir_vec,  # Parsed directional vector
        light_pos=light_pos_vec,  # Parsed and potentially animated position
        light_attenuation=light_atten_vec,  # Parsed attenuation factors
    )

    print("渲染脚本执行完毕！")
