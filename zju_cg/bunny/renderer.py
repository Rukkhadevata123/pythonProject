import taichi as ti
import numpy as np


# --- Taichi Helper Functions (callable within kernels) ---
@ti.func
def barycentric_coordinates_ti(p, v1, v2, v3):
    """计算点 p 相对于三角形 (v1, v2, v3) 的重心坐标"""
    # 使用向量叉积计算面积
    e1 = v2 - v1
    e2 = v3 - v1
    p_v1 = p - v1

    # Area of the main triangle (times 2)
    total_area_x2 = e1[0] * e2[1] - e1[1] * e2[0]
    bary = ti.Vector([0.0, 0.0, 0.0], dt=ti.f32)  # Initialize to zero

    # Avoid division by zero for degenerate triangles
    if ti.abs(total_area_x2) > 1e-9:
        inv_total_area_x2 = 1.0 / total_area_x2

        # Area of subtriangle opposite v2 (p, v3, v1) / total_area -> bary for v2 (beta)
        area2_x2 = p_v1[0] * e2[1] - p_v1[1] * e2[0]
        beta = area2_x2 * inv_total_area_x2

        # Area of subtriangle opposite v3 (p, v1, v2) / total_area -> bary for v3 (gamma)
        area3_x2 = e1[0] * p_v1[1] - e1[1] * p_v1[0]
        gamma = area3_x2 * inv_total_area_x2

        # Bary for v1 (alpha)
        alpha = 1.0 - beta - gamma

        bary = ti.Vector([alpha, beta, gamma], dt=ti.f32)

    return bary


@ti.func
def interpolate_z_ti(bary, z1, z2, z3, is_perspective):
    """根据重心坐标插值 Z 值 (考虑透视校正)"""
    alpha, beta, gamma = bary[0], bary[1], bary[2]
    result = float("inf")  # 默认返回值为无穷大

    # 检查重心坐标是否有效 (在三角形内)
    # 由于浮点精度问题，使用一个小的epsilon值
    epsilon = 1e-5
    is_valid = (
        alpha >= -epsilon
        and beta >= -epsilon
        and gamma >= -epsilon
        and (alpha + beta + gamma <= 1.0 + epsilon)
    )

    # 计算插值Z值
    if is_valid:
        interpolated_z = 0.0  # 初始化插值Z变量
        
        if is_perspective == 0:
            # --- 正交投影: 线性插值 ---
            interpolated_z = alpha * z1 + beta * z2 + gamma * z3
        else:
            # --- 透视投影: 透视校正插值 ---
            inv_z1 = 1.0 / z1 if ti.abs(z1) > 1e-9 else 0.0  # 处理除零
            inv_z2 = 1.0 / z2 if ti.abs(z2) > 1e-9 else 0.0
            inv_z3 = 1.0 / z3 if ti.abs(z3) > 1e-9 else 0.0

            interpolated_inv_z = alpha * inv_z1 + beta * inv_z2 + gamma * inv_z3

            if ti.abs(interpolated_inv_z) > 1e-9:
                interpolated_z = 1.0 / interpolated_inv_z
            else:
                # 如果透视校正失败，回退到线性插值
                interpolated_z = alpha * z1 + beta * z2 + gamma * z3

        # Taichi z-buffer通常存储正深度值，
        # 而视空间Z为负值。取反用于比较。
        result = -interpolated_z

    # 在函数末尾使用单一的return语句
    return result


# --- Triangle Renderer Class ---
@ti.data_oriented
class TriangleRenderer:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        # Depth buffer: stores the *negative* of the view space Z (closer points have smaller positive values)
        self.depth_buffer = ti.field(dtype=ti.f32, shape=(height, width))
        # Color buffer: stores RGB color values [0, 1]
        self.color_buffer = ti.Vector.field(3, dtype=ti.f32, shape=(height, width))
        self.clear_buffers()

    @ti.kernel
    def clear_buffers(self):
        """重置颜色和深度缓冲区"""
        for i, j in self.depth_buffer:
            self.depth_buffer[i, j] = float(
                "inf"
            )  # Initialize depth to positive infinity
            self.color_buffer[i, j] = ti.Vector(
                [0.0, 0.0, 0.0]
            )  # Initialize color to black

    @ti.kernel
    def render_triangles_unified_without_texture(
        self,
        vertices_x: ti.types.ndarray(),  # Pixel space X
        vertices_y: ti.types.ndarray(),  # Pixel space Y
        faces: ti.types.ndarray(),  # Face indices (n_faces, 3)
        z_values: ti.types.ndarray(),  # Original view space Z for vertices (usually negative)
        colors_r: ti.types.ndarray(),  # Per-face color R [0,1] (n_faces,)
        colors_g: ti.types.ndarray(),  # Per-face color G [0,1] (n_faces,)
        colors_b: ti.types.ndarray(),  # Per-face color B [0,1] (n_faces,)
        face_count: ti.i32,
        is_perspective: ti.i32,  # 1 if perspective, 0 if orthographic
        use_zbuffer: ti.i32,  # 1 if z-buffer test enabled, 0 otherwise
    ) -> ti.i32:
        """Taichi kernel for rendering triangles without textures."""
        valid_triangles_rendered = 0
        ti.loop_config(block_dim=128)  # Optimize GPU execution config

        for face_idx in range(face_count):
            # 1. Get triangle vertex indices
            i0, i1, i2 = faces[face_idx, 0], faces[face_idx, 1], faces[face_idx, 2]

            # 2. Get vertex positions in pixel space
            v1_pix = ti.Vector([vertices_x[i0], vertices_y[i0]], dt=ti.f32)
            v2_pix = ti.Vector([vertices_x[i1], vertices_y[i1]], dt=ti.f32)
            v3_pix = ti.Vector([vertices_x[i2], vertices_y[i2]], dt=ti.f32)

            # 3. Get original view space Z values (needed for interpolation)
            z1_view, z2_view, z3_view = z_values[i0], z_values[i1], z_values[i2]

            # 4. Get face color
            face_color = ti.Vector(
                [colors_r[face_idx], colors_g[face_idx], colors_b[face_idx]], dt=ti.f32
            )

            # 5. Calculate bounding box in pixel coordinates
            min_x = ti.max(0, ti.floor(ti.min(v1_pix[0], v2_pix[0], v3_pix[0])))
            min_y = ti.max(0, ti.floor(ti.min(v1_pix[1], v2_pix[1], v3_pix[1])))
            max_x = ti.min(self.width, ti.ceil(ti.max(v1_pix[0], v2_pix[0], v3_pix[0])))
            max_y = ti.min(
                self.height, ti.ceil(ti.max(v1_pix[1], v2_pix[1], v3_pix[1]))
            )

            # 6. Calculate triangle area (for early culling of degenerate triangles)
            area_x2 = ti.abs(
                (v2_pix[0] - v1_pix[0]) * (v3_pix[1] - v1_pix[1])
                - (v3_pix[0] - v1_pix[0]) * (v2_pix[1] - v1_pix[1])
            )

            # Optimization: Cull very small or degenerate triangles quickly
            # Also cull if bounding box is invalid
            if area_x2 < 1e-3 or max_x <= min_x or max_y <= min_y:
                # Optional: Handle very small triangles by drawing their center?
                # (Could be useful for point clouds or distant objects)
                # For now, just skip them.
                continue  # Skip this face

            # 7. Rasterization loop
            pixels_filled = 0
            # Iterate over pixels in the bounding box
            # Use ti.i32() for range casting
            for y in range(ti.i32(min_y), ti.i32(max_y)):
                for x in range(ti.i32(min_x), ti.i32(max_x)):
                    # Pixel center
                    p = ti.Vector([ti.f32(x) + 0.5, ti.f32(y) + 0.5], dt=ti.f32)

                    # Calculate barycentric coordinates for the pixel center
                    bary = barycentric_coordinates_ti(p, v1_pix, v2_pix, v3_pix)

                    # Check if the pixel center is inside the triangle (with tolerance)
                    epsilon = 1e-5
                    if (
                        bary[0] >= -epsilon
                        and bary[1] >= -epsilon
                        and bary[2] >= -epsilon
                        and (bary[0] + bary[1] + bary[2] <= 1.0 + epsilon)
                    ):

                        # Interpolate depth (Z) using original view space Z values
                        # Result is positive depth value for buffer comparison
                        interpolated_depth = interpolate_z_ti(
                            bary, z1_view, z2_view, z3_view, is_perspective
                        )

                        # Check if depth is valid (not inf)
                        if interpolated_depth != float("inf"):
                            # Z-Buffer Test (if enabled)
                            # Compare interpolated_depth with existing depth in buffer
                            # Use ti.atomic_min for safe concurrent writes from different threads
                            if (
                                use_zbuffer == 0
                                or interpolated_depth < self.depth_buffer[y, x]
                            ):
                                old_depth = ti.atomic_min(
                                    self.depth_buffer[y, x], interpolated_depth
                                )
                                # Ensure we were the thread that successfully updated the depth
                                if (
                                    old_depth > interpolated_depth
                                ):  # Check avoids race condition if atomic_min isn't perfect? Or just ensures write.
                                    # if self.depth_buffer[y, x] == interpolated_depth: # Alternative check if atomic_min guarantees return of *old* value
                                    self.color_buffer[y, x] = face_color
                                    pixels_filled += 1

            # If any pixels were filled for this triangle, increment the counter
            if pixels_filled > 0:
                valid_triangles_rendered += 1

        return valid_triangles_rendered

    @ti.kernel
    def render_triangles_unified_with_texture(
        self,
        vertices_x: ti.types.ndarray(),  # Pixel space X
        vertices_y: ti.types.ndarray(),  # Pixel space Y
        faces: ti.types.ndarray(),  # Face vertex indices (n_faces, 3)
        z_values: ti.types.ndarray(),  # Original view space Z for vertices
        # colors_r: ti.types.ndarray(), # Base color (might be modulated by texture)
        # colors_g: ti.types.ndarray(),
        # colors_b: ti.types.ndarray(),
        face_count: ti.i32,
        is_perspective: ti.i32,
        use_zbuffer: ti.i32,
        # --- Texture specific arguments ---
        texcoords_u: ti.types.ndarray(),  # Texcoords U for all unique texcoords
        texcoords_v: ti.types.ndarray(),  # Texcoords V for all unique texcoords
        face_texcoord_indices: ti.types.ndarray(),  # Indices into texcoords_u/v per face vertex (n_faces, 3)
        texture: ti.types.ndarray(
            element_dim=1
        ),  # Texture data (h, w, 4) passed as ndarray
        texture_width: ti.i32,
        texture_height: ti.i32,
    ) -> ti.i32:
        """Taichi kernel for rendering triangles with textures."""
        valid_triangles_rendered = 0
        ti.loop_config(block_dim=128)

        for face_idx in range(face_count):
            # 1. Get triangle vertex indices
            i0, i1, i2 = faces[face_idx, 0], faces[face_idx, 1], faces[face_idx, 2]

            # 2. Get vertex positions in pixel space
            v1_pix = ti.Vector([vertices_x[i0], vertices_y[i0]], dt=ti.f32)
            v2_pix = ti.Vector([vertices_x[i1], vertices_y[i1]], dt=ti.f32)
            v3_pix = ti.Vector([vertices_x[i2], vertices_y[i2]], dt=ti.f32)

            # 3. Get original view space Z values
            z1_view, z2_view, z3_view = z_values[i0], z_values[i1], z_values[i2]

            # 4. Get texture coordinate indices for this face's vertices
            ti0, ti1, ti2 = (
                face_texcoord_indices[face_idx, 0],
                face_texcoord_indices[face_idx, 1],
                face_texcoord_indices[face_idx, 2],
            )
            # Check for invalid texture indices (e.g., -1 from loader)
            if ti0 < 0 or ti1 < 0 or ti2 < 0:
                continue  # Skip faces without valid texture coordinates

            # 5. Get texture coordinates (U, V) for the vertices
            tc1 = ti.Vector([texcoords_u[ti0], texcoords_v[ti0]], dt=ti.f32)
            tc2 = ti.Vector([texcoords_u[ti1], texcoords_v[ti1]], dt=ti.f32)
            tc3 = ti.Vector([texcoords_u[ti2], texcoords_v[ti2]], dt=ti.f32)

            # 6. Calculate bounding box
            min_x = ti.max(0, ti.floor(ti.min(v1_pix[0], v2_pix[0], v3_pix[0])))
            min_y = ti.max(0, ti.floor(ti.min(v1_pix[1], v2_pix[1], v3_pix[1])))
            max_x = ti.min(self.width, ti.ceil(ti.max(v1_pix[0], v2_pix[0], v3_pix[0])))
            max_y = ti.min(
                self.height, ti.ceil(ti.max(v1_pix[1], v2_pix[1], v3_pix[1]))
            )

            # 7. Calculate triangle area for early culling
            area_x2 = ti.abs(
                (v2_pix[0] - v1_pix[0]) * (v3_pix[1] - v1_pix[1])
                - (v3_pix[0] - v1_pix[0]) * (v2_pix[1] - v1_pix[1])
            )
            if area_x2 < 1e-3 or max_x <= min_x or max_y <= min_y:
                continue

            # 8. Rasterization loop
            pixels_filled = 0
            for y in range(ti.i32(min_y), ti.i32(max_y)):
                for x in range(ti.i32(min_x), ti.i32(max_x)):
                    p = ti.Vector([ti.f32(x) + 0.5, ti.f32(y) + 0.5], dt=ti.f32)
                    bary = barycentric_coordinates_ti(p, v1_pix, v2_pix, v3_pix)

                    # Check if inside triangle
                    epsilon = 1e-5
                    if (
                        bary[0] >= -epsilon
                        and bary[1] >= -epsilon
                        and bary[2] >= -epsilon
                        and (bary[0] + bary[1] + bary[2] <= 1.0 + epsilon)
                    ):

                        # Interpolate depth
                        interpolated_depth = interpolate_z_ti(
                            bary, z1_view, z2_view, z3_view, is_perspective
                        )

                        if interpolated_depth != float("inf"):
                            # Z-Buffer Test
                            if (
                                use_zbuffer == 0
                                or interpolated_depth < self.depth_buffer[y, x]
                            ):
                                # Interpolate texture coordinates (perspective correct)
                                # We need to interpolate U/Z, V/Z, 1/Z
                                inv_z1 = (
                                    1.0 / z1_view if ti.abs(z1_view) > 1e-9 else 0.0
                                )
                                inv_z2 = (
                                    1.0 / z2_view if ti.abs(z2_view) > 1e-9 else 0.0
                                )
                                inv_z3 = (
                                    1.0 / z3_view if ti.abs(z3_view) > 1e-9 else 0.0
                                )

                                interpolated_inv_z = (
                                    bary[0] * inv_z1
                                    + bary[1] * inv_z2
                                    + bary[2] * inv_z3
                                )

                                if ti.abs(interpolated_inv_z) > 1e-9:
                                    # Perspective correct interpolation for U and V
                                    interp_u = (
                                        bary[0] * tc1[0] * inv_z1
                                        + bary[1] * tc2[0] * inv_z2
                                        + bary[2] * tc3[0] * inv_z3
                                    ) / interpolated_inv_z
                                    interp_v = (
                                        bary[0] * tc1[1] * inv_z1
                                        + bary[1] * tc2[1] * inv_z2
                                        + bary[2] * tc3[1] * inv_z3
                                    ) / interpolated_inv_z
                                else:
                                    # Fallback to linear interpolation if perspective correction fails
                                    interp_u = (
                                        bary[0] * tc1[0]
                                        + bary[1] * tc2[0]
                                        + bary[2] * tc3[0]
                                    )
                                    interp_v = (
                                        bary[0] * tc1[1]
                                        + bary[1] * tc2[1]
                                        + bary[2] * tc3[1]
                                    )

                                # Texture sampling (using nearest neighbor for simplicity)
                                # Apply texture coordinate wrapping (e.g., repeat)
                                sample_u = interp_u % 1.0
                                sample_v = interp_v % 1.0
                                # Handle potential negative results from modulo
                                if sample_u < 0.0:
                                    sample_u += 1.0
                                if sample_v < 0.0:
                                    sample_v += 1.0

                                # Convert normalized UVs to texture pixel coordinates
                                tex_x = ti.i32(
                                    sample_u * (texture_width - 1)
                                )  # Use width-1 for index
                                tex_y = ti.i32(
                                    (1.0 - sample_v) * (texture_height - 1)
                                )  # Flip V coord (common convention)

                                # Clamp coordinates to be safe
                                tex_x = ti.max(0, ti.min(texture_width - 1, tex_x))
                                tex_y = ti.max(0, ti.min(texture_height - 1, tex_y))

                                # Sample the texture (assuming texture is Ndarray(element_dim=1))
                                # texel = texture[tex_y, tex_x] # This gives a Vector(4)
                                tex_index = tex_y * texture_width + tex_x
                                texel = texture[
                                    tex_index
                                ]  # Access flat texture data if needed, or use 2D indexing if passed correctly

                                # Update depth buffer and color buffer atomically
                                old_depth = ti.atomic_min(
                                    self.depth_buffer[y, x], interpolated_depth
                                )
                                if old_depth > interpolated_depth:
                                    # Use texture color (first 3 components RGB)
                                    self.color_buffer[y, x] = ti.Vector(
                                        [texel[0], texel[1], texel[2]]
                                    )
                                    # Optional: Modulate with base color? self.color_buffer[y, x] = base_color * texel.rgb
                                    pixels_filled += 1

            if pixels_filled > 0:
                valid_triangles_rendered += 1

        return valid_triangles_rendered

    def render_triangles_unified(
        self,
        vertices_x,
        vertices_y,
        faces,
        z_values,  # View space Z
        colors_r,  # Base color R (used if no texture or for modulation)
        colors_g,  # Base color G
        colors_b,  # Base color B
        face_count,
        is_perspective,
        use_zbuffer,
        # Texture related args (can be None)
        use_texture=False,
        texcoords_u=None,
        texcoords_v=None,
        face_texcoord_indices=None,
        texture_data=None,  # NumPy array (h, w, 4)
    ):
        """
        Dispatches rendering to the appropriate Taichi kernel based on texture usage.
        Handles data preparation for the texture kernel.
        """
        if (
            use_texture
            and texcoords_u is not None
            and texcoords_v is not None
            and face_texcoord_indices is not None
            and texture_data is not None
            and texture_data.shape[0] > 0  # Height > 0
            and texture_data.shape[1] > 0  # Width > 0
            and texture_data.shape[2] == 4  # RGBA
        ):
            texture_height, texture_width, _ = texture_data.shape

            # --- Prepare Texture Data for Taichi ---
            # Taichi ndarray expects a specific layout.
            # If element_dim=1, it expects (H, W, C) but accessed linearly or via ti.Vector.
            # Create a Taichi Vector field or pass Ndarray directly. Ndarray is usually easier.
            # Ensure texture_data is contiguous in memory if needed
            texture_data_cont = np.ascontiguousarray(texture_data)
            # Pass texture data as ndarray to the kernel
            # texture_ti = ti.Vector.field(4, dtype=ti.f32, shape=(texture_height, texture_width))
            # texture_ti.from_numpy(texture_data)

            # Ensure index arrays are not empty if provided
            if face_texcoord_indices.size == 0:
                print("警告: 提供了纹理但 face_texcoord_indices 为空。禁用纹理。")
                use_texture = False  # Fallback if indices are missing

        # --- Kernel Dispatch ---
        if use_texture:
            print("使用纹理渲染内核")
            return self.render_triangles_unified_with_texture(
                vertices_x,
                vertices_y,
                faces,
                z_values,
                # colors_r, # Pass base colors if you want to modulate later
                # colors_g,
                # colors_b,
                face_count,
                is_perspective,
                use_zbuffer,
                texcoords_u,
                texcoords_v,
                face_texcoord_indices,
                texture_data_cont,  # Pass the numpy array directly
                texture_width,
                texture_height,
            )
        else:
            print("使用无纹理渲染内核")
            return self.render_triangles_unified_without_texture(
                vertices_x,
                vertices_y,
                faces,
                z_values,
                colors_r,
                colors_g,
                colors_b,
                face_count,
                is_perspective,
                use_zbuffer,
            )

    def get_color_array(self):
        """获取颜色缓冲区的 NumPy 数组"""
        return self.color_buffer.to_numpy()

    def get_depth_array(self):
        """获取深度缓冲区的 NumPy 数组"""
        # Remember depth buffer stores positive values, closer is smaller
        depth_np = self.depth_buffer.to_numpy()
        # Convert infinite values back to NaN or keep as inf? NaN might be better for processing.
        # depth_np[np.isinf(depth_np)] = np.nan
        return depth_np
