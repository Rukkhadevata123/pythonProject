import numpy as np


def ndc_to_pixel(ndc_coords, width, height):
    """将归一化设备坐标 (NDC) 转换为屏幕像素坐标"""
    pixel_coords = np.zeros_like(ndc_coords, dtype=np.float32)
    # X: [-1, 1] -> [0, width]
    pixel_coords[:, 0] = (ndc_coords[:, 0] + 1.0) * width / 2.0
    # Y: [-1, 1] -> [height, 0] (OpenGL/Taichi convention Y up, pixel Y down)
    pixel_coords[:, 1] = height - (ndc_coords[:, 1] + 1.0) * height / 2.0
    # pixel_coords[:, 1] = (ndc_coords[:, 1] + 1.0) * height / 2.0 # Use this if Y is already flipped
    return pixel_coords


def rotate_model(vertices, angle_degrees):
    """绕Y轴旋转顶点"""
    theta = np.radians(angle_degrees)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    # 更高效的旋转矩阵乘法
    rotated_vertices = vertices.copy()
    x = vertices[:, 0]
    z = vertices[:, 2]
    rotated_vertices[:, 0] = x * cos_theta + z * sin_theta
    rotated_vertices[:, 2] = -x * sin_theta + z * cos_theta
    return rotated_vertices


def orthographic_projection(vertices):
    """执行正交投影 (只取 X, Y 坐标)"""
    # 假设顶点已经在合适的范围内，或者将在后续步骤中标准化
    return vertices[:, :2].astype(np.float32)


def perspective_projection(vertices, focal_length=2.0):
    """执行透视投影"""
    projected_vertices = np.zeros((vertices.shape[0], 2), dtype=np.float32)
    z_values = vertices[:, 2]

    # 处理 Z 坐标接近 0 的情况
    # Note: Taichi rasterizer expects coordinates AFTER projection,
    # Z should be view space Z, typically negative.
    # We divide by -Z for perspective projection.
    near_plane_threshold = 1e-6  # Avoid division by zero or values too close
    safe_neg_z = np.where(
        z_values > -near_plane_threshold, -near_plane_threshold, z_values
    )

    # Perform projection: x' = x * f / (-z), y' = y * f / (-z)
    projected_vertices[:, 0] = vertices[:, 0] * focal_length / (-safe_neg_z)
    projected_vertices[:, 1] = vertices[:, 1] * focal_length / (-safe_neg_z)

    # 处理原始 Z 值非常接近 0 的情况 (可选，取决于如何处理裁剪)
    # projected_vertices[z_values > -near_plane_threshold] = np.inf # Mark as invalid/clipped

    return projected_vertices
