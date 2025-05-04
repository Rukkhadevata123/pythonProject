import numpy as np
import os
from PIL import Image
from scipy import ndimage  # For procedural noise generation


def load_texture(texture_path, default_color=[0.7, 0.7, 0.7]):
    """加载纹理图像文件"""
    try:
        if texture_path and os.path.exists(texture_path):
            img = Image.open(texture_path).convert("RGBA")
            # Flip texture vertically (common practice for OpenGL/graphics conventions)
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
            # Normalize to [0, 1] float
            texture_array = np.array(img, dtype=np.float32) / 255.0
            print(f"成功加载纹理: {texture_path} (形状: {texture_array.shape})")
            # Ensure minimum size if needed, although Taichi textures handle dimensions
            if texture_array.shape[0] == 0 or texture_array.shape[1] == 0:
                raise ValueError("纹理尺寸无效 (0)")
            return texture_array
        else:
            print(f"纹理文件未找到或未提供: {texture_path}")

    except Exception as e:
        print(f"错误: 加载纹理 '{texture_path}' 失败: {e}")

    # Return a 1x1 default texture if loading fails or no path provided
    print(f"使用 1x1 默认颜色纹理: {default_color}")
    return np.array(
        [[[default_color[0], default_color[1], default_color[2], 1.0]]],  # RGBA
        dtype=np.float32,
    )


def generate_procedural_texture(
    texture_type="checkerboard",
    size=512,
    color1=[0.8, 0.8, 0.8],
    color2=[0.2, 0.2, 0.2],
):
    """生成程序化纹理（优化版）"""
    # Ensure size is positive
    size = max(1, int(size))
    # Prepare RGBA colors
    color1_rgba = np.array([*color1, 1.0], dtype=np.float32)
    color2_rgba = np.array([*color2, 1.0], dtype=np.float32)

    texture = np.zeros((size, size, 4), dtype=np.float32)

    if texture_type == "checkerboard":
        # Use NumPy broadcasting to create checkerboard pattern
        check_size = max(1, size // 8)  # Ensure check_size is at least 1
        # Create coordinate grids
        x, y = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
        # Determine color based on checker index parity
        checker_mask = (x // check_size) % 2 == (y // check_size) % 2

        # Apply colors using the mask
        texture[checker_mask] = color1_rgba
        texture[~checker_mask] = color2_rgba

    elif texture_type == "gradient":
        # Create a linear gradient (e.g., vertical)
        gradient = np.linspace(0, 1, size)[:, np.newaxis]  # Shape (size, 1)

        # Interpolate RGB channels based on the gradient
        texture[..., 0] = color1[0] * (1 - gradient) + color2[0] * gradient  # R
        texture[..., 1] = color1[1] * (1 - gradient) + color2[1] * gradient  # G
        texture[..., 2] = color1[2] * (1 - gradient) + color2[2] * gradient  # B
        texture[..., 3] = 1.0  # Alpha

    elif texture_type == "noise":
        # Generate low-resolution noise and upscale smoothly
        try:
            from numpy.random import RandomState

            rng = RandomState(42)  # Seed for reproducibility
            small_size = max(1, size // 8)  # Base noise resolution
            noise = rng.rand(small_size, small_size)

            # Upscale using spline interpolation (order=3 for cubic, order=1 for linear)
            zoom_factor = size / small_size
            noise_large = ndimage.zoom(noise, zoom_factor, order=1)  # Linear is faster

            # Ensure the output size is exactly 'size' due to potential float precision issues
            noise_large = noise_large[:size, :size]

            # Clamp values to [0, 1] range after interpolation
            noise_large = np.clip(noise_large, 0.0, 1.0)

        except ImportError:
            print("警告: 未找到 scipy.ndimage。使用简单的 (可能块状的) 噪声。")
            noise_large = np.random.rand(size, size)

        # Expand noise to 3D array for color blending (shape becomes size, size, 1)
        noise_3d = noise_large[..., np.newaxis]

        # Linearly interpolate between color1 and color2 based on noise intensity
        texture[..., :3] = color1_rgba[:3] * (1 - noise_3d) + color2_rgba[:3] * noise_3d
        texture[..., 3] = 1.0  # Alpha

    else:
        print(f"警告: 未知的程序化纹理类型 '{texture_type}'。返回默认棋盘格。")
        # Fallback to checkerboard if type is unknown
        return generate_procedural_texture("checkerboard", size, color1, color2)

    return texture
