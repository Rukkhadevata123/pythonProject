import taichi as ti
import numpy as np
from .vec3 import Color
from .interval import Interval


def linear_to_gamma(linear_component):
    """
    将线性颜色分量转换为伽马空间（伽马值为2）

    参数:
    linear_component: 线性空间中的颜色分量

    返回值:
    伽马空间中的颜色分量
    """
    if linear_component > 0:
        return ti.sqrt(linear_component)
    return 0.0


def write_color(file, pixel_color, samples_per_pixel=1):
    """
    将颜色值写入输出流，包括伽马校正

    参数:
    file: 输出文件对象
    pixel_color: 像素颜色 (Color 类型)
    samples_per_pixel: 每像素的采样数（用于平均多采样）
    """
    # 根据采样数对颜色进行缩放
    scale = 1.0 / samples_per_pixel
    r = pixel_color.r * scale
    g = pixel_color.g * scale
    b = pixel_color.b * scale

    # 应用线性到伽马变换（伽马2）
    r = linear_to_gamma(r)
    g = linear_to_gamma(g)
    b = linear_to_gamma(b)

    # 将 [0,1] 范围的分量值转换为 [0,255] 字节范围
    intensity = Interval(0.000, 0.999)
    r_byte = int(256 * intensity.clamp(r))
    g_byte = int(256 * intensity.clamp(g))
    b_byte = int(256 * intensity.clamp(b))

    # 写出像素颜色分量
    file.write(f"{r_byte} {g_byte} {b_byte}\n")


def write_color_to_array(pixel_array, i, j, pixel_color, samples_per_pixel=1):
    """
    将颜色值写入像素数组（用于后续图像处理或显示）

    参数:
    pixel_array: 像素数组 (numpy或taichi数组)
    i, j: 像素位置
    pixel_color: 像素颜色 (Color 类型)
    samples_per_pixel: 每像素的采样数（用于平均多采样）
    """
    # 根据采样数对颜色进行缩放
    scale = 1.0 / samples_per_pixel
    r = pixel_color.r * scale
    g = pixel_color.g * scale
    b = pixel_color.b * scale

    # 应用线性到伽马变换
    r = linear_to_gamma(r)
    g = linear_to_gamma(g)
    b = linear_to_gamma(b)

    # 将 [0,1] 范围的分量值限制在有效范围内
    intensity = Interval(0.000, 0.999)
    r = intensity.clamp(r)
    g = intensity.clamp(g)
    b = intensity.clamp(b)

    # 将颜色值写入数组
    pixel_array[j, i, 0] = r
    pixel_array[j, i, 1] = g
    pixel_array[j, i, 2] = b


def create_image_buffer(width, height):
    """
    创建一个用于存储渲染结果的图像缓冲区

    参数:
    width: 图像宽度
    height: 图像高度

    返回值:
    taichi 图像场
    """
    # 创建一个 taichi 场来存储图像数据
    image = ti.Vector.field(3, dtype=ti.f32, shape=(height, width))
    return image


def save_image(image_buffer, filename, format="png"):
    """
    将图像缓冲区保存为图像文件

    参数:
    image_buffer: taichi 图像场
    filename: 输出文件名
    format: 图像格式 (默认为png)
    """
    # 将 taichi 场转换为 numpy 数组
    img_np = image_buffer.to_numpy()

    # 确保值在 [0, 1] 范围内
    img_np = np.clip(img_np, 0, 1)

    # 使用 PIL 保存图像
    from PIL import Image

    img_uint8 = (img_np * 255).astype(np.uint8)
    img = Image.fromarray(img_uint8)
    img.save(filename, format=format)
    print(f"Image saved to {filename}")


def save_as_ppm(image_buffer, filename):
    """
    将图像缓冲区保存为 PPM 格式文件

    参数:
    image_buffer: taichi 图像场或 numpy 数组
    filename: 输出文件名
    """
    height, width = image_buffer.shape[0], image_buffer.shape[1]

    with open(filename, "w") as f:
        # 写入 PPM 文件头
        f.write(f"P3\n{width} {height}\n255\n")

        # 如果是 taichi 场，先转换为 numpy 数组
        if hasattr(image_buffer, "to_numpy"):
            img_np = image_buffer.to_numpy()
        else:
            img_np = image_buffer

        # 写入像素值
        for j in range(height):
            for i in range(width):
                r = int(255.999 * img_np[j, i, 0])
                g = int(255.999 * img_np[j, i, 1])
                b = int(255.999 * img_np[j, i, 2])
                f.write(f"{r} {g} {b}\n")

    print(f"PPM image saved to {filename}")
