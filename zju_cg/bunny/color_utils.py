import numpy as np


def get_face_color(face_index, colorize=False):
    """获取面片的基础颜色"""
    if not colorize:
        # 默认灰色
        return np.array([0.7, 0.7, 0.7], dtype=np.float32)
    # 根据面片索引生成伪随机颜色
    np.random.seed(face_index)  # 确保颜色对于同一面片是固定的
    return np.array(
        [
            0.3 + np.random.random() * 0.4,  # R
            0.3 + np.random.random() * 0.4,  # G
            0.3 + np.random.random() * 0.4,  # B
        ],
        dtype=np.float32,
    )


def apply_colormap_jet(normalized_depth):
    """
    将归一化的深度图转换为彩色图像（使用jet色彩映射）

    参数:
    normalized_depth: 归一化的深度图，值在[0,1]范围内

    返回:
    彩色图像，形状为(h, w, 3)，值在[0,255]范围内
    """
    h, w = normalized_depth.shape
    result = np.zeros((h, w, 3), dtype=np.uint8)

    # 创建掩码来检测有效的深度值（非无穷、非NaN）
    mask_valid = ~(np.isnan(normalized_depth) | np.isinf(normalized_depth))

    # 初始化颜色通道
    r = np.zeros((h, w), dtype=np.float32)
    g = np.zeros((h, w), dtype=np.float32)
    b = np.zeros((h, w), dtype=np.float32)

    # 复制归一化深度值进行安全操作
    values = np.copy(normalized_depth)
    values = np.clip(values, 0.0, 1.0)  # 确保在[0, 1]范围内

    # 根据jet色彩映射将深度值映射为颜色
    # 将[0,1]划分为四个区间[0, 0.25, 0.5, 0.75, 1]

    # 第一区间: 0 - 0.25 (蓝色到青色)
    mask_1 = np.logical_and(mask_valid, values <= 0.25)
    b[mask_1] = 1.0
    g[mask_1] = values[mask_1] * 4

    # 第二区间: 0.25 - 0.5 (青色到绿色)
    mask_2 = np.logical_and(mask_valid, np.logical_and(values > 0.25, values <= 0.5))
    g[mask_2] = 1.0
    b[mask_2] = 1.0 - (values[mask_2] - 0.25) * 4

    # 第三区间: 0.5 - 0.75 (绿色到黄色)
    mask_3 = np.logical_and(mask_valid, np.logical_and(values > 0.5, values <= 0.75))
    g[mask_3] = 1.0
    r[mask_3] = (values[mask_3] - 0.5) * 4

    # 第四区间: 0.75 - 1.0 (黄色到红色)
    mask_4 = np.logical_and(mask_valid, values > 0.75)
    r[mask_4] = 1.0
    g[mask_4] = 1.0 - (values[mask_4] - 0.75) * 4

    # 将[0,1]浮点数转换为[0,255]整数
    result[:, :, 0] = np.clip(r * 255, 0, 255).astype(np.uint8)
    result[:, :, 1] = np.clip(g * 255, 0, 255).astype(np.uint8)
    result[:, :, 2] = np.clip(b * 255, 0, 255).astype(np.uint8)

    return result
