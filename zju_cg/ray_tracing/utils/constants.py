import math
import random

# 常量
INFINITY = float("inf")
PI = math.pi


def degrees_to_radians(degrees):
    """将角度转换为弧度"""
    return degrees * PI / 180.0


def random_double(min_val=0.0, max_val=1.0):
    """
    生成指定范围内的随机浮点数

    参数:
    min_val: 最小值（包含）
    max_val: 最大值（不包含）

    返回值:
    [min_val, max_val)范围内的随机浮点数
    """
    # 直接使用Python的random模块，避免在多进程中使用ti.random()
    return min_val + (max_val - min_val) * random.random()


def clamp(x, min_val, max_val):
    """
    将值限制在指定范围内

    参数:
    x: 要限制的值
    min_val: 最小值
    max_val: 最大值

    返回值:
    限制在[min_val, max_val]范围内的值
    """
    if x < min_val:
        return min_val
    if x > max_val:
        return max_val
    return x
