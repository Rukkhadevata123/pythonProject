import taichi as ti
import numpy as np
import math
import random as py_random


@ti.data_oriented
class Vec3:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        """
        创建一个三维向量

        参数:
        x, y, z: 向量的三个分量
        """
        # 使用场表示向量的三个分量
        self.data = ti.Vector([float(x), float(y), float(z)])

    @property
    def x(self):
        return self.data[0]

    @property
    def y(self):
        return self.data[1]

    @property
    def z(self):
        return self.data[2]

    @property
    def r(self):
        return self.data[0]

    @property
    def g(self):
        return self.data[1]

    @property
    def b(self):
        return self.data[2]

    def __getitem__(self, i):
        return self.data[i]

    def __setitem__(self, i, value):
        self.data[i] = value

    def __neg__(self):
        return Vec3(-self.data[0], -self.data[1], -self.data[2])

    def __add__(self, other):
        if isinstance(other, Vec3):
            return Vec3(
                self.data[0] + other.data[0],
                self.data[1] + other.data[1],
                self.data[2] + other.data[2],
            )
        else:
            return Vec3(
                self.data[0] + other, self.data[1] + other, self.data[2] + other
            )

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, Vec3):
            return Vec3(
                self.data[0] - other.data[0],
                self.data[1] - other.data[1],
                self.data[2] - other.data[2],
            )
        else:
            return Vec3(
                self.data[0] - other, self.data[1] - other, self.data[2] - other
            )

    def __rsub__(self, other):
        return Vec3(other - self.data[0], other - self.data[1], other - self.data[2])

    def __mul__(self, other):
        if isinstance(other, Vec3):
            return Vec3(
                self.data[0] * other.data[0],
                self.data[1] * other.data[1],
                self.data[2] * other.data[2],
            )
        else:
            return Vec3(
                self.data[0] * other, self.data[1] * other, self.data[2] * other
            )

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, Vec3):
            return Vec3(
                self.data[0] / other.data[0],
                self.data[1] / other.data[1],
                self.data[2] / other.data[2],
            )
        else:
            inv = 1.0 / other
            return self.__mul__(inv)

    def __rtruediv__(self, other):
        return Vec3(other / self.data[0], other / self.data[1], other / self.data[2])

    def length_squared(self):
        """返回向量长度的平方"""
        return self.data[0] ** 2 + self.data[1] ** 2 + self.data[2] ** 2

    def length(self):
        """返回向量长度"""
        return ti.sqrt(self.length_squared())

    def near_zero(self):
        """检查向量是否接近零向量"""
        s = 1e-8
        return (
            (abs(self.data[0]) < s)
            and (abs(self.data[1]) < s)
            and (abs(self.data[2]) < s)
        )

    def __str__(self):
        return f"Vec3({self.data[0]}, {self.data[1]}, {self.data[2]})"

    def __repr__(self):
        return self.__str__()

    @staticmethod
    def random(min_val=0.0, max_val=1.0):
        """生成随机向量"""
        # 使用Python的random模块来生成随机数，而不是ti.random()
        return Vec3(
            py_random.random() * (max_val - min_val) + min_val,
            py_random.random() * (max_val - min_val) + min_val,
            py_random.random() * (max_val - min_val) + min_val,
        )


# 将点定义为Vec3的别名，提高代码可读性
Point3 = Vec3
# 将颜色定义为Vec3的别名
Color = Vec3


# 向量操作工具函数
def dot(u, v):
    """计算两个向量的点积"""
    return u.data[0] * v.data[0] + u.data[1] * v.data[1] + u.data[2] * v.data[2]


def cross(u, v):
    """计算两个向量的叉积"""
    return Vec3(
        u.data[1] * v.data[2] - u.data[2] * v.data[1],
        u.data[2] * v.data[0] - u.data[0] * v.data[2],
        u.data[0] * v.data[1] - u.data[1] * v.data[0],
    )


def unit_vector(v):
    """返回向量的单位向量"""
    return v / v.length()


def random_in_unit_disk():
    """在单位圆盘内生成随机点"""
    while True:
        # 改用Python的random模块而不是ti.random()
        p = Vec3(py_random.random() * 2 - 1, py_random.random() * 2 - 1, 0)
        if p.length_squared() < 1:
            return p


def random_unit_vector():
    """生成随机单位向量"""
    while True:
        # 使用修改后的Vec3.random，它现在使用Python的随机数生成器
        p = Vec3.random(-1, 1)
        len_squared = p.length_squared()
        if 1e-160 < len_squared <= 1.0:
            return p / ti.sqrt(len_squared)


def random_on_hemisphere(normal):
    """在法线方向的半球上生成随机点"""
    on_unit_sphere = random_unit_vector()
    if dot(on_unit_sphere, normal) > 0.0:  # 在法线同一半球
        return on_unit_sphere
    else:
        return -on_unit_sphere


def reflect(v, n):
    """计算反射向量"""
    return v - 2 * dot(v, n) * n


def refract(uv, n, etai_over_etat):
    """计算折射向量"""
    cos_theta = min(dot(-uv, n), 1.0)
    r_out_perp = etai_over_etat * (uv + cos_theta * n)
    r_out_parallel = -ti.sqrt(abs(1.0 - r_out_perp.length_squared())) * n
    return r_out_perp + r_out_parallel
