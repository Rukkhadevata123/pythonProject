import taichi as ti
from abc import ABC, abstractmethod
from .vec3 import (
    Vec3,
    dot,
    reflect,
    refract,
    random_unit_vector,
    unit_vector,
)  # 添加 unit_vector 导入
from .ray import Ray
from .constants import random_double


class Material(ABC):
    """材质抽象基类"""

    @abstractmethod
    def scatter(self, r_in, rec, attenuation, scattered):
        """
        计算光线散射

        参数:
        r_in: 入射光线
        rec: 碰撞记录
        attenuation: 衰减颜色 (由函数填充)
        scattered: 散射光线 (由函数填充)

        返回值:
        是否发生散射
        """
        return False


class Lambertian(Material):
    """漫反射材质"""

    def __init__(self, albedo):
        """
        初始化漫反射材质

        参数:
        albedo: 反照率 (颜色)
        """
        self.albedo = albedo

    def scatter(self, r_in, rec, attenuation, scattered):
        """实现漫反射散射"""
        scatter_direction = rec.normal + random_unit_vector()

        # 处理退化散射方向
        if scatter_direction.near_zero():
            scatter_direction = rec.normal

        scattered.origin = rec.p
        scattered.direction = scatter_direction
        attenuation.data = self.albedo.data
        return True


class Metal(Material):
    """金属材质"""

    def __init__(self, albedo, fuzz=0.0):
        """
        初始化金属材质

        参数:
        albedo: 反照率 (颜色)
        fuzz: 模糊度 (0=光滑, 1=最大模糊)
        """
        self.albedo = albedo
        self.fuzz = min(fuzz, 1.0)

    def scatter(self, r_in, rec, attenuation, scattered):
        """实现金属反射散射"""
        reflected = reflect(unit_vector(r_in.direction), rec.normal)

        # 添加模糊效果
        if self.fuzz > 0:
            reflected = reflected + self.fuzz * random_unit_vector()

        scattered.origin = rec.p
        scattered.direction = reflected
        attenuation.data = self.albedo.data

        # 只有当散射方向与法线夹角小于90度时才散射
        return dot(scattered.direction, rec.normal) > 0


class Dielectric(Material):
    """电介质材质 (如玻璃、水等)"""

    def __init__(self, refraction_index):
        """
        初始化电介质材质

        参数:
        refraction_index: 折射率
        """
        self.refraction_index = refraction_index

    def scatter(self, r_in, rec, attenuation, scattered):
        """实现电介质的反射/折射"""
        # 电介质不吸收光线，衰减始终为1
        attenuation.data = Vec3(1.0, 1.0, 1.0).data

        # 根据光线是从内部还是外部射入来确定折射率
        refraction_ratio = (
            1.0 / self.refraction_index if rec.front_face else self.refraction_index
        )

        unit_direction = unit_vector(r_in.direction)
        cos_theta = min(dot(-unit_direction, rec.normal), 1.0)
        sin_theta = (1.0 - cos_theta * cos_theta) ** 0.5

        # 判断是否可以折射 (全反射条件)
        cannot_refract = refraction_ratio * sin_theta > 1.0

        # 根据Schlick近似和全反射条件决定是反射还是折射
        if (
            cannot_refract
            or self._reflectance(cos_theta, refraction_ratio) > random_double()
        ):
            # 反射
            direction = reflect(unit_direction, rec.normal)
        else:
            # 折射
            direction = refract(unit_direction, rec.normal, refraction_ratio)

        scattered.origin = rec.p
        scattered.direction = direction
        return True

    def _reflectance(self, cosine, ref_idx):
        """
        使用Schlick近似计算反射率

        参数:
        cosine: 入射角的余弦
        ref_idx: 折射率

        返回值:
        反射率
        """
        r0 = (1 - ref_idx) / (1 + ref_idx)
        r0 = r0 * r0
        return r0 + (1 - r0) * ((1 - cosine) ** 5)
