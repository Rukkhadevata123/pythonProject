import taichi as ti
from .vec3 import Vec3, dot
from .ray import Ray
from .hittable import Hittable, HitRecord
from .interval import Interval


class Sphere(Hittable):
    """表示一个球体"""

    def __init__(self, center, radius, material=None):
        """
        创建一个球体

        参数:
        center: 球心位置，Point3类型
        radius: 球体半径，浮点数
        material: 材质（可选，后续实现）
        """
        self.center = center
        self.radius = radius
        self.mat = material

    def hit(self, ray, ray_t, rec):
        """
        检测光线是否击中球体

        参数:
        ray: 入射光线
        ray_t: 有效参数范围
        rec: 交点记录

        返回值:
        是否击中
        """
        # 计算光线与球体的交点
        oc = ray.origin - self.center
        a = ray.direction.length_squared()
        half_b = dot(oc, ray.direction)
        c = oc.length_squared() - self.radius * self.radius

        discriminant = half_b * half_b - a * c

        if discriminant < 0:
            return False

        # 计算交点参数t
        sqrtd = discriminant**0.5

        # 找到在射线参数范围内的最近根
        root = (-half_b - sqrtd) / a
        if not ray_t.surrounds(root):
            root = (-half_b + sqrtd) / a
            if not ray_t.surrounds(root):
                return False

        # 记录交点信息
        rec.t = root
        rec.p = ray.at(rec.t)
        outward_normal = (rec.p - self.center) / self.radius
        rec.set_face_normal(ray, outward_normal)
        rec.mat = self.mat

        return True
