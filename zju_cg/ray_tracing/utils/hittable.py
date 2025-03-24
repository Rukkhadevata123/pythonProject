import taichi as ti
from abc import ABC, abstractmethod
from .vec3 import Vec3, Point3, dot
from .ray import Ray
from .interval import Interval


class HitRecord:
    """记录光线与物体交点的相关信息"""

    def __init__(self):
        self.p = Point3()  # 交点位置
        self.normal = Vec3()  # 交点法线
        self.t = 0.0  # 光线参数
        self.front_face = False  # 是否为正面
        self.mat = None  # 材质（后续实现）

    def set_face_normal(self, ray, outward_normal):
        """
        根据入射光线和外向法线设置法线方向和前后表面信息

        参数:
        ray: 入射光线
        outward_normal: 外向法线（假设单位长度）
        """
        # 检查光线是从外部射入还是从内部射出
        self.front_face = dot(ray.direction, outward_normal) < 0

        # 根据front_face设置法线方向
        if self.front_face:
            self.normal = outward_normal
        else:
            self.normal = -outward_normal


class Hittable(ABC):
    """可被光线击中的抽象基类"""

    @abstractmethod
    def hit(self, ray, ray_t, rec):
        """
        检测光线是否击中物体

        参数:
        ray: 入射光线
        ray_t: 有效参数范围
        rec: 交点记录

        返回值:
        是否击中
        """
        pass
