import taichi as ti
from .vec3 import Vec3, Point3


@ti.data_oriented
class Ray:
    """表示一条光线，有起点和方向"""

    def __init__(self, origin=None, direction=None, time=0.0):
        """
        创建一条光线

        参数:
        origin: 光线起点，Point3类型
        direction: 光线方向，Vec3类型
        time: 光线时间（用于动态场景）
        """
        self.origin = Point3() if origin is None else origin
        self.direction = Vec3() if direction is None else direction
        self.time = time

    def at(self, t):
        """
        获取光线上参数t处的点

        参数:
        t: 光线参数

        返回值:
        光线上t参数处的点
        """
        return self.origin + t * self.direction

    def __str__(self):
        return (
            f"Ray(origin={self.origin}, direction={self.direction}, time={self.time})"
        )

    def __repr__(self):
        return self.__str__()
