from .hittable import Hittable, HitRecord
from .interval import Interval


class HittableList(Hittable):
    """可被光线击中的物体列表"""

    def __init__(self):
        """初始化空的物体列表"""
        self.objects = []

    def add(self, obj):
        """
        添加物体到列表

        参数:
        obj: 要添加的物体（Hittable类型）
        """
        self.objects.append(obj)

    def clear(self):
        """清空物体列表"""
        self.objects.clear()

    def hit(self, ray, ray_t, rec):
        """
        检测光线是否击中列表中的任何物体

        参数:
        ray: 入射光线
        ray_t: 有效参数范围
        rec: 交点记录

        返回值:
        是否击中
        """
        temp_rec = HitRecord()
        hit_anything = False
        closest_so_far = ray_t.max

        # 遍历所有物体，找到最近的交点
        for obj in self.objects:
            if obj.hit(ray, Interval(ray_t.min, closest_so_far), temp_rec):
                hit_anything = True
                closest_so_far = temp_rec.t
                # 复制临时记录的所有属性
                rec.p = temp_rec.p
                rec.normal = temp_rec.normal
                rec.t = temp_rec.t
                rec.front_face = temp_rec.front_face
                rec.mat = temp_rec.mat

        return hit_anything
