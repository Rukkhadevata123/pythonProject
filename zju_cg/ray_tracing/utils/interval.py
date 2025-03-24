import taichi as ti
from .constants import INFINITY


@ti.data_oriented
class Interval:
    """表示一个数值区间[min, max]"""

    def __init__(self, min_val=INFINITY, max_val=-INFINITY):
        """
        创建一个区间

        参数:
        min_val: 区间最小值，默认为+无穷大（创建空区间）
        max_val: 区间最大值，默认为-无穷大（创建空区间）
        """
        self.min = float(min_val)
        self.max = float(max_val)

    def size(self):
        """返回区间大小"""
        return self.max - self.min

    def contains(self, x):
        """检查值是否在区间内（包括边界）"""
        return self.min <= x <= self.max

    def surrounds(self, x):
        """检查值是否严格在区间内（不包括边界）"""
        return self.min < x < self.max

    def clamp(self, x):
        """将值限定在区间内"""
        if x < self.min:
            return self.min
        if x > self.max:
            return self.max
        return x

    def __str__(self):
        return f"Interval({self.min}, {self.max})"

    def __repr__(self):
        return self.__str__()


# 创建两个特殊的区间实例
EMPTY = Interval(INFINITY, -INFINITY)  # 空区间
UNIVERSE = Interval(-INFINITY, INFINITY)  # 全域区间

# 将两个特殊区间附加到Interval类作为静态属性
Interval.empty = EMPTY
Interval.universe = UNIVERSE
