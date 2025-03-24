import taichi as ti
from .vec3 import Vec3, Point3, Color, unit_vector, cross, random_in_unit_disk
from .ray import Ray
from .constants import degrees_to_radians, random_double


class Camera:
    def __init__(self):
        # 相机参数，默认值
        self.aspect_ratio = 1.0  # 宽高比
        self.image_width = 100  # 图像宽度
        self.image_height = 0  # 图像高度将在初始化时计算
        self.samples_per_pixel = 10  # 每像素的采样数
        self.max_depth = 10  # 光线最大反弹次数

        self.vfov = 90  # 垂直视场角(度)
        self.lookfrom = Point3(0, 0, 0)  # 相机位置
        self.lookat = Point3(0, 0, -1)  # 相机看向的点
        self.vup = Vec3(0, 1, 0)  # 相机的上方向

        self.defocus_angle = 0  # 散焦角度(度)
        self.focus_dist = 10  # 焦距

        # 这些将在initialize()中计算
        self.center = None
        self.pixel00_loc = None
        self.pixel_delta_u = None
        self.pixel_delta_v = None
        self.u = None
        self.v = None
        self.w = None
        self.defocus_disk_u = None
        self.defocus_disk_v = None

    def initialize(self):
        """初始化相机参数"""
        # 计算图像高度，确保至少为1
        self.image_height = int(self.image_width / self.aspect_ratio)
        self.image_height = max(1, self.image_height)

        # 采样比例因子
        self.pixel_samples_scale = 1.0 / self.samples_per_pixel

        # 相机中心就是观察点
        self.center = self.lookfrom

        # 确定视口尺寸
        theta = degrees_to_radians(self.vfov)
        h = ti.tan(theta / 2)
        viewport_height = 2 * h * self.focus_dist
        viewport_width = viewport_height * (self.image_width / self.image_height)

        # 计算相机坐标系的基向量
        self.w = unit_vector(self.lookfrom - self.lookat)  # 指向相机后方
        self.u = unit_vector(cross(self.vup, self.w))  # 指向相机右方
        self.v = cross(self.w, self.u)  # 指向相机上方

        # 计算视口的水平和垂直边缘向量
        viewport_u = viewport_width * self.u  # 视口水平边缘向量
        viewport_v = viewport_height * (-self.v)  # 视口垂直边缘向量(向下)

        # 计算像素间的水平和垂直delta向量
        self.pixel_delta_u = viewport_u / self.image_width
        self.pixel_delta_v = viewport_v / self.image_height

        # 计算左上角像素的位置
        viewport_upper_left = (
            self.center - (self.focus_dist * self.w) - viewport_u / 2 - viewport_v / 2
        )
        self.pixel00_loc = viewport_upper_left + 0.5 * (
            self.pixel_delta_u + self.pixel_delta_v
        )

        # 计算景深散焦盘的基向量
        defocus_radius = self.focus_dist * ti.tan(
            degrees_to_radians(self.defocus_angle / 2)
        )
        self.defocus_disk_u = defocus_radius * self.u
        self.defocus_disk_v = defocus_radius * self.v

    def get_ray(self, i, j):
        """获取通过像素(i,j)的光线，包含抖动采样"""
        # 在像素区域内随机采样
        offset = self._sample_square()
        pixel_sample = (
            self.pixel00_loc
            + ((i + offset.x) * self.pixel_delta_u)
            + ((j + offset.y) * self.pixel_delta_v)
        )

        # 确定光线起点(景深效果)
        ray_origin = self.center
        if self.defocus_angle > 0:
            ray_origin = self._defocus_disk_sample()

        ray_direction = pixel_sample - ray_origin

        return Ray(ray_origin, ray_direction)

    def _sample_square(self):
        """在[-0.5,0.5]x[-0.5,0.5]单位正方形内随机取点"""
        return Vec3(random_double() - 0.5, random_double() - 0.5, 0)

    def _defocus_disk_sample(self):
        """在景深散焦盘上随机取点"""
        p = random_in_unit_disk()
        return self.center + (p.x * self.defocus_disk_u) + (p.y * self.defocus_disk_v)

    def render(self, world, image):
        """渲染场景到图像缓冲区"""
        # 初始化相机参数
        self.initialize()

        # 创建进度条
        from .progress import create_progress_bar

        progress = create_progress_bar(self.image_height, "渲染场景", length=40)

        # 逐像素渲染
        for j in range(self.image_height):
            progress.update()

            for i in range(self.image_width):
                pixel_color = Color(0, 0, 0)

                # 每像素多次采样
                for s in range(self.samples_per_pixel):
                    ray = self.get_ray(i, j)
                    pixel_color += self._ray_color(ray, self.max_depth, world)

                # 采样平均和伽马校正
                r = pixel_color.r * self.pixel_samples_scale
                g = pixel_color.g * self.pixel_samples_scale
                b = pixel_color.b * self.pixel_samples_scale

                # 伽马校正
                r = r**0.5
                g = g**0.5
                b = b**0.5

                # 写入图像缓冲区
                image[j, i] = ti.Vector([r, g, b])

        progress.finish()

    def _ray_color(self, ray, depth, world):
        """计算光线颜色"""
        from .hittable import HitRecord
        from .interval import Interval
        from .constants import INFINITY
        from .vec3 import unit_vector

        # 超过反弹限制，返回黑色
        if depth <= 0:
            return Color(0, 0, 0)

        rec = HitRecord()

        # 检测光线与物体相交
        if world.hit(ray, Interval(0.001, INFINITY), rec):
            scattered = Ray()
            attenuation = Color()

            # 计算材质散射
            if rec.mat and rec.mat.scatter(ray, rec, attenuation, scattered):
                return attenuation * self._ray_color(scattered, depth - 1, world)
            return Color(0, 0, 0)

        # 背景渐变色
        unit_direction = unit_vector(ray.direction)
        t = 0.5 * (unit_direction.y + 1.0)
        return (1.0 - t) * Color(1.0, 1.0, 1.0) + t * Color(0.5, 0.7, 1.0)

    