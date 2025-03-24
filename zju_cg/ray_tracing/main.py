import taichi as ti
import time
import sys

# 在导入任何其他模块之前初始化Taichi
ti.init(arch=ti.gpu)  # 改用CPU模式避免CUDA问题

from utils import (
    Vec3,
    Point3,
    Color,
    Ray,
    create_image_buffer,
    save_image,
    save_as_ppm,
    HittableList,
    Sphere,
    Lambertian,
    Metal,
    Dielectric,
    Camera,
    random_double,
)


def test_lambertian():
    """测试漫反射（Lambertian）材质"""
    world = HittableList()

    # 创建材质和球体
    material = Lambertian(Color(0.7, 0.3, 0.3))  # 粉红色漫反射
    world.add(Sphere(Point3(0, 0, -1), 0.5, material))  # 中央球体

    # 添加地面
    ground_material = Lambertian(Color(0.8, 0.8, 0.0))  # 黄色地面
    world.add(Sphere(Point3(0, -100.5, -1), 100, ground_material))

    # 设置相机
    cam = Camera()
    cam.aspect_ratio = 16.0 / 9.0
    cam.image_width = 400
    cam.samples_per_pixel = 5  # 增加采样以便更好观察漫反射效果
    cam.max_depth = 50
    cam.lookfrom = Point3(0, 0, 0)
    cam.lookat = Point3(0, 0, -1)
    cam.vup = Vec3(0, 1, 0)
    cam.vfov = 90
    cam.defocus_angle = 0  # 无景深效果

    # 创建图像缓冲区并渲染
    image = create_image_buffer(
        cam.image_width, int(cam.image_width / cam.aspect_ratio)
    )
    cam.render(world, image)
    save_image(image, "lambertian_test.png")

    return image


def test_metal():
    """测试金属（Metal）材质"""
    world = HittableList()

    # 创建两个金属球体，一个光滑，一个模糊
    material_smooth = Metal(Color(0.8, 0.8, 0.8), 0.0)  # 光滑金属
    material_fuzzy = Metal(Color(0.8, 0.6, 0.2), 0.5)  # 模糊金属（铜色）

    # 调整位置，增加球体之间的距离
    world.add(Sphere(Point3(-1, 0, -1), 0.5, material_smooth))  # 左侧球体
    world.add(Sphere(Point3(1, 0, -1), 0.5, material_fuzzy))  # 右侧球体

    # 添加地面
    ground_material = Lambertian(Color(0.8, 0.8, 0.8))  # 灰色地面
    world.add(Sphere(Point3(0, -100.5, -1), 100, ground_material))

    # 添加一些背景球体用于反射，位置稍微调整
    background_material = Lambertian(Color(0.2, 0.3, 0.7))  # 蓝色背景球
    world.add(Sphere(Point3(0, 0.1, -2), 0.5, background_material))

    # 设置相机
    cam = Camera()
    cam.aspect_ratio = 16.0 / 9.0
    cam.image_width = 400
    cam.samples_per_pixel = 5
    cam.max_depth = 50
    cam.lookfrom = Point3(0, 0, 0)
    cam.lookat = Point3(0, 0, -1)
    cam.vup = Vec3(0, 1, 0)
    cam.vfov = 90

    # 创建图像缓冲区并渲染
    image = create_image_buffer(
        cam.image_width, int(cam.image_width / cam.aspect_ratio)
    )
    cam.render(world, image)
    save_image(image, "metal_test.png")

    return image


def test_dielectric():
    """测试透明介质（Dielectric）材质"""
    world = HittableList()

    # 创建玻璃球体
    glass_material = Dielectric(1.5)  # 玻璃折射率约为1.5

    # 调整位置，增加球体之间的距离
    # 实心玻璃球
    world.add(Sphere(Point3(-1, 0, -1), 0.5, glass_material))

    # 中空玻璃球（使用负半径）
    world.add(Sphere(Point3(1, 0, -1), 0.5, glass_material))
    world.add(Sphere(Point3(1, 0, -1), -0.45, glass_material))

    # 添加地面
    ground_material = Lambertian(Color(0.8, 0.8, 0.8))
    world.add(Sphere(Point3(0, -100.5, -1), 100, ground_material))

    # 添加背景物体，便于观察折射效果
    background_material = Lambertian(Color(0.1, 0.2, 0.5))
    world.add(Sphere(Point3(0, 0.1, -2), 0.5, background_material))

    # 设置相机
    cam = Camera()
    cam.aspect_ratio = 16.0 / 9.0
    cam.image_width = 400
    cam.samples_per_pixel = 5
    cam.max_depth = 50
    cam.lookfrom = Point3(0, 0, 0)
    cam.lookat = Point3(0, 0, -1)
    cam.vup = Vec3(0, 1, 0)
    cam.vfov = 90

    # 创建图像缓冲区并渲染
    image = create_image_buffer(
        cam.image_width, int(cam.image_width / cam.aspect_ratio)
    )
    cam.render(world, image)
    save_image(image, "dielectric_test.png")

    return image


def test_combined_materials():
    """测试组合所有材质的简单场景"""
    world = HittableList()

    # 创建各种材质
    lambertian_material = Lambertian(Color(0.7, 0.3, 0.3))
    metal_material = Metal(Color(0.8, 0.8, 0.8), 0.3)
    glass_material = Dielectric(1.5)

    # 添加球体
    world.add(
        Sphere(Point3(0, -100.5, -1), 100, Lambertian(Color(0.8, 0.8, 0.0)))
    )  # 地面
    world.add(Sphere(Point3(-1, 0, -1), 0.5, lambertian_material))  # 左侧球体
    world.add(Sphere(Point3(0, 0, -1), 0.5, metal_material))  # 中间球体
    world.add(Sphere(Point3(1, 0, -1), 0.5, glass_material))  # 右侧球体

    # 设置相机
    cam = Camera()
    cam.aspect_ratio = 16.0 / 9.0
    cam.image_width = 400
    cam.samples_per_pixel = 5
    cam.max_depth = 50
    cam.lookfrom = Point3(0, 0, 0)
    cam.lookat = Point3(0, 0, -1)
    cam.vup = Vec3(0, 1, 0)
    cam.vfov = 90

    # 创建图像缓冲区并渲染
    image = create_image_buffer(
        cam.image_width, int(cam.image_width / cam.aspect_ratio)
    )
    cam.render(world, image)
    save_image(image, "combined_materials_test.png")

    return image


def test_camera_position():
    """测试相机位置和视角对场景的影响"""
    world = HittableList()

    # 创建各种材质
    lambertian_material = Lambertian(Color(0.7, 0.3, 0.3))
    metal_material = Metal(Color(0.8, 0.8, 0.8), 0.3)
    glass_material = Dielectric(1.5)

    # 添加球体
    world.add(
        Sphere(Point3(0, -100.5, -1), 100, Lambertian(Color(0.8, 0.8, 0.0)))
    )  # 地面
    world.add(Sphere(Point3(-1, 0, -1), 0.5, lambertian_material))  # 左侧球体
    world.add(Sphere(Point3(0, 0, -1), 0.5, metal_material))  # 中间球体
    world.add(Sphere(Point3(1, 0, -1), 0.5, glass_material))  # 右侧球体

    # 设置相机在侧面观察
    cam = Camera()
    cam.aspect_ratio = 16.0 / 9.0
    cam.image_width = 400
    cam.samples_per_pixel = 10
    cam.max_depth = 50
    cam.lookfrom = Point3(-2, 1, 1)  # 相机位置在场景左上方
    cam.lookat = Point3(0, 0, -1)  # 看向场景中心
    cam.vup = Vec3(0, 1, 0)
    cam.vfov = 90
    cam.defocus_angle = 0.1  # 添加一点景深效果
    cam.focus_dist = 3.0

    # 创建图像缓冲区并渲染
    image = create_image_buffer(
        cam.image_width, int(cam.image_width / cam.aspect_ratio)
    )
    cam.render(world, image)
    save_image(image, "camera_position_test.png")

    return image


def main():
    start_time = time.time()

    print("开始材质测试...")

    # 测试漫反射材质
    print("测试漫反射材质...")
    test_lambertian()
    print("漫反射材质测试完成")

    # 测试金属材质
    print("测试金属材质...")
    test_metal()
    print("金属材质测试完成")

    # 测试透明介质材质
    print("测试透明介质材质...")
    test_dielectric()
    print("透明介质材质测试完成")

    # 测试组合材质
    print("测试组合材质场景...")
    test_combined_materials()
    print("组合材质测试完成")

    # 测试相机位置
    print("测试相机位置和视角...")
    test_camera_position()
    print("相机位置测试完成")

    end_time = time.time()
    print(f"所有测试完成! 总耗时: {end_time - start_time:.2f} 秒")


if __name__ == "__main__":
    main()
