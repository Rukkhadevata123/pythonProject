import argparse


def add_basic_options(parser):
    """添加基础渲染选项"""
    parser.add_argument("--obj", type=str, required=True, help="OBJ文件路径")
    parser.add_argument(
        "--output", type=str, required=True, help="输出文件名 (不含扩展名)"
    )
    parser.add_argument("--width", type=int, default=800, help="输出图像宽度")
    parser.add_argument("--height", type=int, default=800, help="输出图像高度")
    parser.add_argument(
        "--projection",
        type=str,
        default="perspective",
        choices=["perspective", "orthographic"],
        help="投影类型",
    )
    parser.add_argument("--focal", type=float, default=2.0, help="透视投影焦距")
    parser.add_argument(
        "--angle", type=float, default=0, help="绕Y轴旋转角度 (用于单帧或模型动画)"
    )
    parser.add_argument("--output-dir", type=str, default="output", help="输出目录")
    parser.add_argument("--no-zbuffer", action="store_true", help="禁用Z-buffer")
    parser.add_argument(
        "--colorize", action="store_true", help="为每个面启用随机颜色 (覆盖材质)"
    )


def add_texture_options(parser):
    """添加纹理相关选项"""
    parser.add_argument("--texture", type=str, help="纹理图像路径 (优先于程序化纹理)")
    parser.add_argument("--no-texture", action="store_true", help="完全禁用纹理")
    parser.add_argument(
        "--texture-type",
        type=str,
        default="checkerboard",
        choices=["checkerboard", "gradient", "noise"],
        help="程序化纹理类型 (若未提供--texture)",
    )
    parser.add_argument("--texture-size", type=int, default=512, help="程序化纹理大小")
    parser.add_argument(
        "--no-materials", action="store_true", help="禁用材质文件 (.mtl) 加载"
    )


def add_lighting_options(parser):
    """添加光照相关选项"""
    parser.add_argument("--no-lighting", action="store_true", help="禁用光照计算")
    parser.add_argument(
        "--light-model",
        type=str,
        default="blinn-phong",
        choices=["phong", "blinn-phong"],
        help="光照模型",
    )
    parser.add_argument("--ambient", type=float, default=0.2, help="环境光强度")
    parser.add_argument("--diffuse", type=float, default=0.6, help="漫反射强度")
    parser.add_argument("--specular", type=float, default=0.2, help="高光强度")
    parser.add_argument("--shininess", type=float, default=32.0, help="高光锐度")
    parser.add_argument(
        "--light-type",
        type=str,
        default="directional",
        choices=["directional", "point"],
        help="光源类型",
    )
    parser.add_argument(
        "--light-dir", type=str, default="1,-1,1", help="方向光方向 (x,y,z)"
    )
    parser.add_argument(
        "--light-pos", type=str, default="0,0,3", help="点光源初始位置 (x,y,z)"
    )
    parser.add_argument(
        "--light-atten",
        type=str,
        default="1.0,0.09,0.032",
        help="点光源衰减系数 (常量,线性,平方)",
    )


def add_animation_options(parser):
    """添加动画相关选项"""
    parser.add_argument(
        "--light-animation",
        type=str,
        default="none",
        choices=[
            "none",
            "vertical",
            "horizontal",
            "circular",
            "pulse",
            "figure8",
            "spiral",
            "custom",
        ],
        help="光源动画类型",
    )
    parser.add_argument(
        "--light-range", type=float, default=1.0, help="光源动画移动范围/幅度"
    )
    parser.add_argument(
        "--light-frame", type=int, default=0, help="当前动画帧号 (用于生成序列)"
    )
    parser.add_argument(
        "--total-frames", type=int, default=1, help="动画总帧数 (大于1时启用动画)"
    )
    parser.add_argument(
        "--custom-x-expr",
        type=str,
        default="sin(2*pi*t)",
        help="光源X坐标自定义表达式 (t=0..1)",
    )
    parser.add_argument(
        "--custom-y-expr",
        type=str,
        default="cos(2*pi*t)",
        help="光源Y坐标自定义表达式 (t=0..1)",
    )
    parser.add_argument(
        "--custom-z-expr", type=str, default="0", help="光源Z坐标自定义表达式 (t=0..1)"
    )


def add_depth_options(parser):
    """添加深度图相关选项"""
    parser.add_argument("--no-depth", action="store_true", help="不生成深度图")
    parser.add_argument("--depth-min", type=int, default=1, help="深度归一化最小百分位")
    parser.add_argument(
        "--depth-max", type=int, default=99, help="深度归一化最大百分位"
    )


def add_camera_options(parser):
    """添加相机相关选项"""
    parser.add_argument(
        "--animation-type",
        type=str,
        default="model",
        choices=["model", "light", "camera"],
        help="动画类型: model(旋转模型), light(移动光源), camera(相机动画)",
    )
    parser.add_argument(
        "--camera-type",
        type=str,
        default="orbit",
        choices=["yaw", "pitch", "roll", "orbit"],
        help="相机动画类型: yaw(左右摇头), pitch(抬头低头), roll(歪头), orbit(环绕)",
    )
    parser.add_argument(
        "--camera-from", type=str, default="0,0,3", help="相机位置坐标 (x,y,z)"
    )
    parser.add_argument(
        "--camera-at", type=str, default="0,0,0", help="相机观察点坐标 (x,y,z)"
    )
    parser.add_argument(
        "--camera-up", type=str, default="0,1,0", help="相机上方向 (x,y,z)"
    )
    parser.add_argument(
        "--camera-fov", type=float, default=45.0, help="相机视场角 (度)"
    )
    parser.add_argument("--frame", type=int, default=0, help="当前帧号 (用于相机动画)")


def setup_parser():
    """创建并配置 ArgumentParser"""
    parser = argparse.ArgumentParser(description="Taichi GPU加速三角形渲染器")
    add_basic_options(parser)
    add_texture_options(parser)
    add_lighting_options(parser)
    add_animation_options(parser)
    add_depth_options(parser)
    add_camera_options(parser)  # 新增
    return parser
