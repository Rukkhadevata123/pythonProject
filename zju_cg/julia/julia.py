import taichi as ti
import taichi.math as tm
import numpy as np
import subprocess
import os

ti.init(arch=ti.gpu)

n = 320
pixels = ti.Vector.field(3, dtype=float, shape=(n * 2, n))  # 使用三通道颜色

@ti.func
def complex_sqr(z):
    return tm.vec2(z[0] * z[0] - z[1] * z[1], 2 * z[0] * z[1])

@ti.kernel
def paint(t: float):
    for i, j in pixels:
        c = tm.vec2(-0.8, tm.cos(t) * 0.2)
        z = tm.vec2(i / n - 1, j / n - 0.5) * 2
        iterations = 0
        while z.norm() < 20 and iterations < 50:
            z = complex_sqr(z) + c
            iterations += 1
        col = tm.vec3(0.5, 0.5, 0.5)  # 基础颜色
        if iterations < 50:
            smooth_iter = iterations + 1 - ti.log(ti.log(z.norm())) / ti.log(2)
            col = tm.vec3(0.5 + 0.5 * ti.sin(0.1 * smooth_iter + 0.0),
                          0.5 + 0.5 * ti.sin(0.1 * smooth_iter + 2.0),
                          0.5 + 0.5 * ti.sin(0.1 * smooth_iter + 4.0))  
        pixels[i, j] = col

def save_image(filename: str):
    img = np.zeros((n, n * 2, 3), dtype=np.float32)
    for i in range(n):
        for j in range(n * 2):
            img[i, j] = pixels[j, i]

    # 将图像数据转换为 uint8 类型，并缩放到 0-255 范围
    img = (img * 255).astype(np.uint8)

    import imageio
    imageio.imwrite(filename, img)

# 生成 PNG 图像
num_frames = 100
for i in range(num_frames):
    t = i * 0.03
    paint(t)
    filename = f"julia_{i:03d}.png"
    save_image(filename)
    print(f"图像 {filename} 已保存")

print("PNG 图像生成完毕")

# 调用 FFmpeg 将 PNG 图像转换为 GIF 动画
ffmpeg_path = "/usr/bin/ffmpeg"  # 替换为你的 FFmpeg 路径
output_gif = "julia.gif"

args = [
    "-framerate", "10",  # 调整帧率
    "-i", "julia_%03d.png",
    "-vf", "scale=320:-1:flags=lanczos",  # 调整 GIF 尺寸
    "-loop", "0",  # 循环播放
    output_gif
]

try:
    subprocess.run([ffmpeg_path] + args, check=True)
    print(f"GIF 动画 {output_gif} 生成成功！")
except subprocess.CalledProcessError as e:
    print(f"调用 FFmpeg 出错：{e}")
except FileNotFoundError:
    print(f"找不到 FFmpeg 可执行文件：{ffmpeg_path}")

# 清理 PNG 图像 (可选)
for i in range(num_frames):
    filename = f"julia_{i:03d}.png"
    os.remove(filename)
print("PNG 图像已删除")