import numpy as np
import matplotlib.pyplot as plt

# 定义用户窗口（数学/世界坐标系）
user_window = {"xmin": 0, "xmax": 2 * np.pi, "ymin": -1.2, "ymax": 1.2}

# 定义设备窗口（设备坐标系）
# 使用模拟的800x600像素显示器
device_window = {"xmin": 0, "xmax": 800, "ymin": 0, "ymax": 600}

# 标准化设备坐标(NDC)范围
ndc = {"xmin": -1, "xmax": 1, "ymin": -1, "ymax": 1}


def transform_point(point, transform_matrix):
    """使用变换矩阵转换点坐标"""
    # 转换为齐次坐标
    homogeneous_point = np.array([point[0], point[1], 1])
    # 应用变换
    transformed = np.dot(transform_matrix, homogeneous_point)
    return transformed[0], transformed[1]


def create_user_to_ndc_matrix(user_window):
    """创建从用户坐标到NDC的变换矩阵"""
    sx = 2.0 / (user_window["xmax"] - user_window["xmin"])
    sy = 2.0 / (user_window["ymax"] - user_window["ymin"])
    tx = -1 - sx * user_window["xmin"]
    ty = -1 - sy * user_window["ymin"]

    return np.array([[sx, 0, tx], [0, sy, ty], [0, 0, 1]])


def create_ndc_to_device_matrix(device_window):
    """创建从NDC到设备坐标的变换矩阵"""
    sx = (device_window["xmax"] - device_window["xmin"]) / 2.0
    sy = (device_window["ymax"] - device_window["ymin"]) / 2.0
    tx = device_window["xmin"] + sx
    ty = device_window["ymin"] + sy

    # 注意：需要翻转y轴（设备坐标的y轴向下）
    return np.array([[sx, 0, tx], [0, -sy, ty], [0, 0, 1]])


# 创建变换矩阵
user_to_ndc_matrix = create_user_to_ndc_matrix(user_window)
ndc_to_device_matrix = create_ndc_to_device_matrix(device_window)
user_to_device_matrix = np.dot(ndc_to_device_matrix, user_to_ndc_matrix)

# 在用户坐标系中生成点（Sin(x)曲线，x∈[0,2π]）
num_points = 100
x_user = np.linspace(user_window["xmin"], user_window["xmax"], num_points)
y_user = np.sin(x_user)

# 将用户坐标转换为设备坐标
x_device = []
y_device = []
for i in range(num_points):
    x_d, y_d = transform_point((x_user[i], y_user[i]), user_to_device_matrix)
    x_device.append(x_d)
    y_device.append(y_d)

# 创建一个展示原始曲线和变换后曲线的图形
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# 绘制用户坐标系中的Sin(x)曲线
ax1.plot(x_user, y_user, "b-")
ax1.set_xlim(user_window["xmin"], user_window["xmax"])
ax1.set_ylim(user_window["ymin"], user_window["ymax"])
ax1.set_title("Sin(x) Curve in User Window")
ax1.set_xlabel("x")
ax1.set_ylabel("Sin(x)")
ax1.grid(True)
ax1.axhline(y=0, color="k", linestyle="-", alpha=0.3)
ax1.axvline(x=0, color="k", linestyle="-", alpha=0.3)

# 绘制设备坐标系中的Sin(x)曲线
ax2.plot(x_device, y_device, "r-")
ax2.set_xlim(device_window["xmin"], device_window["xmax"])
ax2.set_ylim(device_window["ymax"], device_window["ymin"])  # 注意：y轴方向
ax2.set_title("Sin(x) Curve in Device Window")
ax2.set_xlabel("x (pixels)")
ax2.set_ylabel("y (pixels)")
ax2.grid(True)

# 添加变换信息
plt.figtext(
    0.5,
    0.01,
    f"User Window: [0, 2π] × [-1.2, 1.2] → Device Window: [0, 800] × [0, 600]",
    ha="center",
    fontsize=12,
)

plt.tight_layout()
plt.savefig("user_to_device_transform.png", dpi=300, bbox_inches="tight")
plt.show()

# 生成NDC坐标用于可视化
x_ndc = []
y_ndc = []
for i in range(num_points):
    x_n, y_n = transform_point((x_user[i], y_user[i]), user_to_ndc_matrix)
    x_ndc.append(x_n)
    y_ndc.append(y_n)

# 创建一个展示NDC中曲线的图形
plt.figure(figsize=(7, 7))
plt.plot(x_ndc, y_ndc, "g-")
plt.xlim(ndc["xmin"], ndc["xmax"])
plt.ylim(ndc["ymin"], ndc["ymax"])
plt.title("Sin(x) Curve in Normalized Device Coordinates (NDC)")
plt.xlabel("x (NDC)")
plt.ylabel("y (NDC)")
plt.grid(True)
plt.axhline(y=0, color="k", linestyle="-", alpha=0.3)
plt.axvline(x=0, color="k", linestyle="-", alpha=0.3)

plt.tight_layout()
plt.savefig("ndc_transform.png", dpi=300, bbox_inches="tight")
plt.show()
