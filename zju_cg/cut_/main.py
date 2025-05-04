import matplotlib.pyplot as plt
import random

# 这些是判断点在窗口内外的编码
INSIDE = 0  # 0000
LEFT = 1  # 0001
RIGHT = 2  # 0010
BOTTOM = 4  # 0100
TOP = 8  # 1000


def compute_outcode(x, y, xmin, ymin, xmax, ymax):
    # 计算点的编码
    code = INSIDE
    if x < xmin:  # 在窗口左侧
        code |= LEFT
    elif x > xmax:  # 在窗口右侧
        code |= RIGHT

    if y < ymin:  # 在窗口下方
        code |= BOTTOM
    elif y > ymax:  # 在窗口上方
        code |= TOP

    return code


def cohen_sutherland_clip(x0, y0, x1, y1, xmin, ymin, xmax, ymax):
    # Cohen-Sutherland裁剪算法
    outcode0 = compute_outcode(x0, y0, xmin, ymin, xmax, ymax)
    outcode1 = compute_outcode(x1, y1, xmin, ymin, xmax, ymax)
    accept = False

    while True:
        # 完全在裁剪窗口内部
        if not (outcode0 | outcode1):
            accept = True
            break

        # 完全在裁剪窗口外部(同侧)
        elif outcode0 & outcode1:
            break

        # 部分在窗口内，需要裁剪
        else:
            # 选择一个在窗口外的点
            outcode_out = outcode1 if outcode1 > outcode0 else outcode0

            # 计算与窗口边界的交点
            if outcode_out & TOP:  # 与上边界相交
                x = x0 + (x1 - x0) * (ymax - y0) / (y1 - y0)
                y = ymax
            elif outcode_out & BOTTOM:  # 与下边界相交
                x = x0 + (x1 - x0) * (ymin - y0) / (y1 - y0)
                y = ymin
            elif outcode_out & RIGHT:  # 与右边界相交
                y = y0 + (y1 - y0) * (xmax - x0) / (x1 - x0)
                x = xmax
            elif outcode_out & LEFT:  # 与左边界相交
                y = y0 + (y1 - y0) * (xmin - x0) / (x1 - x0)
                x = xmin
            else:
                raise ValueError("Invalid outcode")

            # 更新端点和编码
            if outcode_out == outcode0:
                x0, y0 = x, y
                outcode0 = compute_outcode(x0, y0, xmin, ymin, xmax, ymax)
            else:
                x1, y1 = x, y
                outcode1 = compute_outcode(x1, y1, xmin, ymin, xmax, ymax)

    return (x0, y0, x1, y1, accept)


def draw_line(ax, x0, y0, x1, y1, color="blue", linewidth=1, alpha=1.0, linestyle="-"):
    ax.plot(
        [x0, x1],
        [y0, y1],
        color=color,
        linewidth=linewidth,
        alpha=alpha,
        linestyle=linestyle,
    )


def generate_test_cases():
    """生成测试用例"""
    # 窗口范围
    xmin, ymin = 50, 50
    xmax, ymax = 350, 250

    # 测试用例：完全在窗口内、完全在窗口外、与窗口相交
    test_cases = [
        # 完全在窗口内的线段
        (100, 100, 300, 200),
        (150, 150, 250, 200),
        # 完全在窗口外的线段
        (10, 10, 40, 40),  # 左下方
        (400, 300, 450, 350),  # 右上方
        (10, 300, 40, 350),  # 左上方
        (400, 10, 450, 40),  # 右下方
        (0, 150, 40, 150),  # 左边平行
        (360, 150, 400, 150),  # 右边平行
        (200, 0, 200, 40),  # 下方平行
        (200, 260, 200, 300),  # 上方平行
        # 与窗口相交的线段
        (0, 0, 400, 300),  # 从左下到右上
        (0, 300, 400, 0),  # 从左上到右下
        (200, 0, 200, 300),  # 垂直穿过
        (0, 150, 400, 150),  # 水平穿过
        (30, 100, 300, 100),  # 从左边进入
        (100, 30, 100, 200),  # 从底部进入
        (370, 100, 300, 100),  # 从右边进入
        (100, 270, 100, 200),  # 从顶部进入
    ]

    # 添加一些随机线段作为附加测试
    for _ in range(10):
        x0, y0 = random.randint(0, 400), random.randint(0, 300)
        x1, y1 = random.randint(0, 400), random.randint(0, 300)
        test_cases.append((x0, y0, x1, y1))

    return test_cases, (xmin, ymin, xmax, ymax)


def visualize_clipping(test_cases, window):
    xmin, ymin, xmax, ymax = window

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # 第一个子图：显示原始线段和裁剪窗口
    ax1.set_title("Original Line Segments")
    ax1.set_xlim(0, 400)
    ax1.set_ylim(0, 300)

    window_rect = plt.Rectangle(
        (xmin, ymin),
        xmax - xmin,
        ymax - ymin,
        fill=False,
        edgecolor="green",
        linewidth=2,
    )
    ax1.add_patch(window_rect)
    for x0, y0, x1, y1 in test_cases:
        draw_line(ax1, x0, y0, x1, y1, color="blue", alpha=0.7)

    # 第二个子图：显示裁剪后的线段
    ax2.set_title("Clipped Line Segments")
    ax2.set_xlim(0, 400)
    ax2.set_ylim(0, 300)

    window_rect = plt.Rectangle(
        (xmin, ymin),
        xmax - xmin,
        ymax - ymin,
        fill=False,
        edgecolor="green",
        linewidth=2,
    )
    ax2.add_patch(window_rect)
    for x0, y0, x1, y1 in test_cases:
        draw_line(ax2, x0, y0, x1, y1, color="lightgrey", alpha=0.4, linestyle="--")
        cx0, cy0, cx1, cy1, accept = cohen_sutherland_clip(
            x0, y0, x1, y1, xmin, ymin, xmax, ymax
        )
        if accept:
            draw_line(ax2, cx0, cy0, cx1, cy1, color="red", linewidth=2)
    plt.tight_layout()
    plt.savefig("clipping_result.png", dpi=300)
    plt.show()


def main():
    # 生成测试用例并可视化
    test_cases, window = generate_test_cases()
    visualize_clipping(test_cases, window)


if __name__ == "__main__":
    main()
