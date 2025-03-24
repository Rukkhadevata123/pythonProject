import taichi as ti

ti.init(arch=ti.cpu)

# 定义一个字段来存储线段
MAX_LINES = 100000  # 根据预期的线段数量进行调整
lines = ti.Vector.field(4, dtype=ti.f32, shape=MAX_LINES)  # 每条线段的 [x1, y1, x2, y2]
line_count = ti.field(dtype=ti.i32, shape=())  # 用于跟踪线段数量

# 定义用于迭代计算的栈
stack = ti.field(dtype=ti.f32, shape=(10000, 5))  # [x1, y1, x2, y2, depth]
stack_top = ti.field(dtype=ti.i32, shape=())

@ti.func
def push(x1, y1, x2, y2, depth):
    idx = ti.atomic_add(stack_top[None], 1)
    stack[idx, 0] = x1
    stack[idx, 1] = y1
    stack[idx, 2] = x2
    stack[idx, 3] = y2
    stack[idx, 4] = depth

@ti.func
def pop():
    stack_top[None] -= 1
    idx = stack_top[None]
    return stack[idx, 0], stack[idx, 1], stack[idx, 2], stack[idx, 3], stack[idx, 4]

@ti.func
def add_line(x1, y1, x2, y2):
    idx = ti.atomic_add(line_count[None], 1)
    lines[idx] = ti.Vector([x1 / 800, y1 / 800, x2 / 800, y2 / 800])

# 关键函数，用于迭代绘制科克雪花
@ti.kernel
def drawKochIterative(x1: ti.f32, y1: ti.f32, x2: ti.f32, y2: ti.f32, depth: ti.i32):
    stack_top[None] = 0
    push(x1, y1, x2, y2, depth)

    while stack_top[None] > 0:
        lx1, ly1, lx2, ly2, ldepth = pop()

        if ldepth == 0:
            add_line(lx1, ly1, lx2, ly2)
        else: # 这是数学公式的实现
            dx = lx2 - lx1
            dy = ly2 - ly1
            x3 = lx1 + dx / 3
            y3 = ly1 + dy / 3
            x4 = lx1 + dx / 2 - dy * ti.sqrt(3) / 6
            y4 = ly1 + dy / 2 + dx * ti.sqrt(3) / 6
            x5 = lx1 + 2 * dx / 3
            y5 = ly1 + 2 * dy / 3

            push(x5, y5, lx2, ly2, ldepth - 1)
            push(x4, y4, x5, y5, ldepth - 1)
            push(x3, y3, x4, y4, ldepth - 1)
            push(lx1, ly1, x3, y3, ldepth - 1)

# def drawKochSnowflake(x, y, size, depth):
#     height = size * (3 ** 0.5) / 2
#     x1 = x - size / 2
#     y1 = y - height / 3
#     x2 = x + size / 2
#     y2 = y - height / 3
#     x3 = x
#     y3 = y + 2 * height / 3

#     drawKochIterative(x1, y1, x2, y2, depth)
#     drawKochIterative(x2, y2, x3, y3, depth)
#     drawKochIterative(x3, y3, x1, y1, depth)

def drawKochSnowflake(x, y, size, depth):
    height = size * (3 ** 0.5) / 2
    x1 = x - size / 2
    y1 = y + height / 3  # 将高度调整为正值
    x2 = x + size / 2
    y2 = y + height / 3  # 将高度调整为正值
    x3 = x
    y3 = y - 2 * height / 3  # 将高度调整为负值

    drawKochIterative(x1, y1, x2, y2, depth)
    drawKochIterative(x2, y2, x3, y3, depth)
    drawKochIterative(x3, y3, x1, y1, depth)

# 初始化 GUI
gui = ti.GUI("Koch Snowflake", res=(800, 800))

# 主循环
while gui.running:
    gui.clear(0x000000)  # 清屏
    line_count[None] = 0  # 重置线段计数
    drawKochSnowflake(400, 400, 300, 4)  # 绘制科克雪花

    for i in range(line_count[None]):
        line = lines[i]
        gui.line((line[0], line[1]), (line[2], line[3]), color=0xFFFFFF)

    gui.show()
