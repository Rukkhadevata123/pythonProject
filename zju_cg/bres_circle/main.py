import numpy as np
import matplotlib.pyplot as plt

def bresenham_reverse_arc(R):
    # 使用Bresenham算法绘制第一象限的逆圆弧（从(R,0)到(0,R)）
    points = []
    x = R
    y = 0
    # 初始决策参数D = x² + y² - R² = 0
    D = 0
    
    # 循环直到x <= 0
    while x > 0:
        # 绘制当前点
        points.append((x, y))
        
        # 计算候选点的距离
        Du = D + 2*y + 1  # 上方点
        Dlu = D + 2*y - 2*x + 2  # 左上方点
        Dl = D - 2*x + 1  # 左方点
        
        # 根据D的值选择下一个点
        if D < 0:
            # 点在圆内，选择U或LU
            dULU = -2*x + 1  # |Du| - |Dlu| = Dlu - Du
            if dULU <= 0:
                # 选择U点
                y += 1
                D = Du
            else:
                # 选择LU点
                x -= 1
                y += 1
                D = Dlu
        elif D > 0:
            # 点在圆外，选择LU或L
            dLUL = 2*y + 1  # |Dlu| - |Dl| = Dlu - Dl
            if dLUL <= 0:
                # 选择LU点
                x -= 1
                y += 1
                D = Dlu
            else:
                # 选择L点
                x -= 1
                D = Dl
        else:
            # 点在圆上，选择LU点
            x -= 1
            y += 1
            D = Dlu
    
    # 添加最后一个点(0,R)
    if (0, R) not in points:
        points.append((0, R))
    
    return points

def main():
    # 设置较大的半径
    R = 1000000
    
    # 获取圆弧上的点
    arc_points = bresenham_reverse_arc(R)
    
    # 将点列表转换为x和y坐标数组
    x_coords = [point[0] for point in arc_points]
    y_coords = [point[1] for point in arc_points]
    
    # 设置图像大小
    plt.figure(figsize=(8, 8))
    
    # 绘制圆弧点
    plt.scatter(x_coords, y_coords, s=1, color='blue')
    
    # 设置图像属性
    plt.axis('equal')  # 确保x和y轴比例相同
    plt.axis('off')    # 关闭坐标轴
    
    # 限制坐标范围以便只显示第一象限
    plt.xlim(-5, R+5)
    plt.ylim(-5, R+5)
    
    # 保存图像
    plt.savefig('bresenham_reverse_arc.png', dpi=300, bbox_inches='tight')
    
    print(f"绘制完成，共生成了{len(arc_points)}个点")

if __name__ == "__main__":
    main()