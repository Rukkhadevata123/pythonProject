import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def point_comparison_line_2nd_quadrant(x1, y1, x2, y2):
    """
    直接比较法绘制第二象限的直线
    """
    x_a = x2 - x1  
    y_a = y2 - y1  

    # Initial error
    F = 0
    # Total steps
    N = abs(x_a) + abs(y_a)

    points = [(x1, y1)]
    x, y = x1, y1

    while N > 0:
        if F < 0:  
            y += 1
            F += abs(x_a)
        else:  
            x -= 1
            F -= abs(y_a)

        points.append((x, y))
        N -= 1

    return np.array(points)


def DDALine(x0, y0, x1, y1):
    """
    DDA算法绘制直线
    """
    dx = x1 - x0
    dy = y1 - y0
    k = dy / dx if dx != 0 else float("inf") 

    points = []
    if abs(k) > 1:
        # Use y as the step length
        y_step = 1 if y1 > y0 else -1
        x = x0
        for y in range(y0, y1 + y_step, y_step):
            points.append((round(x), y))
            x += 1 / k if k != 0 else 0
    else:
        # Use x as the step length
        x_step = 1 if x1 > x0 else -1
        y = y0
        for x in range(x0, x1 + x_step, x_step):
            points.append((x, round(y)))
            y += k

    return np.array(points)


def plot_simplified_2nd_quadrant(x1, y1, x2, y2):
    """
    Draw a simplified plot for the direct comparison method in the second quadrant
    """
    fig, ax = plt.subplots(figsize=(10, 10))

    # Generate approximated line using direct comparison method
    points = point_comparison_line_2nd_quadrant(x1, y1, x2, y2)

    # Draw the approximated line
    ax.plot(points[:, 0], points[:, 1], "r-", linewidth=1, alpha=0.7)

    # Add point coordinates as text
    ax.text(x1, y1, f"({x1}, {y1})", fontsize=12, ha="right", va="bottom")
    ax.text(x2, y2, f"({x2}, {y2})", fontsize=12, ha="right", va="bottom")

    # Remove axes and ticks
    ax.axis("off")

    # Equal aspect ratio
    ax.set_aspect("equal")

    # Save image
    plt.tight_layout()
    plt.savefig(
        "point_comparison_2nd_quadrant_simple.png", dpi=300, bbox_inches="tight"
    )
    plt.close()


def plot_simplified_dda(x0, y0, x1, y1):
    """
    Draw a simplified plot for the DDA algorithm
    """
    fig, ax = plt.subplots(figsize=(10, 10))

    # Generate approximated line using DDA algorithm
    points = DDALine(x0, y0, x1, y1)

    # Draw the approximated line
    ax.plot(points[:, 0], points[:, 1], "r-", linewidth=1, alpha=0.7)

    # Add point coordinates as text
    ax.text(x0, y0, f"({x0}, {y0})", fontsize=12, ha="right", va="bottom")
    ax.text(x1, y1, f"({x1}, {y1})", fontsize=12, ha="right", va="bottom")

    # Remove axes and ticks
    ax.axis("off")

    # Equal aspect ratio
    ax.set_aspect("equal")

    # Save image
    plt.tight_layout()
    plt.savefig("dda_line_simple.png", dpi=300, bbox_inches="tight")
    plt.close()


# Example 1: Direct comparison method in the second quadrant with large distance
x1, y1 = -100, 100  # Starting point (second quadrant)
x2, y2 = (
    -600,
    500,
)  # Ending point (second quadrant) - 500 point difference in x, 400 in y
plot_simplified_2nd_quadrant(x1, y1, x2, y2)

# Example 2: DDA algorithm example with large distance
# First quadrant example with slope = 2
x0, y0 = 100, 100  # Starting point
x1, y1 = 600, 1100  # Ending point - 500 point difference in x, 1000 in y (slope = 2)
plot_simplified_dda(x0, y0, x1, y1)
