import numpy as np
import matplotlib.pyplot as plt


def plot_3d_surface():
    x = np.linspace(-20, 20, 400)
    y = np.linspace(-20, 20, 400)
    X, Y = np.meshgrid(x, y)
    Z = 3 * X ** 2 + 2 * Y ** 2

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap='viridis')
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

    ax.set_label('X')
    ax.set_label('Y')
    ax.set_label('Z')
    ax.set_title('3D Surface Plot for $3X^2 + 2Y^2$')

    plt.show()
