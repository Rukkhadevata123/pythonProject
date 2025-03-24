import taichi as ti
ti.init(arch=ti.cpu)

window_size = 800
pixels = ti.field(dtype=float, shape=(window_size, window_size))

@ti.func
def draw_circle(center_x, center_y, radius):

    for i, j in ti.ndrange(window_size, window_size):
        # 计算当前像素到圆心的距离
        dist = ((i - center_x)**2 + (j - center_y)**2)**0.5
        if abs(dist - radius) <0.2:
            pixels[i, j] = 0.0  

@ti.kernel
def paint():
    
    pixels.fill(1)
    center_x = window_size // 2
    center_y = window_size // 2
    
    for i in range(10):
        radius = 400 - i*40
        draw_circle(center_x, center_y, radius)

gui = ti.GUI("Concentric Circles", res=(window_size, window_size))

while gui.running:
    paint()
    gui.set_image(pixels)
    gui.show()