import taichi as ti

PI = 3.141592653589793

ti.init(arch=ti.gpu)

n = 400
WINDOW_WIDTH = n * 2
WINDOW_HEIGHT = n

pixels = ti.var(dt=ti.f32, shape=(n * 2, n))


@ti.func
def alpha_blend(x:ti.i32, y:ti.i32, alpha:ti.f32, color):
    pixels[x, y] = pixels[x, y] * (1 - alpha) + color * alpha


@ti.func
def boxSDF(x:ti.i32, y:ti.i32, cx:ti.f32, cy:ti.f32, theta:ti.f32, w:ti.f32, h:ti.f32):
    w *= 0.5
    h *= 0.5
    costheta = ti.cos(theta)
    sintheta = ti.sin(theta)
    dx = abs((x - cx) * costheta + (y - cy) * sintheta) - w
    dy = abs((y - cy) * costheta - (x - cx) * sintheta) - h
    ax = max(dx, 0.0)
    ay = max(dy, 0.0)
    return min(max(dx, dy), 0.0) + ti.sqrt(ax ** 2 + ay ** 2)


@ti.func
def draw_box(cx:ti.f32, cy:ti.f32, theta:ti.f32, w:ti.f32, h:ti.f32, r:ti.f32):
    costheta = abs(ti.cos(theta))
    sintheta = abs(ti.sin(theta))
    tw = w * 0.5
    th = h * 0.5
    x0 = max(int(ti.floor(cx - tw * costheta - th * sintheta)) - 1, 0)
    x1 = min(int(ti.ceil( cx + tw * costheta + th * sintheta)) + 1, WINDOW_WIDTH)
    y0 = max(int(ti.floor(cy - tw * sintheta - th * costheta)) - 1, 0)
    y1 = min(int(ti.ceil( cy + tw * sintheta + th * costheta)) + 1, WINDOW_HEIGHT)
    w -= r * 2.0
    h -= r * 2.0
    for i, j in ti.ndrange((y0, y1), (x0, x1)):
        alpha_blend(j, i, max(min(0.5 - boxSDF(j, i, cx, cy, theta, w, h) + r, 1.0), 0.0), 0.8)

QUANTITY = 5

@ti.kernel
def paint(t: ti.f32):
    for i, j in pixels:
        pixels[i, j] = 0.15 - 0.05 * j / WINDOW_HEIGHT
    t = t % 2
    #angle = (t * PI + ti.sin(t * PI)) / 4
    #angle = PI * t * (1 + 3 * t - t**2) / 12
    angle = (PI * t * (4 + 3 * t - t**2)) / 24 - ti.sin(PI * t) / 8
    for i, j in ti.ndrange((1, QUANTITY), (1, 2 * QUANTITY )):
        draw_box(WINDOW_WIDTH * (j / QUANTITY / 2), WINDOW_HEIGHT * (i / QUANTITY), angle, 4, 30, 1.2)
        draw_box(WINDOW_WIDTH * (j / QUANTITY / 2), WINDOW_HEIGHT * (i / QUANTITY), angle + PI / 2, 4, 30, 1.2)

gui = ti.GUI("Crosses", res=(n * 2, n))

for i in range(1000000):
    paint(i * 0.08)
    gui.set_image(pixels)
    gui.show()
