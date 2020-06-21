import taichi as ti
import random

ti.init(arch=ti.gpu)
num = 8
width = 1080
height = 720
speed = 2
isoV = 50


pixels = ti.var(dt=ti.f32, shape=(width, height))

vs = ti.Vector(2, dt=ti.f32, shape=num)  # position
randomX = ti.var(ti.f32, num)
randomY = ti.var(ti.f32, num)
randomR = ti.var(ti.f32, num)

for i in range(num):
    vs[i] = [random.randint(0, width), random.randint(0, height)]
    randomX[i] = random.randint(-speed, speed)
    randomY[i] = random.randint(-speed, speed)
    randomR[i] = random.randint(100, 200)


def move():
    for k in range(num):
        vs[k].x += randomX[k]
        vs[k].y += randomY[k]
        if vs[k].x - randomR[k] < 0:
            randomX[k] = abs(randomX[k])
        if vs[k].x + randomR[k] > width:
            randomX[k] = -abs(randomX[k])
        if vs[k].y - randomR[k] < 0:
            randomY[k] = abs(randomY[k])
        if vs[k].y + randomR[k] > height:
            randomY[k] = -abs(randomY[k])

@ti.kernel
def drawPixels(iso:ti.f32):
    for i, j in pixels:  # Parallized over all pixels
        fxy = 0
        for k in range(num):
            dx = i + 0.5 - vs[k].x
            dy = j + 0.5 - vs[k].y
            d2 = dx * dx + dy * dy
            ti.atomic_add(fxy,randomR[k] ** 2 / d2)

        if fxy >= iso:
            pixels[i,j] = 0.2
        else:
            pixels[i, j] = 1


gui = ti.GUI("Metaballs2d", (width, height))  # 设置gui

for i in range(1000000):
    move()
    drawPixels(isoV)
    gui.set_image(pixels,)  
    gui.show()
