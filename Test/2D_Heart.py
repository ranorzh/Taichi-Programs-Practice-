import taichi as ti
import numpy as np

ti.init(arch=ti.gpu)

Width, Height = 512, 512
pixels = ti.Vector(3, dt=ti.f32, shape=(Width, Height))

@ti.func
def mod(x,y):
    return x-y*ti.floor(x/y)

@ti.func
def clamp(x, a_min, a_max):
    return min(max(x,a_min), a_max)

@ti.func
def smoothstep(edge0, edge1, x):
    t = clamp((x-edge0) / (edge1-edge0), 0.0, 1.0)
    return t*t*(3.0-2.0*t)

@ti.func
def mix(x ,y, a):
    return x*(1.0-a) + y*a

@ti.kernel
def paint(t:ti.f32) :

    for i, j in pixels:
        p = ti.Vector([2*i-Width, 2*j-Height])/min(Width,Height)

        #background-color
        bcol = ti.Vector([1.0,0.8,0.7-0.07*p[1]]) * (1.0-0.25*p.norm())

        #animate
        tt = mod(t, 1.5)/1.5
        ss = pow(tt, 0.2)*0.5+0.5
        ss = 1.0 + ss*0.5*ti.sin(tt*6.2831*3.0+p[1]*0.5)*ti.exp(-4.0*tt)
        p *= ti.Vector([0.5, 1.5]) + ss*ti.Vector([0.5, -0.5])  

        #shape
        p[1] -= 0.25
        a = ti.atan2(p[0], p[1]) / 3.141593
        r = p.norm()
        h = abs(a)
        d = (13.0*h - 22.0*h*h + 10.0*h*h*h)/(6.0-5.0*h)

        #color
        s = 0.75 + 0.75*p[0]
        s *= 1.0 - 0.4*r
        s = 0.3 + 0.7*s
        s *= 0.5 + 0.5*pow(1.0-clamp(r/d, 0.0, 1.0), 1.0)

        hcol = ti.Vector([1.0, 0.5*r, 0.3])*s
        pixels[i,j] = mix(bcol , hcol ,smoothstep(-0.01, 0.01, d-r))

gui = ti.GUI("2D_Heart", res = (Width,Height))

for z in range(1000000):
    paint(z*0.03)
    gui.set_image(pixels.to_numpy())
    gui.show()
