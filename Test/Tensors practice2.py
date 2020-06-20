import taichi as ti
ti.init()

x=ti.var(ti.i32,shape=(3,2)) #A tensor of scalars
y=ti.Vector(3, dt=ti .f32 , shape=4) # A tensor of 4x 3D vectors 
z=ti.Matrix(2,2,dt=ti.f32,shape=(3,5)) # A tensor of 3x5 2x2 matrices

@ti.kernel
def foo():
    x[1,0]=1
    print('x[1,0]=',x[3,4])

    y[2] = [6,7,8,]
    print('y[0]=' ,y[0], ',y[1]=' ,y[1], ',y[2]=',y[2])

    z[0,0][0,1]=1
    print('z[2,1]=',z[0,0])
foo()
