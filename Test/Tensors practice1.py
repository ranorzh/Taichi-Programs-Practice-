import taichi as ti 
ti.init()
print('一维张量：')
x=ti.var(ti.i32,shape=4)#类似于C语言里的一维数组，期数据类型为int32,长度为4
x[1]=12
print('x[1]=',x[1])
arr = x.to_numpy()
print(arr)

print('kernei中一维张量:')
@ti.kernel  #声明Taichi的装饰器kernel 
def foreach_array():
    for i in x:
        x[i]=i
foreach_array()
print(x.to_numpy())

def hello(i:ti.i32):
    a=500
    print('Hello World!',a+i)
hello(20)

def accumulate() ->ti.i32:
    print('accumulate:1+2+...+9:')
    sum=0
    for i in range(10):
        sum+=i #原子操作
    print('sum=',sum)
accumulate()

def calculate_PI() ->ti.f32:
    sum=0.0
    for i in range(666666):
        n=2*i+1
        sum += pow(-1,i)/n #原子操作
    return sum*4
print('PI=',calculate_PI())

