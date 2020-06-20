import taichi as ti
ti.init(arch=ti.gpu)

img = ti.var(ti.f32, (512, 512, 3))  # 分辨率 512x512，RGB 三通道

@ti.kernel
def paint():
    for i, j, in ti.ndrange(512, 512):  # 并行执行
        # 红色从左到右，蓝色色从下到上，亮度从0.0增加到1.0
        img[i, j, 0] = i / 512 #左上蓝色
        img[i, j, 2] = j / 512 #右下红色
        img[i, j, 1] = 0

paint()  # 填充张量内容
ti.imshow(img)  # 将张量作为图像显示
