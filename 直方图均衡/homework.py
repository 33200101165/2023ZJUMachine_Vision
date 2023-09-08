from matplotlib import pyplot as plt
import numpy as np
import cv2 as cv
img=cv.imread("1.jpg")
plt.figure(1)
plt.imshow(img)#绘制原图
height=img.shape[0]
width=img.shape[1]
histogram=np.zeros((1,256))
for row in range(height):
    for col in range(width):
        i = img[row][col]
        histogram[0][i] += 1
plt.figure(2)
plt.hist(img.ravel(), bins=256)
plt.xlabel('grayscale')
plt.ylabel('frequency')
plt.title('histogram')
plt.savefig('./histogram.jpg')
plt.show()
n_sum=height*width#像素点个数
Tr=np.zeros((1,256),dtype=np.float16)
for i in range(256):
    if i == 0:
        Tr[0][i] = histogram[0][i] /n_sum
    elif i == 255:
        Tr[0][i] = 1.0
    else:
        Tr[0][i] = histogram[0][i] / n_sum + Tr[0][i - 1]
x = list(range(0,256))
plt.figure(3)
plt.xlim(0, 255)
plt.ylim(0, 1.1)
plt.xlabel('grayscale')
plt.ylabel('p')
plt.title('Tr')
plt.plot(x, Tr[0, :], linewidth=0.5)
plt.savefig('./Tr.jpg')
plt.show()
Tr = Tr* 255# 累积分布函数计算完成后，把值域缩放到0~255的范围之内
f = np.zeros((1, 256), dtype=np.int32)  # 映射函数
a=0.5
for i in range(256):
    f[0][i] = a*Tr[0][i] +(1-a)*i
img_equal=np.zeros((img.shape))
img_equal=img.copy()
for row in range(height):
    for col in range(width):
        value = f[0][img[row][col]]
        img_equal[row][col] = value
cv.imwrite('./2.jpg', img_equal)
for row in range(height):
    for col in range(width):
        i = img_equal[row][col]
        img_equal[0][i] += 1
# 绘制直方图
plt.figure(4)
plt.imshow(img_equal)
plt.figure(5)
plt.hist(img_equal.ravel(), bins=256)
plt.xlabel('grayscale')
plt.ylabel('frequency')
plt.title('histogram_equal')
plt.savefig('./histogram_equal.jpg')
plt.show()
