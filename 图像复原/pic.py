import matplotlib.pyplot as plt
import numpy as np
from numpy import fft
import cv2
 
def Point_spread_function(image_size,motion_angle=45,offset=50):
    PSF = np.zeros(image_size)
    center_position=(image_size[0]-1)/2 
    print(image_size,center_position)
    for i in range(offset):
        offset_x=round(i*np.cos(motion_angle)) #对位移在x,y方向上分解
        offset_y=round(i*np.sin(motion_angle))   
        PSF[int(center_position+offset_y),int(center_position+offset_x)]=1
    return PSF / PSF.sum()  #对点扩散函数进行归一化亮度
 
def motion_blur(input, PSF):#对图片进行运动模糊
    input_fft = fft.fft2(input)# 进行二维数组的傅里叶变换
    PSF_fft = fft.fft2(PSF)
    blurred = fft.ifft2(input_fft * PSF_fft)
    blurred = np.abs(fft.fftshift(blurred))
    return blurred
 
def inverse(input, PSF, k):       # 逆滤波
    input_fft = fft.fft2(input)
    PSF_fft = fft.fft2(PSF) + k #克服分母等于0带来的计算问题
    result = fft.ifft2(input_fft / PSF_fft) #计算F(u,v)的傅里叶反变换
    result = np.abs(fft.fftshift(result))
    return result
 
def wiener(input,PSF,K=0.01):        #维纳滤波
    input_fft=fft.fft2(input)
    PSF_fft=fft.fft2(PSF) 
    PSF_fft_1=np.conj(PSF_fft) /(np.abs(PSF_fft)**2 + K)
    result=fft.ifft2(input_fft * PSF_fft_1)
    result=np.abs(fft.fftshift(result))
    return result
 
image = cv2.imread('1.jpg')
image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
img_h=image.shape[0]
img_w=image.shape[1]

plt.figure(1)
#进行运动模糊处理
plt.gray()
PSF = Point_spread_function((img_h,img_w))
plt.imshow(PSF)
plt.figure(2)
blurred = np.abs(motion_blur(image, PSF))
plt.subplot(121)
plt.imshow(image) 
plt.xlabel("Original Image")
plt.subplot(122)
plt.imshow(blurred)
plt.xlabel("Motion blurred")
 
plt.figure(3)
plt.gray()
plt.subplot(131)
result = inverse(blurred, PSF, 0.1)   #逆滤波
plt.imshow(result)
plt.xlabel("k=0.1")
plt.subplot(132)
result = inverse(blurred, PSF, 0.01)   #逆滤波
plt.imshow(result)
plt.title("inverse filter")
plt.xlabel("k=0.01")
plt.subplot(133)
result = inverse(blurred, PSF, 0.001)   #逆滤波
plt.imshow(result)
plt.xlabel("k=0.001")

plt.figure(4)
plt.subplot(131)
result=wiener(blurred,PSF)     #维纳滤波
plt.imshow(result)
plt.xlabel("K=0.01")
plt.subplot(132)
result=wiener(blurred,PSF,0.001)     #维纳滤波
plt.imshow(result)
plt.title("wiener filter")
plt.xlabel("K=0.001")
plt.subplot(133)
result=wiener(blurred,PSF,0.0001)     #维纳滤波
plt.imshow(result)
plt.xlabel("K=0.0001")

plt.figure(5)
PSF_long = Point_spread_function((img_h,img_w),45,60)
PSF_steep = Point_spread_function((img_h,img_w),50,50)
plt.subplot(131)
plt.imshow(PSF) 
plt.xlabel("PSF")
plt.subplot(132)
plt.imshow(PSF_steep) 
plt.xlabel("PSF_steep")
plt.subplot(133)
plt.imshow(PSF_long)
plt.xlabel("PSF_long")

plt.figure(6)
plt.subplot(131)
result = inverse(blurred, PSF, 0.001)   #逆滤波
plt.imshow(result)
plt.xlabel("PSF")
plt.subplot(132)
result = inverse(blurred, PSF_steep, 0.001)   #逆滤波
plt.imshow(result)
plt.title("different PSF")
plt.xlabel("PSF_steep")
plt.subplot(133)
result = inverse(blurred, PSF_long, 0.001)   #逆滤波
plt.imshow(result)
plt.xlabel("PSF_long")

mean = 0 
sigma = 10#根据均值和标准差生成符合高斯分布的噪声
gauss = np.random.normal(mean,sigma,np.array(image/255,dtype=float).shape) 
gauss_blurred=blurred+gauss
possion_blurred=blurred+np.random.poisson(lam=20,size=image.shape).astype(dtype='uint8') #lam的值越大，添加的噪声越多 
plt.figure(7)
plt.subplot(131)
plt.imshow(blurred) 
plt.xlabel("only motion")
plt.subplot(132)
plt.imshow(gauss_blurred) 
plt.xlabel("motion+gauss")
plt.subplot(133)
plt.imshow(possion_blurred) 
plt.xlabel("motion+possion") 

plt.figure(8)
plt.subplot(131)
result = inverse(blurred, PSF, 0.1)   #逆滤波
plt.imshow(result)
plt.xlabel("only motion")
plt.subplot(132)
result = inverse(gauss_blurred, PSF, 0.1)   #逆滤波
plt.imshow(result)
plt.xlabel("motion+gauss")
plt.subplot(133)
result = inverse(possion_blurred, PSF, 0.1)   #逆滤波
plt.imshow(result)
plt.xlabel("motion+possion")

plt.figure(9)
plt.subplot(131)
result = wiener(blurred, PSF, 0.01)   #维纳滤波
plt.imshow(result)
plt.xlabel("only motion")
plt.subplot(132)
result = wiener(gauss_blurred, PSF, 0.01)   #维纳滤波
plt.imshow(result)
plt.xlabel("motion+gauss")
plt.subplot(133)
result = wiener(possion_blurred, PSF, 0.01)   #维纳滤波
plt.imshow(result)
plt.xlabel("motion+possion")
plt.show()