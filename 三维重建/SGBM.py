import cv2
import numpy as np
from pylab import *
#SGBM法
window_size = 5 # 匹配的块大小 > = 1的奇数
min_disp = 32 # 最小可能的差异值
num_disp = 192-min_disp # 最大差异减去最小差异
blockSize = window_size # 匹配的块大小
uniquenessRatio = 1 # 最佳（最小）计算成本函数值
speckleRange = 3 # 每个连接组件内的最大视差变化
speckleWindowSize = 3 # 平滑视差区域的最大尺寸
disp12MaxDiff = 500 # 左右视差检查中允许的最大差异
P1 = 600 # 控制视差平滑度的第一个参数
P2 = 2400 # 第二个参数控制视差平滑度
imgL = cv2.imread('tsukuba_l.png') # 左目图像
imgR = cv2.imread('tsukuba_r.png') # 右目图像
# 创建StereoSGBM对象并计算
stereo = cv2.StereoSGBM_create(minDisparity = min_disp,numDisparities = num_disp,blockSize = window_size,uniquenessRatio = uniquenessRatio,speckleRange = speckleRange,speckleWindowSize = speckleWindowSize,disp12MaxDiff = disp12MaxDiff,P1 = P1,P2 = P2)
disp = stereo.compute(imgL, imgR).astype(np.float32)/16  # 计算视差图
plt.imsave('Depth_SGBM.jpg', (disp-min_disp)/num_disp)# 显示视差图结果
