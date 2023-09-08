
import cv2
import numpy as np
# 读入灰度图像，并进行二值化处理
img = cv2.imread('1.jpg', 0)
ret,thresh = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
cv2.imshow("thresh",thresh)
# 初始化连接方向列表
connectivity = 8
cclist = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]

# 初始化标记图像（每个像素初始为0），并定义当前标签和堆栈
h, w = thresh.shape
label = 1
stack = []

# 开始扫描图像，遇到非0像素就进行区域种子填充
for i in range(h):
    for j in range(w):
        if thresh[i,j] != 0 and label == 1:
            # 新的连通区域，打上标签
            label +=1
            thresh[i,j] = label
            stack.append((i,j))

            # 循环扫描堆栈中的点，直到堆栈为空
            while len(stack) > 0:
                x, y = stack.pop() 

                # 检查周围8个像素是否需要扩展该连通区域的标签
                for dx,dy in cclist:
                    nx, ny = x+dx, y+dy
                    if nx>=0 and ny>=0 and nx<h and ny<w:
                        if thresh[nx,ny]!=0 and thresh[nx,ny] != label:
                            # 在标记图像中打上当前标签
                            thresh[nx,ny] = label
                            stack.append((nx,ny))

# 全部连通区域扫描并标记完毕，将每个连通区域随机着色
output = cv2.cvtColor(thresh.astype('uint8'),cv2.COLOR_GRAY2BGR)
for l in range(2,label+1):
    color = (255,0,0)
    output[thresh==l] = color
    
# 显示标记后的图像
cv2.imshow('result', output)
cv2.waitKey(0)