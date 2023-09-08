import cv2

# 读入原始图像，转化为灰度图像，并进行二值化处理
img = cv2.imread('1.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)

# 定义用于轮廓跟踪的初始位置和方向
start_pixel = None
current_pixel = None
direction = None

# 先找到图片中的第一个非0像素，作为轮廓跟踪时的起点
for x in range(thresh.shape[0]):
    for y in range(thresh.shape[1]):
        if thresh[x,y]!=0:
            start_pixel = (x,y)
            break
    if start_pixel:
        break

# 如果没有找到起点，则返回空轮廓
if not start_pixel:
    contour = []

else:
    # 定义一个空列表，用于存放轮廓上的像素坐标
    contour = [start_pixel]

    # 开始进行轮廓跟踪
    current_pixel = start_pixel
    direction = 'east'
    while True:
        if direction == 'east':
            next_pixel = (current_pixel[0], current_pixel[1]+1)
            if thresh[next_pixel]==255:
                direction = 'south'
                contour.append(next_pixel)
                current_pixel = next_pixel
            else:
                direction = 'north'

        elif direction == 'south':
            next_pixel = (current_pixel[0]+1, current_pixel[1])
            if thresh[next_pixel]==255:
                direction = 'west'
                contour.append(next_pixel)
                current_pixel = next_pixel
            else:
                direction = 'east'

        elif direction == 'west':
            next_pixel = (current_pixel[0], current_pixel[1]-1)
            if thresh[next_pixel]==255:
                direction = 'north'
                contour.append(next_pixel)
                current_pixel = next_pixel
            else:
                direction = 'south'

        elif direction == 'north':
            next_pixel = (current_pixel[0]-1, current_pixel[1])
            if thresh[next_pixel]==255:
                direction = 'east'
                contour.append(next_pixel)
                current_pixel = next_pixel
            else:
                direction = 'west'

        # 如果回到了起点，则跳出循环 
        if current_pixel == start_pixel:
            break

# 将轮廓上的像素画在图像上并显示出来
for pixel in contour:
    img[pixel] = (0,0,255)
cv2.imshow('contour', img)
cv2.waitKey(0)