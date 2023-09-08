import math
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import cv2

def hough_transform(img):
    """
    对于给定的二值化图像，进行哈夫变换
    返回(r, theta)空间中的投票矩阵
    """
    # 图像的高和宽
    height, width = img.shape

    # 对角线的长度
    diag_len = int(round(math.sqrt(width * width + height * height)))
    
    # 极角和极径的步长
    thetas = np.deg2rad(np.arange(0.0, 180.0))
    rs = np.arange(-diag_len, diag_len + 1)

    # 极径和极角数量
    num_thetas = len(thetas)
    num_rs = len(rs)

    # 极坐标空间的投票矩阵
    accumulator = np.zeros((num_rs, num_thetas), dtype=np.uint64)
    
    # 非零像素的坐标
    y_idxs, x_idxs = np.nonzero(img)

    # 迭代每个非零像素
    for i in range(len(x_idxs)):
        x = x_idxs[i]
        y = y_idxs[i]

        # 在(r, theta)空间中进行投票
        for j in range(num_thetas):
            r = int(round(x * np.cos(thetas[j]) + y * np.sin(thetas[j]))) + diag_len
            accumulator[r, j] += 1

    return accumulator, thetas, rs


def detect_lines(image_path, output_path, threshold):
    """
    给定一个图像路径，进行直线检测
    超过给定阈值的直线会被绘制在原始图像中，并保存为一张新的图像
    """

    # 读入图像，转为灰度图
    image = Image.open(image_path).convert('L')
    img_data = np.array(image)

    # 获取边缘
    edges = cv2.Canny(img_data, 50, 150, apertureSize=3)
    
    # 进行哈夫变换
    accumulator, thetas, rhos = hough_transform(edges)

    # 找出投票高于阈值的(r, theta)值
    indices = np.argwhere(accumulator > threshold)
    indices = indices[np.argsort(-accumulator[indices[:,0], indices[:,1]])]
    
    # 投票高于阈值的直线
    lines = []
    for i in range(len(indices)):
        r = rhos[indices[i][0]]
        theta = thetas[indices[i][1]]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * r
        y0 = b * r
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        lines.append(((x1, y1), (x2, y2)))

    # 绘制直线，并保存结果
    img = Image.open(image_path).convert('RGB')
    draw = ImageDraw.Draw(img)
    for line in lines:
        draw.line(line, fill='red', width=2)
    img.save(output_path)


def main():
    # 设置输入文件、阈值和输出文件名
    image_path = '2.jpg'
    threshold = 280
    output_path = 'result.jpg'

    # 进行直线检测
    detect_lines(image_path, output_path, threshold)

    # 显示结果
    img = Image.open(output_path)
    plt.imshow(img)
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    main()