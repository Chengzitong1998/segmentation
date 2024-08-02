import cv2
import matplotlib.pyplot as plt

# 读取图像
image = cv2.imread('/root/autodl-tmp/segmentation/1.png', cv2.IMREAD_GRAYSCALE)

# 应用阈值分割，将灰度图像中大于等于 128 的像素设定为 255，小于 128 的像素设定为 0
_, thresholded = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)

# 显示原始图像和分割后的图像
plt.subplot(121), plt.imshow(image, cmap='gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(thresholded, cmap='Greens')
plt.title('Thresholded Image'), plt.xticks([]), plt.yticks([])

plt.savefig('my_image.png', bbox_inches='tight', pad_inches=0.0, dpi=300)

plt.show()
