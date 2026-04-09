import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread('messi5.jpg')
assert img is not None, "Không thể đọc ảnh"

px = img[100, 100]
print("Pixel tại (100,100):", px)

blue = img[100, 100, 0]
print("Kênh blue:", blue)
img[100, 100] = [255, 255, 255]

print("Shape:", img.shape)
print("Size:", img.size)
print("Dtype:", img.dtype)

ball = img[280:340, 330:390]
img[273:333, 100:160] = ball

b, g, r = cv.split(img)
img_merge = cv.merge((b, g, r))

img[:, :, 2] = 0

BLUE = [255, 0, 0]

replicate = cv.copyMakeBorder(img, 10, 10, 10, 10, cv.BORDER_REPLICATE)
reflect = cv.copyMakeBorder(img, 10, 10, 10, 10, cv.BORDER_REFLECT)
reflect101 = cv.copyMakeBorder(img, 10, 10, 10, 10, cv.BORDER_REFLECT_101)
wrap = cv.copyMakeBorder(img, 10, 10, 10, 10, cv.BORDER_WRAP)
constant = cv.copyMakeBorder(img, 10, 10, 10, 10, cv.BORDER_CONSTANT, value=BLUE)

plt.subplot(231), plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB)), plt.title('Original')
plt.subplot(232), plt.imshow(cv.cvtColor(replicate, cv.COLOR_BGR2RGB)), plt.title('Replicate')
plt.subplot(233), plt.imshow(cv.cvtColor(reflect, cv.COLOR_BGR2RGB)), plt.title('Reflect')
plt.subplot(234), plt.imshow(cv.cvtColor(reflect101, cv.COLOR_BGR2RGB)), plt.title('Reflect101')
plt.subplot(235), plt.imshow(cv.cvtColor(wrap, cv.COLOR_BGR2RGB)), plt.title('Wrap')
plt.subplot(236), plt.imshow(cv.cvtColor(constant, cv.COLOR_BGR2RGB)), plt.title('Constant')

plt.show()