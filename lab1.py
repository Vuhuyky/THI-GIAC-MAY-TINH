import cv2 

# 1. Đọc ảnh
img = cv2.imread('test.jpg')

if img is None:
    print("Không tìm thấy ảnh!")
    exit()

print("Shape:", img.shape)
print("Size:", img.size)
print("Datatype:", img.dtype)

cv2.imshow('Original Image', img)

px = img[100, 100]
print("Pixel at (100,100):", px)

img[100, 100] = [255, 0, 0]

roi = img[100:300, 200:400]
img[0:200, 0:200] = roi

b, g, r = cv2.split(img)

cv2.imshow("Blue Channel", b)
cv2.imshow("Green Channel", g)
cv2.imshow("Red Channel", r)

img2 = cv2.merge((b, g, r))
cv2.imshow("Merged Image", img2)

cv2.waitKey(0)
cv2.destroyAllWindows()