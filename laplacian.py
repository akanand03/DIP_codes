import cv2
import numpy as np


input_image = cv2.imread('lena.jpg', cv2.IMREAD_GRAYSCALE)


laplacian_kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])


input_image = cv2.copyMakeBorder(input_image, 1, 1, 1, 1, cv2.BORDER_REPLICATE)


height, width = input_image.shape


result_image = np.zeros((height, width), dtype=np.float32)

lap = np.zeros_like(input_image, dtype=np.float32)

c = -1

for y in range(1, height - 1):
    for x in range(1, width - 1):
        laplacian_sum = 0
        for ky in range(-1, 2):
            for kx in range(-1, 2):
                laplacian_sum += input_image[y + ky, x + kx] * laplacian_kernel[ky + 1, kx + 1]
        lap[y, x] = laplacian_sum
print(min(lap.flatten()))
result_image = input_image + (c * lap)


min_value = np.min(result_image)
fm = result_image - min_value


K = 255  
fs = np.uint8(K * (fm / np.max(fm)))


laplacian_result = np.uint8(result_image)
scaled_up_image = np.uint8(fm)


laplacian_opencv = cv2.Laplacian(input_image, cv2.CV_32F, ksize=3)


cv2.imshow('Original Image', input_image)
cv2.imshow('Laplacian Result', np.uint8(lap))
cv2.imshow('Laplacian OpenCV', laplacian_opencv)
cv2.imshow('Scaled-Up Image (fm)', scaled_up_image)
cv2.imshow('Filtered Image (fs)', fs)

cv2.waitKey(0)
cv2.destroyAllWindows()
