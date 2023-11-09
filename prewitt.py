#prewitt 
import numpy as np
import matplotlib.pyplot as plt

# Load an example image
img = np.array([[0, 0, 0, 0, 0],
                [0, 255, 255, 255, 0],
                [0, 255, 255, 255, 0],
                [0, 255, 255, 255, 0],
                [0, 0, 0, 0, 0]], dtype=np.uint8)

# Define the Gaussian filter
sigma = 1
size = int(6 * sigma)
size = size + 1 if size % 2 == 0 else size
pad_size = size // 2
x, y = np.meshgrid(np.linspace(-pad_size, pad_size, size), np.linspace(-pad_size, pad_size, size))
kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))
kernel = kernel / np.sum(kernel)

# Pad the image
pad_img = np.pad(img, [(1, 1), (1, 1)], mode='constant')

# Apply Gaussian filter
img_gaussian = np.zeros_like(img, dtype=np.float)
for i in range(1, img.shape[0] + 1):
    for j in range(1, img.shape[1] + 1):
        img_gaussian[i - 1, j - 1] = np.sum(pad_img[i - pad_size:i + pad_size + 1, j - pad_size:j + pad_size + 1] * kernel)

# Define Prewitt operators
maskx = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
masky = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])

# Apply Prewitt operators
gx = np.zeros_like(img, dtype=np.float)
gy = np.zeros_like(img, dtype=np.float)
for i in range(1, img.shape[0] + 1):
    for j in range(1, img.shape[1] + 1):
        gx[i - 1, j - 1] = np.sum(pad_img[i - 1:i + 2, j - 1:j + 2] * maskx)
        gy[i - 1, j - 1] = np.sum(pad_img[i - 1:i + 2, j - 1:j + 2] * masky)

# Compute the gradient magnitude
gradient_magnitude = np.sqrt(gx**2 + gy**2)

# Display the results
plt.gray()
plt.figure(figsize=(12, 4))

plt.subplot(131)
plt.imshow(img, cmap='gray')
plt.title("Original Image")

plt.subplot(132)
plt.imshow(img_gaussian, cmap='gray')
plt.title("Gaussian Filtered Image")

plt.subplot(133)
plt.imshow(gradient_magnitude, cmap='gray')
plt.title("Prewitt Filtered Image")

plt.show()
