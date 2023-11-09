#Robert filter
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color
from scipy.ndimage import convolve

# Load the Lena image from a file
image = io.imread('lena.jpg')

# Convert the image to grayscale if it's not already
if len(image.shape) == 3 and image.shape[2] == 3:
    image = color.rgb2gray(image)

# Define the Roberts operators
maskx = np.array([[1, 0], [0, -1]])
masky = np.array([[0, 1], [-1, 0]])

# Apply Roberts operators
gx = convolve(image, maskx)
gy = convolve(image, masky)

# Compute the gradient magnitude
gradient_magnitude = np.sqrt(gx**2 + gy**2)

# Display the results
plt.gray()
plt.figure(figsize=(12, 4))

plt.subplot(121)
plt.imshow(image, cmap='gray')
plt.title("Original Lena Image")

plt.subplot(122)
plt.imshow(gradient_magnitude, cmap='gray')
plt.title("Roberts Filtered Lena Image")

plt.show()
