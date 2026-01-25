import cv2
import matplotlib.pyplot as plt
import numpy as np
from ipl import ImageProcessor as ip

image = cv2.imread('../data/raw/Apple/Fresh/apple_fresh_001.jpg (1).jpg')

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
wight = image.shape[1]
height = image.shape[0]
aspect_ratio = wight / height
plt.figure(figsize=(5*aspect_ratio, 5))
plt.title("Grayscale Image")
plt.imshow(image_gray, cmap='gray')
plt.show()

