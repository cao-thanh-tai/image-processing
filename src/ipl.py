
import numpy as np
import cv2

class ImageProcessor:
    def __init__(self):
        pass
    
    def image_negative(self, image):
        return 255 - image
    
    def compute_histogram(self, image):
        return cv2.calcHist([image], [0], None, [256], [0, 256])
    
    def contrast_stretching(self, image, r1, s1, r2, s2):
        def pixel_val(pix):
            if 0 <= pix <= r1:
                return (s1 / r1) * pix
            elif r1 < pix <= r2:
                return ((s2 - s1) / (r2 - r1)) * (pix - r1) + s1
            else:
                return ((255 - s2) / (255 - r2)) * (pix - r2) + s2
        
        pixel_val = np.vectorize(pixel_val)
        return pixel_val(image).astype(np.uint8)
    
    def contrast_stretching_lookup(self, image, r1, s1, r2, s2):
        lookup_table = np.zeros(256, dtype=np.uint8)
        for i in range(256):
            if 0 <= i <= r1:
                lookup_table[i] = (s1 / r1) * i
            elif r1 < i <= r2:
                lookup_table[i] = ((s2 - s1) / (r2 - r1)) * (i - r1) + s1
            else:
                lookup_table[i] = ((255 - s2) / (255 - r2)) * (i - r2) + s2
        
        return cv2.LUT(image, lookup_table)
    
    