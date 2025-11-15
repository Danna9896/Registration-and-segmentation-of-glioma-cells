import numpy as np
import cv2

def denoise_subtraction(Reference_image, Registered_image):
    # This function computes the difference between reg image and ref and clamp negative values to zero
    image1 = cv2.normalize(Reference_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    image2 = cv2.normalize(Registered_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, binary1 = cv2.threshold(image1, 50, 255, cv2.THRESH_BINARY)  # Adjust threshold as needed
    _, binary2 = cv2.threshold(image2, 50, 255, cv2.THRESH_BINARY)
    unique_mask = cv2.subtract(binary1, binary2)
    unique_mask[unique_mask < 0] = 0
    result = cv2.bitwise_and(image1, image1, mask=unique_mask)

    return result

def denoise_normalize(Reference_image, Registered_image):
    _, binary1 = cv2.threshold(Reference_image, 255, 255, cv2.THRESH_BINARY)  # Adjust threshold as needed
    _, binary2 = cv2.threshold(Registered_image, 255, 255, cv2.THRESH_BINARY)
    binary1[binary1 == 0] = 1  # to avoid dividing by zero
    binary2[binary2 == 0] = 1
    unique_mask = np.array([])
    unique_mask = cv2.divide(binary1, binary2, unique_mask)
    unique_mask[unique_mask < 1.2] = 0
    unique_mask = unique_mask.astype(np.uint8)
    result = cv2.bitwise_and(Reference_image, Reference_image, mask=unique_mask)

    return result

def denoise_normalize2(Reference_image, Registered_image):
    Registered_image[Registered_image == 0] = 1e-8
    unique_mask = cv2.divide(Reference_image, Registered_image)
    result = Reference_image.copy()
    mask = unique_mask <= 1
    result[mask] = 0

    return result




