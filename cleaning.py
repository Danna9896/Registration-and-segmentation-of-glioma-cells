import Segmentation_funcs as sfuncs
import numpy as np
import os
import tifffile as tiff
import cv2

tiff_paths = ["P41_OR1.tif", "P41_OR2.tif", "P41_OR3.tif",
              "P42_OR1.tif", "P42_OR2.tif", "P42_OR3.tif",
              "R14_OR1.tif", "R14_OR2.tif", "R14_OR3.tif"]

red_channels = ["P41/C1_P41_OR1.tif", "P41/C1_P41_OR2.tif", "P41/C1_P41_OR3.tif",
                "P42/C1_P42_OR1.tif", "P42/C1_P42_OR2.tif", "P42/C1_P42_OR3.tif",
                "R14/C1_R14_OR1.tif", "R14/C1_R14_OR2.tif", "R14/C1_R14_OR3.tif"]

def cleaner(imgs):
    cleaned_imgs = []
    for i in range(len(imgs)):
        img = imgs[i]

        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        img[mask == 0] = 0

        contours, _ = cv2.findContours(img.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if ((cv2.contourArea(contour) < 15)):
                cv2.drawContours(img, [contour], -1, 0, thickness=cv2.FILLED)

        cleaned_imgs.append(img)
        print(i)
    return cleaned_imgs


for i in range(0,10):
    # reading tiff images
    tiff_path1 = os.path.join("../images/denoised_images", tiff_paths[i])
    img = tiff.imread(tiff_path1)
    cleaned = cleaner(img)
    tiff_path2 = os.path.join("../images/cleaned", tiff_paths[i])
    os.makedirs(os.path.dirname(tiff_path2), exist_ok=True)
    cleaned_array = np.stack(cleaned, axis=0).astype(np.uint16)
    tiff.imwrite(tiff_path2, cleaned_array, imagej=True)