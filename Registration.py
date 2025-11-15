# imports
import tifffile as tiff
import numpy as np
import time
import os

from skimage import exposure
from skimage.metrics import structural_similarity as ssim
from skimage.feature import match_template
import Registration_funcs
import average_deviation

avg_dev_list = []

blue_channels = ["../images/P41/C2_P41_OR1.tif",
                 "../images/P41/C2_P41_OR2.tif",
                 "../images/P41/C2_P41_OR3.tif",
                 "../images/P42/C2_P42_OR1.tif",
                 "../images/P42/C2_P42_OR2.tif",
                 "../images/P42/C2_P42_OR3.tif",
                 "../images/R14/C2_R14_OR1.tif",
                 "../images/R14/C2_R14_OR2.tif",
                 "../images/R14/C2_R14_OR3.tif"]

red_channels = ["../images/P41/C1_P41_OR1.tif",
                 "../images/P41/C1_P41_OR2.tif",
                 "../images/P41/C1_P41_OR3.tif",
                 "../images/P42/C1_P42_OR1.tif",
                 "../images/P42/C1_P42_OR2.tif",
                 "../images/P42/C1_P42_OR3.tif",
                 "../images/R14/C1_R14_OR1.tif",
                 "../images/R14/C1_R14_OR2.tif",
                 "../images/R14/C1_R14_OR3.tif"]

tiff_paths = ["P41_OR1.tif", "P41_OR2.tif", "P41_OR3.tif",
              "P42_OR1.tif", "P42_OR2.tif", "P42_OR3.tif",
              "R14_OR1.tif", "R14_OR2.tif", "R14_OR3.tif"]

for i in range(7,9):
    # reading tiff images
    blue_channel = tiff.imread(blue_channels[i])
    red_channel = tiff.imread(red_channels[i])
    tiff_path = os.path.join("../images/registered_images_3D_final", tiff_paths[i])
    os.makedirs(os.path.dirname(tiff_path), exist_ok=True)

    # Initiating params
    if (i == 0):
        avg_dev = 5
    else:
        avg_dev = average_deviation.find_avg_dev_reg(blue_channel, red_channel) #from average_deviation algorithm


    avg_dev_list.append(avg_dev)
    num_iter = 5
    start_idx, end_idx = 0, blue_channel.shape[0]
    a = 1 # a decides which registration algorithm to use
    redo = 0 # param to fix a faulty registration
    time_taken = 0 #param to test how long registration takes
    counter = 0 #param that counts faulty registrations for the same frame
    ssim_value, diff = 0, 0 #tests
    yoff, prev_yoff = 0, 0 #following frames require similar shift
    xoff, prev_xoff = 0, 0
    alpha, beta = 0.5, 0.5
    ssim_before_reg, correlation_before_reg = 0.8, 0.8
    registered_images_list = []
    start_run_time = time.time()

    # registration process for multiple images
    k = start_idx
    while k<end_idx:
        print(f"idx number: {k}")
        #out of bounds
        while k + avg_dev < start_idx or (k + avg_dev > end_idx - 1 and k < end_idx):
            blue = blue_channel[k]
            registered_images_list.append(blue)
            k += 1

        if k + avg_dev > end_idx - 1:
            break

        blue = blue_channel[k]
        red = red_channel[k + avg_dev]

        #contrasting images
        red_contrast = exposure.equalize_adapthist(red, clip_limit=0.04)
        blue_contrast = exposure.equalize_adapthist(blue, clip_limit=0.04)

        prev_ssim = ssim_before_reg
        ssim_before_reg, _ = ssim(red, blue, full=True, data_range=65535)
        diff_image_ssim_quality = ssim_before_reg - prev_ssim
        prev_corr = correlation_before_reg
        correlation_before_reg = match_template(red, blue)
        diff_image_corr_quality = correlation_before_reg  - prev_corr

        # registration
        a, num_iter = Registration_funcs.decide_algorithm(a, time_taken, ssim_value, diff, num_iter)
        start_time = time.time()
        if a == 1:
            registered_image, yoff, xoff = Registration_funcs.registration1(red_contrast, blue_contrast, blue, prev_yoff, prev_xoff, num_iter)
        elif a == 2:
            registered_image, yoff, xoff = Registration_funcs.registration2(red_contrast, blue_contrast, blue, prev_yoff, prev_xoff)
        elif a == 3:
            registered_image, yoff, xoff = Registration_funcs.registration3(red_contrast, blue_contrast, blue, prev_yoff, prev_xoff)
        end_time = time.time()
        time_taken = end_time - start_time

        # test quality of registration
        ssim_after_reg, _ = ssim(red, registered_image, full=True, data_range=65535)
        correlation_after_reg = match_template(red, registered_image)
        diff_image_reg_ssim = ssim_after_reg - ssim_before_reg
        ssim_value = beta * diff_image_reg_ssim + (1 - beta) * diff_image_ssim_quality
        diff_image_reg_corr = correlation_after_reg.max() - correlation_before_reg.max()
        diff = alpha * diff_image_reg_corr + (1 - alpha) * diff_image_corr_quality
        if np.abs(diff_image_reg_corr) > np.abs(diff_image_corr_quality) and alpha < 1:
            alpha += 0.1
        elif np.abs(diff_image_reg_corr) < np.abs(diff_image_corr_quality) and alpha > 0:
            alpha -= 0.1
        if np.abs(diff_image_reg_ssim) > np.abs(diff_image_ssim_quality) and beta < 1:
            beta += 0.1
        elif np.abs(diff_image_reg_ssim) < np.abs(diff_image_ssim_quality) and beta > 0:
            beta -= 0.1
        if ssim_before_reg > 0.9 and ssim_value > 0 :
            ssim_value += 0.01
        if correlation_before_reg > 0.8 and diff > 0:
            diff += 0.01
        redo = (ssim_value < 0 or diff < 0)
        print(ssim_after_reg, correlation_before_reg)

        # prints
        print(f"redo: {redo}")
        print(f"SSIM diff: {ssim_value}")
        print(f"correlation diff: {diff}")
        print(f"Time taken for registration: {time_taken}")
        print(f"Algorithm num:{a}")

        # saving tiff file of registered images
        if redo :
            counter += 1
        if counter == 3:
            prev_yoff = 0
            prev_xoff = 0
        if counter > 5:
            counter = 0
            registered_image = blue
            print(f"Registration has failed")
            redo = False
        if not redo:
            if diff > 0.02:
                prev_yoff = yoff
                prev_xoff = xoff
            counter = 0
            registered_images_list.append(registered_image)
            k+=1

# saving the tiff images
    registered_images_array = np.stack(registered_images_list, axis=0).astype(np.uint16)
    tiff.imwrite(tiff_path, registered_images_array, imagej=True)
    print(f"runtime:{time.time() - start_run_time}")
    print(f"finished {tiff_paths[i]} successfully!")

# saving average deviation per organoid
with open('avg_dev_list.txt', 'w') as file:
    for avg_dev in avg_dev_list:
        file.write(f"{avg_dev}\n")