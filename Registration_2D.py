# imports
import tifffile as tiff
from skimage import exposure
import numpy as np
import time
import os
from skimage.metrics import structural_similarity as ssim
from skimage.feature import match_template
import useful_reg_funcs

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

for i in range(0,9):
    # reading tiff images
    blue_channel = tiff.imread(blue_channels[i])
    red_channel = tiff.imread(red_channels[i])
    tiff_path = os.path.join("../images/Registered_images_two_dim", tiff_paths[i])
    os.makedirs(os.path.dirname(tiff_path), exist_ok=True)

    # Initiating params
    num_iter = 5
    start_idx, end_idx = 0, blue_channel.shape[0]
    a = 1 # a decides which registration algorithm to use
    redo = 0 # param to fix a faulty registration
    time_taken = 0 #param to test how long registration takes
    counter = 0 #param that counts faulty registrations for the same frame
    ssim_value, diff = 0, 0 #tests
    yoff, prev_yoff = 0, 0 #following frames require similar shift
    xoff, prev_xoff = 0, 0
    registered_images_list = []
    start_run_time = time.time()

    # registration process for multiple images
    k = start_idx
    while k<end_idx:
        print(f"idx number: {k}")
        blue = blue_channel[k]
        red = red_channel[k]

        #contrasting images
        red_contrast = exposure.equalize_adapthist(red, clip_limit=0.04)
        blue_contrast = exposure.equalize_adapthist(blue, clip_limit=0.04)

        # registration
        a, num_iter = useful_reg_funcs.decide_algorithm(a, time_taken, ssim_value, diff, num_iter)
        start_time = time.time()
        if a == 1:
            registered_image_k, yoff, xoff = useful_reg_funcs.registration1(red_contrast, blue_contrast, blue, prev_yoff, prev_xoff, num_iter)
        elif a == 2:
            registered_image_k, yoff, xoff = useful_reg_funcs.registration2(red_contrast, blue_contrast, blue, prev_yoff, prev_xoff)
        elif a == 3:
            registered_image_k, yoff, xoff = useful_reg_funcs.registration3(red_contrast, blue_contrast, blue, prev_yoff, prev_xoff)
        end_time = time.time()
        time_taken = end_time - start_time

        # test quality of registration
        ssim_before, _ = ssim(red, blue, full=True, data_range=65535)
        ssim_after, _ = ssim(red, registered_image_k, full=True, data_range=65535)
        correlation_before = match_template(red, blue)
        correlation_after = match_template(red, registered_image_k)
        ssim_value = ssim_after - ssim_before
        diff = correlation_after.max() - correlation_before.max()
        redo = (ssim_value < 0 or diff < 0)

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
            registered_image_k = blue
            print(f"Registration has failed")
            redo = False
        if not redo:
            if diff > 0.02:
                prev_yoff = yoff
                prev_xoff = xoff
            counter = 0
            registered_images_list.append(registered_image_k)
            k+=1

# saving the tiff images
    registered_images_array = np.stack(registered_images_list, axis=0).astype(np.uint16)
    tiff.imwrite(tiff_path, registered_images_array, imagej=True)
    print(f"runtime:{time.time() - start_run_time}")
    print(f"finished {tiff_paths[i]} successfully!")

