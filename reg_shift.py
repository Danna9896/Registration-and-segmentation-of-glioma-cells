# imports
import tifffile as tiff
from skimage import exposure
import time
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

names = ["P41 OR1", "P41 OR2", "P41 OR3", "P42 OR1", "P42 OR2", "P42 OR3", "R14 OR1",  "R14 OR2", "R14 OR3"]

for i in range(0,9):
    x_off_list = []
    y_off_list = []
    # reading tiff images
    blue_channel = tiff.imread(blue_channels[i])
    red_channel = tiff.imread(red_channels[i])

    # Initiating params
    avg_dev = average_deviation.find_avg_dev_reg(blue_channel, red_channel) #from average_deviation algorithm
    avg_dev_list.append(avg_dev)
    num_iter = 5
    start_idx, end_idx = blue_channel.shape[0]-200, blue_channel.shape[0]-100
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

        # registration
        a, num_iter = Registration_funcs.decide_algorithm(a, time_taken, ssim_value, diff, num_iter)
        if a == 1:
            registered_image_k, yoff, xoff = Registration_funcs.registration1(red_contrast, blue_contrast, blue, prev_yoff, prev_xoff, num_iter)
        elif a == 2:
            registered_image_k, yoff, xoff = Registration_funcs.registration2(red_contrast, blue_contrast, blue, prev_yoff, prev_xoff)
        elif a == 3:
            registered_image_k, yoff, xoff = Registration_funcs.registration3(red_contrast, blue_contrast, blue, prev_yoff, prev_xoff)

        # test quality of registration
        ssim_before, _ = ssim(red, blue, full=True, data_range=65535)
        ssim_after, _ = ssim(red, registered_image_k, full=True, data_range=65535)
        correlation_before = match_template(red, blue)
        correlation_after = match_template(red, registered_image_k)
        ssim_value = ssim_after - ssim_before
        diff = correlation_after.max() - correlation_before.max()
        redo = (ssim_value < 0 or diff < 0)

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
            x_off_list.append(prev_xoff)
            y_off_list.append(prev_yoff)

    #saving registration shifts
    with open(f'x_off {names[i]}.txt', 'w') as file:
        for x_off in x_off_list:
            file.write(f"{x_off}\n")

    with open(f'y_off {names[i]}.txt', 'w') as file:
        for y_off in y_off_list:
            file.write(f"{y_off}\n")