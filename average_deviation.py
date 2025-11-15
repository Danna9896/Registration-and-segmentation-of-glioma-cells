from skimage import exposure
import time
from skimage.metrics import structural_similarity as ssim
from skimage.feature import match_template
import numpy as np
import Registration_funcs as reg

def find_avg_dev_reg(blue_channel, red_channel):

    #initiating params
    num_images = blue_channel.shape[0]
    print(f"Number of images: {num_images}")
    start = 200
    end = num_images -  200
    step_size = round((end-start)/6)
    mid1 = start + step_size
    mid2 = mid1 + step_size
    mid3 = mid2 + step_size
    mid4 = mid3 + step_size
    mid5 = mid4 + step_size
    steps = [start, mid1, mid2, mid3, mid4, mid5, end] #testing three different sections for deviation
    best_reg_idx = np.zeros(7)
    start_run_time = time.time()


    for step in steps:
        k = - 10
        a = 1
        counter = 0
        best_ssim = 0
        best_diff = 0
        time_taken = 0
        diff = 0
        ssim_value = 0
        num_iter = 5
        alpha, beta = 0.5, 0.5
        ssim_before_reg, correlation_before_reg = 0.8, 0.8

        while k <= 0 :
            blue = blue_channel[step]
            red = red_channel[step - k]
            print(f"blue idx:{step}")
            print(f"red idx:{step-k}")
            blue_contrast = exposure.equalize_adapthist(blue, clip_limit=0.04)
            red_contrast = exposure.equalize_adapthist(red, clip_limit=0.04)

            prev_ssim = ssim_before_reg
            ssim_before_reg, _ = ssim(red, blue, full=True, data_range=65535)
            diff_image_ssim_quality = ssim_before_reg - prev_ssim
            prev_corr = correlation_before_reg
            correlation_before_reg = match_template(red, blue)
            diff_image_corr_quality = correlation_before_reg - prev_corr

            a, num_iter = reg.decide_algorithm(a, time_taken, ssim_value, diff, num_iter)

            # registration
            start_time = time.time()
            if a == 1:
                registered_image, yoff, xoff = reg.registration1(red_contrast, blue_contrast, blue, 0, 0, num_iter)
            elif a == 2:
                registered_image, yoff, xoff = reg.registration2(red_contrast, blue_contrast, blue, 0, 0)
            elif a == 3:
                registered_image, yoff, xoff = reg.registration3(red_contrast, blue_contrast, blue, 0, 0)
            end_time = time.time()
            time_taken = end_time - start_time

            # scaling visuals
            if registered_image.max() < 1 or registered_image.min() > 10:
                registered_image = exposure.rescale_intensity(registered_image, in_range=(
                registered_image.min(), registered_image.max()), out_range=(blue.min(), blue.max()))

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
            if ssim_before_reg > 0.9 and ssim_value > 0:
                ssim_value += 0.01
            if correlation_after_reg > 0.8 and diff > 0:
                diff += 0.01
            redo = (ssim_value < 0 or diff < 0)

            # prints
            print(f"SSIM diff: {ssim_value}")
            print(f"correlation diff: {diff}")
            print(f"Time taken for registration: {time_taken}")
            print(f"Algorithm num:{a}")
            print(f"best_reg_idx:{best_reg_idx}")
            print(f"best_ssim:{best_ssim}")
            print(f"best_diff:{best_diff}")
            print(f"redo:{redo}")

            # saving best test values in order to find frame deviation
            if best_ssim < ssim_value and best_diff < diff:
                best_ssim = ssim_value
                best_diff = diff
                index = steps.index(step)
                best_reg_idx[index] = step - k

            if not redo or counter == 5:
                print(f"k:{k}")
                k += 1
                counter = 0

            if redo:
                counter += 1

    # calculating average deviation
    average_deviation = sum(best_reg_idx[i] - steps[i] for i in range(0,7)) / 7
    average_deviation = np.round(average_deviation)
    print(f"average deviation:{average_deviation}")
    print(f"runtime to find average deviation:{time.time() - start_run_time}")
    return int(average_deviation)

def find_avg_dev(blue_channel, red_channel):

    #initiating params
    num_images = blue_channel.shape[0]
    print(f"Number of images: {num_images}")
    start = 200
    end = num_images -  200
    step_size = round((end-start)/6)
    mid1 = start + step_size
    mid2 = mid1 + step_size
    mid3 = mid2 + step_size
    mid4 = mid3 + step_size
    mid5 = mid4 + step_size
    steps = [start, mid1, mid2, mid3, mid4, mid5, end] #testing three different sections for deviation
    best_reg_idx = np.zeros(7)


    for step in steps:
        k = - 10
        best_ssim = 0
        best_corr = 0

        while k <= 10 :
            blue = blue_channel[step]
            red = red_channel[step - k]
            print(f"blue idx:{step}")
            print(f"red idx:{step-k}")

            # test quality of deviation
            ssim_value, _ = ssim(red, blue, full=True, data_range=65535)
            correlation_value = match_template(red, blue)

            # prints
            print(f"SSIM : {ssim_value}")
            print(f"correlation : {correlation_value}")
            print(f"best reg idx:{best_reg_idx}")
            print(f"best ssim:{best_ssim}")
            print(f"best correlation:{best_corr}")

            # saving best test values in order to find frame deviation
            if best_ssim < ssim_value and best_corr < correlation_value:
                best_ssim = ssim_value
                best_corr = correlation_value
                index = steps.index(step)
                best_reg_idx[index] = step - k

            k += 1
            print(f"k:{k}")


    # calculating average deviation
    average_deviation = sum(best_reg_idx[i] - steps[i] for i in range(0,7)) / 7
    average_deviation = np.round(average_deviation)
    print(f"average deviation:{average_deviation}")
    return int(average_deviation)

