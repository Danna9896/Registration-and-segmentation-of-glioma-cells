# imports
from image_registration import chi2_shift
from skimage.registration import phase_cross_correlation
from skimage.registration import optical_flow_tvl1
from scipy.ndimage import shift, median_filter
import numpy as np
from skimage.util import img_as_float
yoff, xoff = 0, 0


# registration algorithms
def registration1(contrasted_red, contrasted_blue, image, prev_yoff=0, prev_xoff=0, num_iter=10):
    red = img_as_float(contrasted_red)
    blue = img_as_float(contrasted_blue)
    v, u = optical_flow_tvl1(red, blue, num_iter = num_iter)
    vfilt = median_filter(v, size=15)
    ufilt = median_filter(u, size=15)
    vstd = np.std(vfilt)
    ustd = np.std(ufilt)
    mask = (np.abs(vfilt - np.median(vfilt)) < vstd) & (np.abs(ufilt - np.median(ufilt)) < ustd)
    if np.any(mask):
        xoff = np.median(vfilt[mask])
        yoff = np.median(ufilt[mask])
    else:
        xoff = np.median(vfilt)
        yoff = np.median(ufilt)
    alpha = 1 if (prev_xoff == 0 and prev_yoff == 0) else 0.2
    xoff = alpha * xoff + (1 - alpha) * prev_xoff
    yoff = alpha * yoff + (1 - alpha) * prev_yoff

    registered_image = shift(image, shift=(yoff, xoff), mode='constant')
    return registered_image, yoff, xoff

def registration2(contrasted_image1, contrasted_image2, image, prev_yoff = 0, prev_xoff = 0):
    # Subpixel precision using phase_cross_correlation
    shifted, error, _ = phase_cross_correlation(contrasted_image1, contrasted_image2, upsample_factor= 100)
    alpha = 1 if (prev_xoff == 0 and prev_yoff == 0) else 0.2
    xoff = alpha*shifted[1] + (1-alpha)*prev_xoff
    yoff = alpha*shifted[0] + (1-alpha)*prev_yoff
    registered_image = shift(image, shift=(yoff, xoff), mode='constant')
    return registered_image, yoff, xoff

def registration3(contrasted_image1, contrasted_image2, image, prev_yoff = 0, prev_xoff = 0):
    xoff, yoff, _, _ = chi2_shift(contrasted_image1, contrasted_image2, 0.1, return_error= True, upsample_factor= 'auto')
    registered_image = shift(image, shift= (xoff,yoff), mode= 'constant')
    alpha = 1 if (prev_xoff == 0 and prev_yoff == 0) else 0.2
    xoff = alpha*xoff + (1-alpha)*prev_xoff
    yoff = alpha*yoff + (1-alpha)*prev_yoff
    return registered_image, yoff, xoff

# decision maker function to decide between algorithms
def decide_algorithm(a, time_taken, ssim_value, diff, num_iter):
    # improving registration quality
    if time_taken <= 0.5:
        num_iter = 20
        if ssim_value < 0.01 or diff < 0.02:
            if a == 1:
                a = 2
            elif a == 2:
                a = 3
            elif a == 3:
                a = 1
    elif 0.5 < time_taken < 1:
        num_iter = 10
        if ssim_value < 0.01 or diff < 0.03:
            if a == 1:
                a = 2
            elif a == 2:
                a = 3
            elif a == 3:
                a = 1
                num_iter = 20
    # registration is too time costly
    elif time_taken >= 1:
        num_iter = 5
        if ssim_value < 0.02 or diff < 0.04:
            if a == 1:
                a = 2
            elif a == 2:
                a = 3
            elif a == 3:
                a = 1
                num_iter = 20
    return a, num_iter
