# imports
import tifffile as tiff
import numpy as np
import os
import Denoising_funcs

reg_channels = ["../images/registered_images/P41_OR1.tif",
                "../images/registered_images/P41_OR2.tif",
                "../images/registered_images/P41_OR3.tif",
                "../images/registered_images/P42_OR1.tif",
                "../images/registered_images/P42_OR2.tif",
                "../images/registered_images/P42_OR3.tif",
                "../images/registered_images/R14_OR1.tif",
                "../images/registered_images/R14_OR2.tif",
                "../images/registered_images/R14_OR3.tif"]

red_channels =  ["../images/P41/C1_P41_OR1.tif",
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

for i in range(5, 6):
    # reading tiff images
    reg_channel = tiff.imread(reg_channels[i])
    red_channel = tiff.imread(red_channels[i])
    tiff_path = os.path.join("../images/denoised_images", tiff_paths[i])
    os.makedirs(os.path.dirname(tiff_path), exist_ok=True)

    # initiating params
    norm_denoise = []
    start_idx, end_idx = 0, reg_channel.shape[0]

    # Denoising process for multiple images
    k = start_idx
    while k < end_idx:
        reference_tif = red_channel[k]
        registered_tif = reg_channel[k]
        ratio = (reference_tif + 1e-6) / (registered_tif + 1e-6)
        mask = ratio >= 1
        denoised = (reference_tif * mask).astype(np.uint16)
        norm_denoise.append(denoised)
        k += 1

    # saving the tiff images
    print(f'Denoised tiff: {tiff_paths[i]}')
    norm_denoise_array = np.stack(norm_denoise, axis=0)
    norm_denoise_array = norm_denoise_array.astype(np.uint16)
    tiff.imwrite(tiff_path, norm_denoise_array, imagej=True)

