import tifffile as tiff
import numpy as np
from skimage.metrics import structural_similarity as ssim

# Paths
tiff_paths_after = [
    "../images/registered_images_3D_final/P41_OR1.tif",
    "../images/registered_images_3D_final/P41_OR2.tif",
    "../images/registered_images_3D_final/P41_OR3.tif",
    "../images/registered_images_3D_final/P42_OR1.tif",
    "../images/registered_images_3D_final/P42_OR2.tif",
    "../images/registered_images_3D_final/P42_OR3.tif",
    "../images/registered_images_3D_final/R14_OR1.tif",
    "../images/registered_images_3D_final/R14_OR2.tif",
    "../images/registered_images_3D_final/R14_OR3.tif"
]

tiff_paths_before = [
    "../images/registered_images_3D/P41_OR1.tif",
    "../images/registered_images_3D/P41_OR2.tif",
    "../images/registered_images_3D/P41_OR3.tif",
    "../images/registered_images_3D/P42_OR1.tif",
    "../images/registered_images_3D/P42_OR2.tif",
    "../images/registered_images_3D/P42_OR3.tif",
    "../images/registered_images_3D/R14_OR1.tif",
    "../images/registered_images_3D/R14_OR2.tif",
    "../images/registered_images_3D/R14_OR3.tif"
]

red_channels = [
    "../images/P41/C1_P41_OR1.tif",
    "../images/P41/C1_P41_OR2.tif",
    "../images/P41/C1_P41_OR3.tif",
    "../images/P42/C1_P42_OR1.tif",
    "../images/P42/C1_P42_OR2.tif",
    "../images/P42/C1_P42_OR3.tif",
    "../images/R14/C1_R14_OR1.tif",
    "../images/R14/C1_R14_OR2.tif",
    "../images/R14/C1_R14_OR3.tif"
]

results = []

# Loop over all sets
for i in range(4,9):
    print(f"Processing {red_channels[i]}...")

    reg_before = tiff.imread(tiff_paths_before[i])
    reg_after = tiff.imread(tiff_paths_after[i])
    red = tiff.imread(red_channels[i])
    print(f'read all tiff files in idx: {i}')

    ssim_deltas = []
    corr_deltas = []

    for k in range(0, red.shape[0], 5):
        red_frame = red[k].astype(np.float32)
        reg_before_frame = reg_before[k].astype(np.float32)
        reg_after_frame = reg_after[k].astype(np.float32)

        ssim_before = ssim(reg_before_frame, red_frame, data_range=65535)
        ssim_after = ssim(reg_after_frame, red_frame, data_range=65535)
        ssim_deltas.append(ssim_after - ssim_before)

        corr_before = np.corrcoef(reg_before_frame.flatten(), red_frame.flatten())[0, 1]
        corr_after = np.corrcoef(reg_after_frame.flatten(), red_frame.flatten())[0, 1]
        corr_deltas.append(corr_after - corr_before)
        print(f'computed ssim and corr for idx num {k}, ssim:{ssim_after - ssim_before}, diff in corr:{corr_after - corr_before}')

    avg_ssim = np.mean(ssim_deltas)
    avg_corr = np.mean(corr_deltas)
    median_ssim = np.median(ssim_deltas)
    median_corr = np.median(corr_deltas)
    result_line = f"{red_channels[i]} | ΔSSIM: average {avg_ssim} median {median_ssim} | ΔCorrelation: average {avg_corr} median {median_corr}"
    results.append(result_line)
    print(result_line)

# Save results
with open("registration_improvement_results_new.txt", "w") as f:
    for line in results:
        f.write(line + "\n")

print("\nAll results saved to registration_improvement_results.txt")
