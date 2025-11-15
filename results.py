import tifffile as tif
import os
import numpy as np
import Segmentation_funcs as sf
import quality_reg_grid

# Reading files
red = tif.imread('../images/R14/C1_R14_OR3.tif')
blue = tif.imread('../images/R14/C2_R14_OR3.tif')
reg = tif.imread('../images/registered_images/R14_OR3.tif')
denoised = tif.imread('../images/denoised_images/R14_OR3.tif')
clean = tif.imread('../images/cleaned/R14_OR3.tif')

out_path = "../images/segmentation/R14_OR3"
tiff_path1 = os.path.join(out_path, "grid.tif")
tiff_path2 = os.path.join(out_path, "segmentation.tif")
os.makedirs(os.path.dirname(tiff_path1), exist_ok=True)
os.makedirs(os.path.dirname(tiff_path2), exist_ok=True)




# Initializing params
met_grid_list = []
seg_list = []
start = 0
end = red.shape[0]


## Segmentation ##

# Big tumoroid segmentation
mask1 = sf.segmentation_mask(clean, 'tumoroid', clean, start, end)
mask1 = np.stack(mask1, axis=0).astype(np.uint16)
tumoroid_seg = sf.segmentation_images(clean, mask1, red, start, end)
tumoroid_seg_array = np.stack(tumoroid_seg, axis=0).astype(np.uint16)

# Cancer cells segmentation
mask2 = sf.segmentation_mask(clean, 'normal', tumoroid_seg_array, start, end)
mask2 = np.stack(mask2, axis=0).astype(np.uint16)
cancer_cells_seg = sf.segmentation_images(clean, mask2, red, start, end)
cancer_cells_seg_array = np.stack(cancer_cells_seg, axis=0).astype(np.uint16)

# Combining both segmentations
segmented_imgs = sf.combining_segmentations(tumoroid_seg_array, cancer_cells_seg_array, red, start, end)
segmented_images_array = np.stack(segmented_imgs, axis=0).astype(np.uint16)

# Creating arrays
grid_list = quality_reg_grid.create_grid_list(red, blue, reg, start, end)
grid_array = np.stack(grid_list, axis=0).astype(np.uint16)


# Saving files
tif.imwrite(tiff_path1, grid_array, imagej=True)
tif.imwrite(tiff_path2, segmented_images_array, imagej=True)


## MASKS ##
# tiff_path3 = os.path.join(out_path, "mask1_R14_OR2.tif")
# tiff_path4 = os.path.join(out_path, "mask2_R14_OR2.tif")
# tif.imwrite(tiff_path3, mask1, imagej=True)
# tif.imwrite(tiff_path4, mask2, imagej=True)
#
# tiff_path5 = os.path.join(out_path, "red_R14_OR2.tif")
# new = red[start:end]
# tif.imwrite(tiff_path5, new, imagej=True)
# tiff_path6 = os.path.join(out_path, "blue_R14_OR2.tif")
# new2 = blue[start:end]
# tif.imwrite(tiff_path6, new2, imagej=True)
# tiff_path7 = os.path.join(out_path, "reg_R14_OR2.tif")
# new3 = reg[start:end]
# tif.imwrite(tiff_path7, new3, imagej=True)
# tiff_path8 = os.path.join(out_path, "denoised_R14_OR2.tif")
# new4 = denoised[start:end]
# tif.imwrite(tiff_path8, new4, imagej=True)
# tiff_path9 = os.path.join(out_path, "cleaned_R14_OR2.tif")
# new5 = clean[start:end]
# tif.imwrite(tiff_path9, new5, imagej=True)