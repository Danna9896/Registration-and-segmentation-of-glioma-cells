from skimage import exposure
import tifffile as tiff
import numpy as np
import os


tiff_paths = ["P41_OR1.tif", "P41_OR2.tif", "P41_OR3.tif",
              "P42_OR1.tif", "P42_OR2.tif", "P42_OR3.tif",
              "R14_OR1.tif", "R14_OR2.tif", "R14_OR3.tif"]

for i in range(0,9):

    tiff_path1 = os.path.join("../images/denoised_images", tiff_paths[i])
    tiff_path2 = os.path.join("../images/contrast_images", tiff_paths[i])
    os.makedirs(os.path.dirname(tiff_path2), exist_ok=True)
    images = tiff.imread(tiff_path1)

    # initiating params
    black_threshold = 2000
    final = []
    avg = []
    idx = 0

    for i in range(images.shape[0]):
        img = images[i]
        avg.append(np.mean(img[img>2000]))
        img[img < black_threshold] = 0
        final.append(img)



    final = np.array(final)
    avg = np.array(avg)
    avg_idx = np.argsort(avg)

    for i in range(len(avg_idx)):
        if avg[avg_idx[i]] > 4000:
            idx = i
            print(idx)
            break


    small_avg = avg_idx[:int(idx/2)]
    big_avg = avg_idx[int(idx/2):idx]
    normal_avg = avg_idx[idx:]

    small = np.mean([np.mean(final[i][final[i]>2000]) for i in small_avg])
    big = np.mean([np.mean(final[i][final[i]>2000]) for i in big_avg])
    normal = np.mean([np.mean(final[i][final[i]>2000]) for i in normal_avg])


    for i in small_avg:
        final[i] = final[i] * (normal / small)

    for i in big_avg:
        final[i] = final[i] * (normal / big)



    contrasted_array = np.stack(final, axis=0)
    tiff.imwrite(tiff_path2, contrasted_array, imagej=True)
    print(f'finished {tiff_path2}')
