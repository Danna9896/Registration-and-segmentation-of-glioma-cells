# Imports
import numpy as np
import cv2
from skimage.segmentation import chan_vese
from skimage.util import img_as_float
from skimage.measure import label, regionprops
from skimage.morphology import closing, opening, disk

#parameters
tumoroid_thresh = 6000
min_grey = 600 #200
min_area = 20
mid_area = 100
bright = 2500
min_lap = 0.3

def segmentation_mask(denoised_images, threshold_type, tumoroid_images, start, end):
    # This function creates a segmentation mask based on greyscale intensity (keeps lighter values)
    masks = []
    for i, k in zip(range(start, end), range(tumoroid_images.shape[0])):
        print(f'processing mask for image {i}')
        denoised_img = denoised_images[i]
        tumoroid_img = tumoroid_images[k]

        temp = denoised_img.copy()
        if threshold_type == 'tumoroid':
            threshold = tumoroid_thresh
        elif threshold_type == 'normal':
            temp[tumoroid_img != 0] = 0
            non_black = temp[temp > min_grey]
            if non_black.size > 0:
                threshold = np.percentile(non_black, 90) #85
            else:
                threshold = min_grey

        temp[temp == 0] = threshold
        blurred = cv2.GaussianBlur(temp, (5, 5), 0)
        _, mask = cv2.threshold(blurred, threshold, 2 ** 16 - 1, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=1)
        result = np.zeros_like(denoised_img)
        result[mask == 255] = denoised_img[mask == 255]
        result = cv2.normalize(result, None, alpha=0, beta=2 ** 16 - 1, norm_type=cv2.NORM_MINMAX)
        masks.append(result)

    return masks


def _to_u8(img16):
    # robust 16→8 for flow (no external min/max)
    return cv2.convertScaleAbs(img16, alpha=255.0/65535.0)

def warp_prev_mask(prev_img16, curr_img16, prev_mask_u8):
    """
    Compute dense flow prev→curr and warp prev_mask into curr coordinates.
    Inputs: uint16 images, uint8 mask (0/255). Returns uint8 mask (0/255).
    """
    prev_u8 = _to_u8(prev_img16)
    curr_u8 = _to_u8(curr_img16)

    # Farnebäck flow; parameters are conservative and fast
    flow = cv2.calcOpticalFlowFarneback(prev_u8, curr_u8,
                                        None, 0.5, 3, 25, 3, 5, 1.2, 0)
    h, w = prev_u8.shape
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (grid_x + flow[...,0]).astype(np.float32)
    map_y = (grid_y + flow[...,1]).astype(np.float32)

    warped = cv2.remap(prev_mask_u8, map_x, map_y, interpolation=cv2.INTER_NEAREST,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return warped


def segmentation_images(denoised_images, masks, red_imgs, start, end):
    # Segmentation using Chan-vese
    out = []
    for i, k in zip(range(start, end), range(masks.shape[0])):
        print(f'Segmenting image {i}')
        img  = img_as_float(denoised_images[i])        # scale to [0,1]
        init = (masks[k] > 0).astype(bool)             # boolean seeds only
        # label seeds so CV evolves *per object* (prevents global merging)
        lbl  = label(init)
        seg  = np.zeros_like(init, dtype=bool)

        for r in regionprops(lbl):
            r0, c0, r1, c1 = r.bbox
            pad = 20 #45 #6
            r0 = max(0, r0 - pad); c0 = max(0, c0 - pad)
            r1 = min(img.shape[0], r1 + pad); c1 = min(img.shape[1], c1 + pad)

            sub_img  = img[r0:r1, c0:c1]
            sub_init = (lbl[r0:r1, c0:c1] == r.label)

            sub_cv = chan_vese(
                sub_img,
                mu=0.08, #0.08
                lambda1=1.2, #1.5
                lambda2=1,
                dt=0.2,
                tol=1e-3,
                max_num_iter=100,
                init_level_set=sub_init,
                extended_output=False,
            )

            seg[r0:r1, c0:c1] |= sub_cv


        cv = seg.astype(np.uint8) * 255
        img = red_imgs[i].copy()
        img[cv == 0] = 0
        out.append(img)


    return out

def combining_segmentations(tumoroid_seg, cancel_cells_seg , red_imgs, start, end):

    #Combining both tumoroid and cancell cell segmentations
    temporal_seg = []
    for i, k in zip(range(start, end),range(tumoroid_seg.shape[0])):
        seg = red_imgs[i].copy()
        tum_seg = tumoroid_seg[k].copy()
        cancel_cell_seg = cancel_cells_seg[k].copy()
        seg[(cancel_cell_seg == 0) & (tum_seg == 0)] = 0
        temporal_seg.append(seg)


    #forward pass of aligning frames
    forward_aligned = []
    prev_mask = None
    for i, curr in enumerate(temporal_seg):
        curr_mask = (curr > 0).astype(np.uint8) * 255

        if prev_mask is not None:
            warped_prev = warp_prev_mask(red_imgs[i - 1], red_imgs[i], prev_mask)
            overlap = np.count_nonzero((warped_prev > 0) & (curr_mask > 0))
            prev_area = np.count_nonzero(warped_prev > 0)
            if prev_area > 2 and overlap / prev_area > min_lap:
                curr_mask = cv2.bitwise_or(curr_mask, warped_prev)

        forward_aligned.append(curr_mask)
        prev_mask = curr_mask.copy()


    #backwards pass of aligning frames
    back_filled = [forward_aligned[-1]]
    for i in range(len(forward_aligned) - 2, -1, -1):
        curr = forward_aligned[i]
        next_mask = back_filled[0]
        warped_next = warp_prev_mask(red_imgs[i + 1], red_imgs[i], next_mask)
        overlap = np.count_nonzero((warped_next > 0) & (curr > 0))
        next_area = np.count_nonzero(warped_next > 0)
        if next_area > 2 and overlap / next_area > min_lap:
            curr = cv2.bitwise_or(curr, warped_next)
        back_filled.insert(0, curr)

    seg_list = []
    for i in range(len(temporal_seg)):
        back = back_filled[i]
        seg = temporal_seg[i].copy()
        seg[back == 0] = 0
        seg_list.append(seg)

    final_seg = []
    for i, seg in enumerate(seg_list):
        mask_gray = seg.astype(np.uint16)
        work_seg = seg.copy()  # this one we actually erase from
        bin_mask = (work_seg > 0).astype(np.uint8) * 255

        contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            mask = np.zeros_like(mask_gray, dtype=np.uint8)
            cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
            mean_intensity = cv2.mean(mask_gray, mask=mask)[0]

            if (area < mid_area and mean_intensity < bright) or area < min_area:
                cv2.drawContours(work_seg, [contour], -1, 0, thickness=cv2.FILLED)

        final_seg.append(work_seg)
    return final_seg

