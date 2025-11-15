import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim
from skimage.feature import match_template
from skimage import exposure


def create_grid_list(red, blue, reg, start, end):
    grid_list = []
    H, W = red[0].shape
    grid_y, grid_x = 8, 8


    # (Cells span the full image; the last row/col absorb any remainder)
    ys = [(H * i) // grid_y for i in range(grid_y)] + [H]
    xs = [(W * j) // grid_x for j in range(grid_x)] + [W]

    # ---- compute metrics over grid ----
    ssim_rb = np.zeros((grid_y, grid_x), dtype=np.float32)
    ssim_rr = np.zeros((grid_y, grid_x), dtype=np.float32)
    corr_rb = np.zeros((grid_y, grid_x), dtype=np.float32)
    corr_rr = np.zeros((grid_y, grid_x), dtype=np.float32)

    # ---- choose an SSIM window size that fits even small cells (odd number) ----
    def best_win_size(h, w):
        m = min(h, w)
        for cand in [11, 9, 7, 5, 3]:
            if m >= cand:
                return cand
        # if cells are tiny, fallback to 3 (skimage requires odd >=3)
        return 3

    # ---- helper: per-cell metrics ----
    def cell_metrics(a, b, win_size):
        # SSIM
        ssim_val = ssim(a, b, data_range=65535, win_size=win_size)
        # Correlation (normalized cross-correlation); take max (usually 1x1 -> same value)
        # convert to float in [0,1] to be safe for match_template
        a_float = a.astype(np.float32) / 65535.0
        b_float = b.astype(np.float32) / 65535.0
        corr_map = match_template(a_float, b_float, pad_input=False)
        corr_val = float(np.max(corr_map))
        return ssim_val, corr_val

    for i in range(start, end):
        red_img = red[i]
        blue_img = blue[i]
        reg_img = reg[i]

        for gy in range(grid_y):
            y0, y1 = ys[gy], ys[gy + 1]
            for gx in range(grid_x):
                x0, x1 = xs[gx], xs[gx + 1]
                r = red_img[y0:y1, x0:x1]
                b = blue_img[y0:y1, x0:x1]
                rg = reg_img[y0:y1, x0:x1]

                win = best_win_size(r.shape[0], r.shape[1])
                rb_ssim, rb_corr = cell_metrics(r, b, win)
                rr_ssim, rr_corr = cell_metrics(r, rg, win)

                ssim_rb[gy, gx] = rb_ssim
                ssim_rr[gy, gx] = rr_ssim
                corr_rb[gy, gx] = rb_corr
                corr_rr[gy, gx] = rr_corr

        # ---- build an annotation canvas (8-bit RGB) from red image for readability ----
        vis = exposure.rescale_intensity(red_img, in_range='image', out_range=(0, 255)).astype(np.uint8)
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

        # ---- draw grid and metrics ----
        for gy in range(grid_y):
            y0, y1 = ys[gy], ys[gy + 1]
            for gx in range(grid_x):
                x0, x1 = xs[gx], xs[gx + 1]

                # draw cell rectangle
                cv2.rectangle(vis, (x0, y0), (x1 - 1, y1 - 1), (255, 255, 255), 1)

                # choose text position & size
                cell_w, cell_h = (x1 - x0), (y1 - y0)
                font = cv2.FONT_HERSHEY_SIMPLEX
                # scale a bit by cell size (bounds help keep it legible)
                font_scale = max(0.3, min(cell_w, cell_h) / 120.0)
                thickness = 1

                # build the text (short forms to fit)
                line1 = f"SSIM rb={ssim_rb[gy, gx]:.2f} rr={ssim_rr[gy, gx]:.2f}"
                line2 = f"Corr rb={corr_rb[gy, gx]:.2f} rr={corr_rr[gy, gx]:.2f}"

                # inside the grid loop
                tx = x0 + 3
                # add a margin from the top
                top_margin = int(cell_h * 0.2)  # 20% of cell height
                line_spacing = int(cell_h * 0.5)  # 30% of cell height

                ty1 = y0 + top_margin
                ty2 = ty1 + line_spacing

                # draw with outline for readability
                def put_outlined(img, text, org):
                    font_scale = max(0.4, min(cell_w, cell_h) / 120.0) / 2.6
                    cv2.putText(img, text, org, font, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
                    cv2.putText(img, text, org, font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

                put_outlined(vis, line1, (tx, ty1))
                put_outlined(vis, line2, (tx, ty2))
                gray_vis = cv2.cvtColor(vis, cv2.COLOR_BGR2GRAY)

        grid_list.append(gray_vis)

    return grid_list
