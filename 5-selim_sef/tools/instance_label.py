import warnings

import numpy as np
from skimage import measure
from skimage.morphology import watershed

warnings.simplefilter("ignore")
def label_mask(pred, main_threshold=0.4, seed_threshold=0.75, w_pixel_t=50, pixel_t=90):
    av_pred = pred
    av_pred = av_pred[..., 0] * (1 - av_pred[..., 2]) * (1 -  0.5 * av_pred[..., 1])
    av_pred = 1 * (av_pred > seed_threshold)
    av_pred = av_pred.astype(np.uint8)

    y_pred = measure.label(av_pred, neighbors=8, background=0)
    props = measure.regionprops(y_pred)
    for i in range(len(props)):
        if props[i].area < w_pixel_t:
            y_pred[y_pred == i + 1] = 0
    y_pred = measure.label(y_pred, neighbors=8, background=0)

    nucl_msk = (1 - pred[..., 0])
    nucl_msk = nucl_msk.astype('uint8')
    y_pred = watershed(nucl_msk, y_pred, mask=(pred[..., 0] > main_threshold), watershed_line=True)

    props = measure.regionprops(y_pred)

    for i in range(len(props)):
        if props[i].area < pixel_t:
            y_pred[y_pred == i + 1] = 0
    y_pred = measure.label(y_pred, neighbors=8, background=0)
    return y_pred
