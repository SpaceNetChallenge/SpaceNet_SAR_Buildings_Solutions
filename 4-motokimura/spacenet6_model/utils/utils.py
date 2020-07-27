import cv2
import git
import json
import numpy as np
import pandas as pd
import rasterio
import shapely
import solaris as sol

from skimage import measure
from skimage.morphology import watershed


def config_filename():
    """
    """
    return 'config.yml'


def experiment_subdir(exp_id):
    """
    """
    assert 0 <= exp_id <= 9999
    return f'exp_{exp_id:04d}'


def ensemble_subdir(exp_ids):
    """
    """
    exp_ids_ = exp_ids.copy()
    exp_ids_.sort()
    subdir = 'exp'
    for exp_id in exp_ids_:
        subdir += f'_{exp_id:04d}'
    return subdir


def git_filename():
    """
    """
    return 'git.json'


def weight_best_filename():
    """
    """
    return 'model_best.pth'


def poly_filename():
    """
    """
    return 'solution.csv'


def imageid_filename():
    """
    """
    return 'imageid.json'


def train_list_filename(split_id):
    """
    """
    return f'train_{split_id}.json'


def val_list_filename(split_id):
    """
    """
    return f'val_{split_id}.json'


def dump_git_info(path):
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha

    git_info = {
        'version': '0.0.0',
        'sha': sha
    }

    with open(path, 'w') as f:
        json.dump(
            git_info,
            f,
            ensure_ascii=False,
            indent=4,
            sort_keys=False,
            separators=(',', ': ')
        )


def get_roi_mask(sar_image):
    mask = sar_image.sum(axis=-1) > 0.0
    return mask


def crop_center(array, crop_wh):
    """
    """
    _, h, w = array.shape
    crop_w, crop_h = crop_wh
    assert w >= crop_w
    assert h >= crop_h

    left = (w - crop_w) // 2
    right = crop_w + left
    top = (h - crop_h) // 2
    bottom = crop_h + top

    return array[:, top:bottom, left:right]


def dump_prediction_to_png(path, pred):
    """
    """
    c, h, w = pred.shape
    assert c <= 3

    array = np.zeros(shape=[h, w, 3], dtype=np.uint8)
    array[:, :, :c] = (pred * 255).astype(np.uint8).transpose((1, 2, 0))
    cv2.imwrite(path, array)


def load_prediction_from_png(path, n_channels):
    """
    """
    assert n_channels <= 3

    array = cv2.imread(path)
    pred = (array.astype(float) / 255.0)[:, :, :n_channels]
    return pred.transpose((2, 0, 1))  # HWC to CHW


def compute_building_score(pr_score_footprint, pr_score_boundary, alpha=0.0):
    """
    """
    pr_score_building = pr_score_footprint * (1.0 - alpha * pr_score_boundary)
    return pr_score_building.clip(min=0.0, max=1.0)


def score_to_mask(building_score, thresh=0.5):
    """
    """
    assert building_score.min() >= 0 and building_score.max() <= 1
    building_mask = (building_score > 0.5).astype(np.uint8)
    building_mask *= 255
    return building_mask


def gen_building_polys_using_contours(
        building_score,
        min_area_pix,
        score_thresh,
    ):
    """
    """
    df = sol.vector.mask.mask_to_poly_geojson(
        building_score,
        output_path=None,
        output_type='csv',
        min_area=min_area_pix,
        bg_threshold=score_thresh,
        do_transform=False,
        simplify=True
    )
    return df['geometry']


def gen_building_polys_using_watershed(
        building_score,
        seed_min_area_pix,
        min_area_pix,
        seed_score_thresh,
        main_score_thresh
    ):
    """
    """
    def remove_small_regions(pred, min_area):
        """
        """
        props = measure.regionprops(pred)
        for i in range(len(props)):
            if props[i].area < min_area:
                pred[pred == i + 1] = 0
        return measure.label(pred, connectivity=2, background=0)

    def mask_to_polys(mask):
        """
        """
        shapes = rasterio.features.shapes(y_pred.astype(np.int16), mask > 0)
        mp = shapely.ops.cascaded_union(
            shapely.geometry.MultiPolygon([
            shapely.geometry.shape(shape)
            for shape, value in shapes
        ]))

        if isinstance(mp, shapely.geometry.Polygon):
            df = pd.DataFrame({
                'geometry': [mp],
            })
        else:
            df = pd.DataFrame({
                'geometry': [p for p in mp],
            })
        return df['geometry']

    av_pred = (building_score > seed_score_thresh).astype(np.uint8)
    y_pred = measure.label(av_pred, connectivity=2, background=0)
    y_pred = remove_small_regions(y_pred, seed_min_area_pix)

    nucl_msk = 1 - building_score
    nucl_msk = (nucl_msk * 65535).astype('uint16')
    y_pred = watershed(
        nucl_msk,
        y_pred,
        mask=(building_score > main_score_thresh),
        watershed_line=True
    )
    y_pred = remove_small_regions(y_pred, min_area_pix)

    return mask_to_polys(y_pred)
