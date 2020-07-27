import matplotlib.pyplot as plt
import numpy as np


def adjust_sar_contrast(sar_intensity, max_percentile=99.95, min_percentile=0.05):
    """Stretch histogram of SAR intensity image
    """
    sar_intensity = sar_intensity.copy().astype(np.float32)
    min_ = np.percentile(sar_intensity[sar_intensity > 0.0], min_percentile)
    max_ = np.percentile(sar_intensity, max_percentile)
    
    sar_intensity = sar_intensity.clip(min=min_, max=max_)
    sar_intensity = (sar_intensity - min_) / (max_ - min_)
    
    sar_intensity = (sar_intensity * 255.0).astype(np.uint8)
    return sar_intensity


def compute_building_score(pr_score_footprint, pr_score_boundary, alpha):
    """
    """
    # XXX: deprecated. will be moved under spacenet6_model/utils/utils.py
    pr_score_building = pr_score_footprint * (1.0 - alpha * pr_score_boundary)
    return pr_score_building.clip(min=0.0, max=1.0)


def plot_images(**image_cmap_pairs):
    """PLot images in one row
    """
    n = len(image_cmap_pairs)
    plt.figure(figsize=(16, 5))
    for i, (name, image_cmap) in enumerate(image_cmap_pairs.items()):
        image, cmap = image_cmap
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')))
        plt.imshow(image, cmap=cmap)
    plt.tight_layout()
    plt.show()
