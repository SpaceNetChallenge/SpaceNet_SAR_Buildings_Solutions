import skimage
import pandas as pd
import torch
from torch.utils.data import DataLoader
import numpy as np
from pytorch_toolbelt.inference.tiles import ImageSlicer, CudaTileMerger
from pytorch_toolbelt.utils.torch_utils import to_numpy
import sys
sys.path.append('.')


def infer_one(model, mask, tile_size=(512, 512), tile_step=(256, 256), weight='mean'):
    image = mask.cpu().numpy()
    image = np.moveaxis(image, 0, -1)

    with torch.no_grad():
        tiler = ImageSlicer((900, 900), tile_size=tile_size, tile_step=tile_step, weight=weight)
        tiles = [np.moveaxis(tile, -1, 0) for tile in tiler.split(image)]
        merger = CudaTileMerger(tiler.target_shape, 1, tiler.weight)

        for tiles_batch, coords_batch in DataLoader(list(zip(tiles, tiler.crops)), batch_size=10, pin_memory=False):
            tiles_batch = tiles_batch.float().cuda()
            pred_batch = model(tiles_batch)
            tiles_batch.cpu().detach()
            merger.integrate_batch(pred_batch, coords_batch)
    merged_mask = np.moveaxis(to_numpy(merger.merge()), 0, -1)
    merged_mask = tiler.crop_to_orignal_size(merged_mask)

    m = merged_mask[..., 0].copy()
    return m


def space_metric(dl, model, truth_df, threshold=0.5, minbuildingsize=120,
                 tile_size=(512, 512), tile_step=(256, 256), weight='mean'):
    f1metric = []
    tmc = []
    pmc = []
    icount = 0
    with torch.no_grad():
        for (x_batch, y_batch, c_batch, tile_batch, orient_batch) in dl:

            icount = icount + 1
            if icount % 10 == 0:
                print('Batch ', icount)

            for i in range(len(x_batch)):
                pred = infer_one(model, x_batch[i, ...], tile_size, tile_step, weight)

                orient = orient_batch[i]
                tile_id = tile_batch[i]

                if orient == 1:
                    pred = np.fliplr(np.flipud(pred))

                pred[np.where(pred > threshold)] = 1
                pred[np.where(pred <= threshold)] = 0

                regionlabels, regioncount = skimage.measure.label(pred, background=0, connectivity=1, return_num=True)
                regionproperties = skimage.measure.regionprops(regionlabels)
                for blab in range(regioncount):
                    if regionproperties[blab].area < minbuildingsize:
                        pred[regionlabels == blab + 1] = 0

                vectordata = solaris.vector.mask.mask_to_poly_geojson(
                    pred,
                    min_area=0,
                    bg_threshold=0.5,
                    do_transform=False,
                    simplify=True
                )

                csvaddition = pd.DataFrame({'ImageId': tile_id,
                                            'BuildingId': 0,
                                            'PolygonWKT_Pix': vectordata['geometry'],
                                            'Confidence': 1
                                            })
                csvaddition.to_csv('tmp_proposal_val_train.csv', index=False)

                #                 #
                #                 # Extract ground truth from global truth_df
                aTruth = truth_df[truth_df.ImageId == tile_id]
                aTruth.to_csv('tmp_ground_truth_val_train.csv', index=False)
                truth_masks_count = len(aTruth)
                pred_masks_count = len(csvaddition)

                #                 #
                #                 # do eval
                evaluator = solaris.eval.base.Evaluator('tmp_ground_truth_val_train.csv')
                evaluator.load_proposal('tmp_proposal_val_train.csv', proposalCSV=True,
                                        conf_field_list=[])
                report = evaluator.eval_iou_spacenet_csv(miniou=0.5, min_area=80)

                assert (len(report) == 1)
                f1metric.append(report[0]['F1Score'])
                pmc.append(pred_masks_count)
                tmc.append(truth_masks_count)

    f1metric = np.array(f1metric)
    pmc = np.array(pmc)
    tmc = np.array(tmc)
    return f1metric.mean(), pmc.mean(), tmc.mean()


from segmentation_models_pytorch.utils import functional  as F
from segmentation_models_pytorch.utils.base import Loss, Activation


class BCEDiceLoss2Class(Loss):

    def __init__(self, eps=1., beta=1., activation=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.beta = beta
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        # interior
        y_pr_0 = y_pr[:, 0, :, :]
        y_gt_0 = y_gt[:, 0, :, :]
        bce_0 = torch.nn.functional.binary_cross_entropy(y_pr_0, y_gt_0)
        y_pr_0 = self.activation(y_pr_0)
        dice_0 = 1 - F.f_score(y_pr_0, y_gt_0, beta=self.beta, eps=self.eps, threshold=None,
                               ignore_channels=self.ignore_channels)
        loss_0 = 0.5 * bce_0 + 0.5 * dice_0
        # outline
        y_pr_1 = y_pr[:, 1, :, :]
        y_gt_1 = y_gt[:, 1, :, :]
        bce_1 = torch.nn.functional.binary_cross_entropy(y_pr_1, y_gt_1)
        y_pr_1 = self.activation(y_pr_1)
        dice_1 = 1 - F.f_score(y_pr_1, y_gt_1, beta=self.beta, eps=self.eps, threshold=None,
                               ignore_channels=self.ignore_channels)
        loss_1 = 0.5 * bce_1 + 0.5 * dice_1

        return 0.7 * loss_0 + 0.3 * loss_1

