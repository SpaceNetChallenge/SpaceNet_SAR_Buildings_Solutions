import argparse
import os
import warnings

import zoo
from spacenet_dataset import TestSpacenetLocDataset

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import cv2

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)
import numpy as np
import torch
from albumentations import Compose, Resize
from torch.utils.data import DataLoader
from tqdm import tqdm
from tools.config import load_config

warnings.simplefilter("ignore")


def load_model(config_path, weights_path):
    conf = load_config(config_path)
    model = zoo.__dict__[conf['network']](seg_classes=3, backbone_arch=conf['encoder'])
    model = torch.nn.DataParallel(model).cuda()
    print("=> loading checkpoint '{}'".format(weights_path))
    checkpoint = torch.load(weights_path, map_location="cpu")
    print("best_dice", checkpoint['dice_best'])
    print("f_score_best", checkpoint['f_score_best'])
    print("epoch", checkpoint['epoch'])
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model

def predict_model(model, out_root_dir, prefix, scale):
    out_dir = os.path.join(out_root_dir, "{}_{}".format(prefix, str(scale)))
    os.makedirs(out_dir, exist_ok=True)
    transforms = Compose([
        Resize(scale, scale),
    ])
    dataset = TestSpacenetLocDataset(data_path=args.data_path,
                                     transforms=transforms,
                                     orientation_csv="sar_orientations.txt")
    data_loader = DataLoader(dataset, batch_size=8, num_workers=12, shuffle=False, pin_memory=False)

    def predict(image, model):
        logits = model(image)
        preds = torch.sigmoid(logits).cpu().numpy()
        preds = np.around((np.moveaxis(preds, 1, -1) * 255)).astype(np.uint8)
        return preds

    with torch.no_grad():
        for sample in tqdm(data_loader):
            ids = sample['img_name']
            orientations = sample['orientation']
            image = sample['image'].cuda()
            predictions = predict(image, model)
            for i in range(len(ids)):
                preds = predictions[i]
                if orientations[i] > 0:
                    preds = cv2.rotate(preds, cv2.ROTATE_180)
                cv2.imwrite(out_dir + "/" + ids[i] + ".png", preds)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Spacenet Test Predictor")
    arg = parser.add_argument
    arg('--config', metavar='CONFIG_FILE', default='configs/b5.json', help='path to configuration file')
    arg('--data-path', type=str, default='/mnt/sota/datasets/spacenet/test_public/AOI_11_Rotterdam/',
        help='Path to test images')
    arg('--gpu', type=str, default='0', help='List of GPUs for parallel training, e.g. 0,1,2,3')
    arg('--dir', type=str, default='../test_results/multiscale')
    arg('--prefix', type=str, default='b5run4')
    arg('--model', type=str, default='weights/runs/run4_eff_unet_tf_efficientnet_b5_ns_0_last')
    arg('--scales', type=int, nargs='+', default=(864, 928, 1024, 1088, 1152, 1280))

    args = parser.parse_args()
    os.makedirs(args.dir, exist_ok=True)

    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    model = load_model(args.config, args.model)
    for scale in args.scales:
        predict_model(model, args.dir, args.prefix, scale)
