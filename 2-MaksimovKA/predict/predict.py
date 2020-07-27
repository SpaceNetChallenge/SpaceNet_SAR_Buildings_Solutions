import torch
import os
import shutil
import numpy as np
import albumentations as albu
import cv2
from utils.helpers import get_config
from models.model_factory import make_model
from pytorch_toolbelt.inference import tta
from tta import flip_image2mask
from dataset.semseg_dataset import TestSemSegDataset
from torch.utils.data import DataLoader
from fire import Fire
from tqdm import tqdm


def main(config_path='/project/configs/senet154_gcc_fold1.py',
         test_images='/data/SN6_buildings/test_public/AOI_11_Rotterdam/',
         test_predict_result='/wdata/folds_predicts/',
         batch_size=1,
         workers=1,
         gpu='1'):

    with torch.no_grad():

        config = get_config(config_path)
        model_name = config['model_name']
        weights_path = config['load_from']
        device = config['device']
        val_batch_size = batch_size
        input_channels = config['input_channels']

        original_size = config['original_size']
        cropper = albu.Compose([albu.CenterCrop(original_size[0], original_size[1], p=1.0)])
        n_classes = config['n_classes']
        preprocessing_fn = config['preprocessing_fn']
        valid_augs = config['valid_augs']
        limit_files = config['limit_files']
        num_workers = workers
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu
        if not os.path.exists(test_predict_result):
            os.mkdir(test_predict_result)
        fold_name = weights_path.split('/')[-3]
        folder_to_save = os.path.join(test_predict_result, fold_name)
        if os.path.exists(folder_to_save):
            shutil.rmtree(folder_to_save)

        os.mkdir(folder_to_save)

        test_dataset = TestSemSegDataset(images_dir=os.path.join(test_images, 'SAR-Intensity'),
                                         preprocessing=preprocessing_fn,
                                         augmentation=valid_augs,
                                         limit_files=limit_files)

        test_loader = DataLoader(dataset=test_dataset,
                                 batch_size=val_batch_size,
                                 shuffle=False,
                                 num_workers=num_workers)
        print('Loading {}'.format(weights_path))
        model = make_model(
            model_name=model_name,
            weights=None,
            n_classes=n_classes,
            input_channels=input_channels).to(device)

        model.load_state_dict(torch.load(weights_path)['model_state_dict'])

        model.eval()
        model = tta.TTAWrapper(model, flip_image2mask)
        model = torch.nn.DataParallel(model).cuda()

        file_names = sorted(test_dataset.ids)

        for batch_i, test_batch in enumerate(tqdm(test_loader)):
            runner_out = model(test_batch.cuda())
            image_pred = runner_out

            image_pred = image_pred.cpu().detach().numpy()
            names = file_names[batch_i * val_batch_size:(batch_i + 1) * val_batch_size]
            for i in range(len(names)):
                file_name = os.path.join(folder_to_save, names[i].split('.')[0] + '.png')

                data = image_pred[i, ...]
                data = np.moveaxis(data, 0, -1)
                sample = cropper(image=data)
                data = sample['image']
                data = (data * 255).astype(np.uint8)
                cv2.imwrite(file_name, data)


if __name__ == '__main__':
    Fire(main)


