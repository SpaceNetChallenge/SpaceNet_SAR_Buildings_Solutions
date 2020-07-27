import json
import os.path
import torch.utils.data

from glob import glob

from .spacenet6 import SpaceNet6Dataset, SpaceNet6TestDataset
from ..transforms import get_augmentation, get_preprocess
from ..utils import train_list_filename, val_list_filename


def get_dataloader(
    config,
    is_train
):
    """
    """
    # get path to train/val json files
    split_id = config.INPUT.TRAIN_VAL_SPLIT_ID
    train_list = os.path.join(
        config.INPUT.TRAIN_VAL_SPLIT_DIR,
        train_list_filename(split_id)
    )
    val_list = os.path.join(
        config.INPUT.TRAIN_VAL_SPLIT_DIR,
        val_list_filename(split_id)
    )

    preprocessing = get_preprocess(config, is_test=False)
    augmentation = get_augmentation(config, is_train=is_train)

    if is_train:
        data_list_path = train_list
        batch_size = config.DATALOADER.TRAIN_BATCH_SIZE
        num_workers = config.DATALOADER.TRAIN_NUM_WORKERS
        shuffle = config.DATALOADER.TRAIN_SHUFFLE
    else:
        data_list_path = val_list
        batch_size = config.DATALOADER.VAL_BATCH_SIZE
        num_workers = config.DATALOADER.VAL_NUM_WORKERS
        shuffle = False

    dataset = SpaceNet6Dataset(
        config,
        data_list_path,
        augmentation=augmentation,
        preprocessing=preprocessing
    )

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )


def get_test_dataloader(config):
    """
    """
    preprocessing = get_preprocess(config, is_test=True)
    augmentation = get_augmentation(config, is_train=False)

    # get full paths to SAR-Intensity image files
    if config.TEST_TO_VAL:
        # use val split for test.
        val_list_path = os.path.join(
            config.INPUT.TRAIN_VAL_SPLIT_DIR,
            val_list_filename(config.INPUT.TRAIN_VAL_SPLIT_ID)
        )
        with open(val_list_path) as f:
            val_list = json.load(f)
        image_paths = [
            os.path.join(
                config.INPUT.IMAGE_DIR,
                'SAR-Intensity',
                data['SAR-Intensity']
            ) for data in val_list
        ]
    else:
        # use test data for test (default).
        image_paths = glob(os.path.join(config.INPUT.TEST_IMAGE_DIR, '*.tif'))

    dataset = SpaceNet6TestDataset(
        config,
        image_paths,
        augmentation=augmentation,
        preprocessing=preprocessing
    )

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=config.DATALOADER.TEST_BATCH_SIZE,
        shuffle=False,
        num_workers=config.DATALOADER.TEST_NUM_WORKERS
    )
