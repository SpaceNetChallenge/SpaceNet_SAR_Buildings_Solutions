from .augmentations import get_spacenet6_augmentation
from .preprocesses import get_spacenet6_preprocess


def get_preprocess(config, is_test):
    """
    """
    return get_spacenet6_preprocess(config, is_test)


def get_augmentation(config, is_train):
    """
    """
    return get_spacenet6_augmentation(config, is_train)
