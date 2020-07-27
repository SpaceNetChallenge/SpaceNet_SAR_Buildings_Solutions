import random

from albumentations.augmentations import functional as F
from albumentations import Compose, PadIfNeeded, RandomBrightnessContrast, \
    RandomGamma, RandomRotate90, IAAAdditiveGaussianNoise, MedianBlur, DualTransform, HorizontalFlip, OneOf, \
    GridDistortion, ElasticTransform, VerticalFlip, ImageOnlyTransform, ISONoise, GaussNoise, GaussianBlur, Resize, \
    IAAPiecewiseAffine, IAAPerspective, IAAAffine, OpticalDistortion

import cv2

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)

IMAGE_SIZE = 900


class RandomSizedCrop2x(DualTransform):
    def __init__(
            self, height, width, always_apply=False, p=1.0, scale_shift=0.65):
        super(RandomSizedCrop2x, self).__init__(always_apply=always_apply, p=p)
        self.height = height
        self.width = width
        self.scale = scale_shift

    def get_params(self):
        if random.random() > 0.5:
            # scale up
            scale = 1 + self.scale * random.random()
        else:
            # scale down
            scale = 1 + self.scale * random.random()
            scale = 1 / scale
        crop_size_height = int(self.height * scale)
        crop_size_width = int(self.width * scale)
        if crop_size_width > IMAGE_SIZE:
            crop_size_height = int(crop_size_height / (crop_size_width / IMAGE_SIZE))
            crop_size_width = IMAGE_SIZE
        elif crop_size_height > IMAGE_SIZE:
            crop_size_width = int(crop_size_width / (crop_size_height / IMAGE_SIZE))
            crop_size_height = IMAGE_SIZE
        h_start = 0
        w_start = 0

        if crop_size_height < IMAGE_SIZE:
            h_start = max(0, random.randint(0, IMAGE_SIZE - crop_size_height))
        if crop_size_width < IMAGE_SIZE:
            w_start = max(0, random.randint(0, IMAGE_SIZE - crop_size_width))

        interpolation = random.choice([cv2.INTER_LINEAR, cv2.INTER_CUBIC]) if scale >= 1 else random.choice(
            [cv2.INTER_LINEAR, cv2.INTER_AREA])
        params =  {
            "h_start": h_start,
            "w_start": w_start,
            "crop_height": crop_size_height,
            "crop_width": crop_size_width,
            "interpolation": interpolation
        }
        return params

    def apply(self, img, crop_height=0, crop_width=0, h_start=0, w_start=0, interpolation=cv2.INTER_LINEAR, **params):
        img = img[h_start:h_start + crop_height, w_start:w_start + crop_width]
        return F.resize(img, self.height, self.width, interpolation)

    def get_transform_init_args_names(self):
        return "height"

def train_trasforms_standard(conf):
    height = conf['crop_height']
    width = conf['crop_width']
    return Compose([
        RandomSizedCrop2x(height, width, scale_shift=0.5, p=1),
        OneOf([IAAPiecewiseAffine(),
               IAAPerspective(),
               OpticalDistortion(border_mode=cv2.BORDER_CONSTANT),
               GridDistortion(border_mode=cv2.BORDER_CONSTANT),
               ElasticTransform(border_mode=cv2.BORDER_CONSTANT)],
              p=0.3),
        RandomBrightnessContrast(),
        RandomGamma(),
        OneOf([MedianBlur(), GaussianBlur()], p=0.2),
        OneOf([IAAAdditiveGaussianNoise(per_channel=True),GaussNoise()])
    ]
    )
