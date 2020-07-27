import albumentations as albu

data_type = 'SAR-Intensity'
train_images = '/data/SN6_buildings/train/AOI_11_Rotterdam/'
masks_data_path = '/wdata/train_masks'
logs_path = '/wdata/segmentation_logs/'
folds_file = '/wdata/folds.csv'
load_from = '/wdata/segmentation_logs/8_3_reduce_2_unet_senet154/checkpoints/best.pth'
multiplier = 5

main_metric = 'dice'
minimize_metric = False
scheduler_mode = 'max'
device = 'cuda'
fold_number = 2
n_classes = 3
input_channels = 4
crop_size = (320, 320)
val_size = (928, 928)
original_size = (900, 900)

batch_size = 14
num_workers = 14
val_batch_size = 1

shuffle = True
lr = 1e-4
momentum = 0.0
decay = 0.0
loss = 'focal_dice'
optimizer = 'adam_gcc'
fp16 = False

alias = '8_3_reduce_'
model_name = 'unet_senet154'
scheduler = 'reduce_on_plateau'
patience = 3
alpha = 0.5
min_lr = 1e-6
thershold = 0.005

early_stopping = 75
min_delta = 0.005


augs_p = 0.5


best_models_count = 1

epochs = 75

weights = 'imagenet'
limit_files = None

preprocessing_fn = None
train_augs = albu.Compose([albu.OneOf([albu.RandomCrop(crop_size[0], crop_size[1], p=1.0)
                                       ], p=1.0),
                           albu.OneOf([albu.HorizontalFlip(p=augs_p),
                                       albu.VerticalFlip(p=augs_p)], p=augs_p),
                           albu.ShiftScaleRotate(shift_limit=0.0, scale_limit=0.1, rotate_limit=5, p=augs_p)
                           ], p=augs_p)

valid_augs = albu.Compose([albu.PadIfNeeded(min_height=val_size[0], min_width=val_size[1], p=1.0)])
