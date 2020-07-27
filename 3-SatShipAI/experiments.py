import os
import shutil
import numpy as np
import segmentation_models_pytorch as smp
import torch
from pytorch_toolbelt.inference.tiles import ImageSlicer, CudaTileMerger
from pytorch_toolbelt.utils.torch_utils import to_numpy

from albumentations import (
    HorizontalFlip,
    VerticalFlip,
    CenterCrop,
    Compose,
    RandomRotate90,
    RandomResizedCrop,
    OneOf,
    RandomBrightnessContrast,
    Normalize, RandomCrop, Blur)
from albumentations.pytorch.transforms import ToTensor
from torch.utils.data import DataLoader

import datagen
import epoch
import logs
import loss
#from datagen import SpaceNetSAR2RGBSteroids, SpaceNetRGB
#from epoch import TrainEpoch, ValidEpoch, WeightedTrainEpoch, WeightedValidEpoch
#from logs import print_log_headers, print_log_lines, print_log_lines_console
#from loss import BCEDiceLoss, BCEDiceLossWeighted

TYPE_VOGLIS = -1
TYPE_NASIOS =  1
EXPERIMENT_TYPE_SAR = 0
EXPERIMENT_TYPE_RGB = 1
EXPERIMENT_TYPE_LOSS_SIMPLE = 0
EXPERIMENT_TYPE_LOSS_WEIGHTED = 1
EXPERIMENT_TYPE_USE_STEROID = 0
EXPERIMENT_TYPE_USE_ORIGINAL = 1
MASK_TYPE_BUILDING = 0
MASK_TYPE_CONTACT = 1


experiments = [
    {  # Experiment no1 with se_resnext50_32x4d
        'exp_id': 'Unet_se_resnext50_32x4d_v1',
        'weight': 3.0,
        'nfolds': 4,
        'rgb': {
            'id': 'Unet_se_resnext50_32x4d_v1_rgb_full',
            'type': EXPERIMENT_TYPE_RGB,
            'mask': MASK_TYPE_BUILDING,
            'model': {
                'model_type': smp.Unet,
                'encoder': 'se_resnext50_32x4d',
                'activation': 'sigmoid',
                'output_class': 1,
                'size': 512
            },
            'train': {
                'mean': [0.485, 0.456, 0.406],
                'std': [0.229, 0.224, 0.225],
                'loss': EXPERIMENT_TYPE_LOSS_SIMPLE,
                'data': EXPERIMENT_TYPE_USE_ORIGINAL,
                'contact': 13,
                'init_lr': 0.0005,
                'batch_size':  12,
                'epochs': 60,
                'train_transform': Compose(
                    [RandomCrop(512, 512, p=1.0),
                     HorizontalFlip(p=.5),
                     VerticalFlip(p=.5),
                     RandomRotate90(p=0.5),
                     RandomBrightnessContrast(p=0.3),
                     Blur(p=0.3),
                     Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225]),
                     ToTensor(),
                     ]),
                'valid_transform': Compose(
                    [CenterCrop(512, 512, p=1.0),
                     Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225]),
                     ToTensor()
                     ]),
                'infer_transform': Compose(
                    [
                        Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225]),
                        ToTensor()
                    ])
            }
        },
        'sar': {
            'id': 'Unet_se_resnext50_32x4d_v1_sar',
            'type': EXPERIMENT_TYPE_SAR,
            'mask': MASK_TYPE_BUILDING,

            'model': {
                'model_type': smp.Unet,
                'encoder': 'se_resnext50_32x4d',
                'activation': 'sigmoid',
                'output_class': 1,
                'size': 512
            },
            'train': {
                'mean': [0.485, 0.456, 0.406],
                'std': [0.229, 0.224, 0.225],
                'loss': EXPERIMENT_TYPE_LOSS_SIMPLE,
                'data': EXPERIMENT_TYPE_USE_STEROID,
                'init_lr': 0.0005,
                'lee': 0,
                'scaling': True,
                'contact': 13,
                'batch_size': 12,
                'epochs': 100,
                'train_transform': Compose(
                    [
                        OneOf([RandomCrop(512, 512, p=0.7),
                               RandomResizedCrop(512, 512, scale=(0.6, 1.0), p=0.3)],
                              p=1.0),

                        RandomBrightnessContrast(p=0.3),
                        Blur(p=0.3),
                        Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225]),
                        ToTensor(),
                    ]),
                'valid_transform': Compose(
                    [CenterCrop(512, 512, p=1.0),
                     Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225]),
                     ToTensor()
                     ]),
                'infer_transform': Compose(
                    [
                        Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225]),
                        ToTensor()
                    ])
            }
        }
    }
    , # Experiment no2 with inceptionresnetv2
    {
        'exp_id': 'Unet_inceptionresnetv2_v1',
        'weight': 3.0,
        'nfolds': 4,
        'rgb': {
            'id': 'Unet_inceptionresnetv2_v1_rgb_full',
            'type': EXPERIMENT_TYPE_RGB,
            'mask': MASK_TYPE_BUILDING,

            'model': {
                'model_type': smp.Unet,
                'encoder': 'inceptionresnetv2',
                'activation': 'sigmoid',
                'output_class': 1,
                'size': 512
            },
            'train': {
                'mean': [0.5, 0.5, 0.5],
                'std': [0.5, 0.5, 0.5],
                'loss': EXPERIMENT_TYPE_LOSS_SIMPLE,
                'data': EXPERIMENT_TYPE_USE_ORIGINAL,
                'contact': 13,
                'init_lr': 0.0005,
                'batch_size': 10,
                'epochs': 60,
                'train_transform': Compose(
                    [RandomCrop(512, 512, p=1.0),
                     HorizontalFlip(p=.5),
                     VerticalFlip(p=.5),
                     RandomRotate90(p=0.5),
                     RandomBrightnessContrast(p=0.3),
                     Blur(p=0.3),
                     Normalize(mean=[0.5, 0.5, 0.5],
                               std=[0.5, 0.5, 0.5]),
                     ToTensor(),
                     ]),
                'valid_transform': Compose(
                    [CenterCrop(512, 512, p=1.0),
                     Normalize(mean=[0.5, 0.5, 0.5],
                               std=[0.5, 0.5, 0.5]),
                     ToTensor()
                     ]),
                'infer_transform': Compose(
                    [
                        Normalize(mean=[0.5, 0.5, 0.5],
                                  std=[0.5, 0.5, 0.5]),
                        ToTensor()
                    ])
            }
        },
        'sar': {
            'id': 'Unet_inceptionresnetv2_v1_sar',
            'mask': MASK_TYPE_BUILDING,
            'type': EXPERIMENT_TYPE_SAR,
            'model': {
                'model_type': smp.Unet,
                'encoder': 'inceptionresnetv2',
                'activation': 'sigmoid',
                'output_class': 1,
                'size': 512
            },
            'train': {
                'mean': [0.5, 0.5, 0.5],
                'std': [0.5, 0.5, 0.5],
                'loss': EXPERIMENT_TYPE_LOSS_SIMPLE,
                'data': EXPERIMENT_TYPE_USE_STEROID,
                'init_lr': 0.0005,
                'lee': 0,
                'scaling': True,
                'contact': 13,
                'epochs': 2,
                'batch_size': 9,
                'train_transform': Compose(
                    [
                        OneOf([RandomCrop(512, 512, p=0.7),
                               RandomResizedCrop(512, 512, scale=(0.6, 1.0), p=0.3)],
                              p=1.0),

                        RandomBrightnessContrast(p=0.3),
                        Blur(p=0.3),
                        Normalize(mean=[0.5, 0.5, 0.5],
                                  std=[0.5, 0.5, 0.5]),
                        ToTensor(),
                    ]),
                'valid_transform': Compose(
                    [CenterCrop(512, 512, p=1.0),
                     Normalize(mean=[0.5, 0.5, 0.5],
                               std=[0.5, 0.5, 0.55]),
                     ToTensor()
                     ]),
                'infer_transform': Compose(
                    [
                        Normalize(mean=[0.5, 0.5, 0.5],
                                  std=[0.5, 0.5, 0.5]),
                        ToTensor()
                    ])
            }
        }
    },
]


def infer_one(model, mask, tile_size=(512, 512), tile_step=(256, 256), weight='mean'):
    image = mask.cpu().numpy()
    image = np.moveaxis(image, 0, -1)

    with torch.no_grad():
        tiler = ImageSlicer((900, 900), tile_size=tile_size, tile_step=tile_step, weight=weight)
        tiles = [np.moveaxis(tile, -1, 0) for tile in tiler.split(image)]
        merger = CudaTileMerger(tiler.target_shape, 1, tiler.weight)

        for tiles_batch, coords_batch in DataLoader(list(zip(tiles, tiler.crops)), batch_size=8, pin_memory=False):
            tiles_batch = tiles_batch.float().cuda()
            pred_batch = model(tiles_batch)
            tiles_batch.cpu().detach()
            merger.integrate_batch(pred_batch, coords_batch)
    merged_mask = np.moveaxis(to_numpy(merger.merge()), 0, -1)
    merged_mask = tiler.crop_to_orignal_size(merged_mask)

    m = merged_mask[..., 0].copy()
    return m

#
# Create models
def creat_rgb_experiment(df, exp, fold=None, base_path='/root/spacenet/train/AOI_11_Rotterdam/'):
    print('Creating RGB exp')
    print('type', exp['model']['model_type'])
    print('encoder', exp['model']['encoder'])
    print('activation', exp['model']['activation'])
    print('init_lr', exp['train']['init_lr'])
    print('fold', fold)
    print('train_trans', exp['train']['train_transform'])
    print('mask_type', exp['mask'])
    print('loss', exp['train']['loss'])
    print('data', exp['train']['data'])


    model, preprocessing_fn, optimizer, train_epoch, valid_epoch = create_model_optimizer(
        smp_model=exp['model']['model_type'],
        encoder=exp['model']['encoder'],
        activation=exp['model']['activation'],
        init_weights=None,
        init_lr=exp['train']['init_lr'],
        loss_type=exp['train']['loss'])

    train_loader, valid_loader, valid_loader_metric = create_model_RGB_generators(df, base_path, fold=fold,
                                                                                  bs=exp['train']['batch_size'],
                                                                                  transform_train=exp['train'][
                                                                                      'train_transform'],
                                                                                  transform_valid=exp['train'][
                                                                                      'valid_transform'],
                                                                                  transform_infer=exp['train'][
                                                                                      'infer_transform'])
    return model, preprocessing_fn, optimizer, train_epoch, valid_epoch, train_loader, valid_loader, valid_loader_metric


def creat_sar_experiment(df, exp, fold=0, init_weights=None, base_path='/root/spacenet/train/AOI_11_Rotterdam/'):
    print('Creating SAR exp')
    print('type', exp['model']['model_type'])
    print('encoder', exp['model']['encoder'])
    print('activation', exp['model']['activation'])
    print('init_lr', exp['train']['init_lr'])
    print('fold', fold)
    print('train_trans', exp['train']['train_transform'])
    print('mask_type', exp['mask'])

    model, preprocessing_fn, optimizer, train_epoch, valid_epoch = create_model_optimizer(
        smp_model=exp['model']['model_type'],
        encoder=exp['model']['encoder'],
        activation=exp['model']['activation'],
        init_weights=init_weights,
        init_lr=exp['train']['init_lr'],
        loss_type=exp['train']['loss'])

    train_loader, valid_loader, valid_loader_metric = create_model_SAR_generators(df, base_path, fold=fold,
                                                                                  bs=exp['train']['batch_size'],
                                                                                  transform_train=exp['train'][
                                                                                      'train_transform'],
                                                                                  transform_valid=exp['train'][
                                                                                      'valid_transform'],
                                                                                  transform_infer=exp['train'][
                                                                                      'infer_transform'],
                                                                                  train_on_steroids=(exp['train']['data']==EXPERIMENT_TYPE_USE_STEROID))
    return model, preprocessing_fn, optimizer, train_epoch, valid_epoch, train_loader, valid_loader, valid_loader_metric


#
# Create trainers
def train_sar(exp, outdir, truth_df, model, preprocessing_fn, optimizer, train_epoch, valid_epoch, train_loader,
              valid_loader, valid_loader_metric, fold):
    if fold is not None:
        exp_id = exp['id'] + '_' + str(fold)
    else:
        exp_id = exp['id']
    output_path = os.path.join(outdir, exp_id)
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.mkdir(output_path)

    csv_file = os.path.join(output_path, exp_id + '.csv')
    model_file = os.path.join(output_path, exp_id+ '.pth')

    print ('SAR path ', output_path)
    print ('SAR csv ', csv_file)
    print ('SAR model_file ', model_file)

    fp = open(csv_file, 'w')
    f1score = 0
    tmc = 0
    pmc = 0
    nepochs = exp['train']['epochs']
    for i in range(0, exp['train']['epochs']):

        print('\nEpoch: {}'.format(i))
        train_logs = train_epoch.run(train_loader)
        if fold is not None:
            valid_logs = valid_epoch.run(valid_loader)
        else:
            valid_logs = train_logs

        if i == 0:
            logs.print_log_headers(fp, train_logs, valid_logs)

        logs.print_log_lines(fp, i, train_logs, valid_logs, f1score, tmc, pmc)
        logs.print_log_lines_console(fp, i, train_logs, valid_logs, f1score, tmc, pmc)

        # do something (save model, change lr, etc.)

        torch.save(model.state_dict(), model_file)
        print('Model saved!')

        if i == int(nepochs/5):
            optimizer.param_groups[0]['lr'] = 1e-4
            print('Decrease decoder learning rate to 1e-4!')
        if i == int(2*nepochs/5):
            optimizer.param_groups[0]['lr'] = 5e-5
            print('Decrease decoder learning rate to 1e-5!')
        #if i == int(3*nepochs/5):
        #    optimizer.param_groups[0]['lr'] = 5e-6
        #    print('Decrease decoder learning rate to 5e-6!')
        if i == int(4*nepochs/5):
            optimizer.param_groups[0]['lr'] = 1e-5
            print('Decrease decoder learning rate to 1e-6!')
        fp.flush()


    # model.eval()
    # f1score, tmc, pmc = space_metric(valid_loader_metric, model, truth_df,
    #                                  tile_size=(exp['model']['size'], exp['model']['size']),
    #                                  tile_step=(224, 224),
    #                                  weight='mean')
    # model.train()
    # print_log_lines(fp, i, train_logs, valid_logs, f1score, tmc, pmc)
    # print_log_lines_console(fp, i, train_logs, valid_logs, f1score, tmc, pmc)
    fp.close()
    return


def train_rgb(exp, outdir, truth_df, model, preprocessing_fn, optimizer, train_epoch, valid_epoch, train_loader,
              valid_loader, valid_loader_metric, fold):


    if fold is not None:
        exp_id = exp['id'] + '_' + str(fold)
    else:
        exp_id = exp['id']

    output_path = os.path.join(outdir, exp_id)
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.mkdir(output_path)

    csv_file = os.path.join(output_path, exp_id + '.csv')
    model_file = os.path.join(output_path, exp_id + '.pth')
    print ('RGB path ', output_path)
    print ('RGB csv ', csv_file)
    print ('RGB model_file ', model_file)
    fp = open(csv_file, 'w')
    nepochs = exp['train']['epochs']
    for i in range(0, exp['train']['epochs']):

        print('\nEpoch: {}'.format(i))
        train_logs = train_epoch.run(train_loader)
        if fold is not None:
            valid_logs = valid_epoch.run(valid_loader)
        else:
            valid_logs = train_logs

        if i == 0:
            logs.print_log_headers(fp, train_logs, valid_logs)

        logs.print_log_lines(fp, i, train_logs, valid_logs, 0, 0, 0)
        # do something (save model, change lr, etc.)

        torch.save(model.state_dict(), model_file)
        print('Model saved!')


        if i == int(2*nepochs/5):
            optimizer.param_groups[0]['lr'] = 1e-5
            print('Decrease decoder learning rate to 1e-6!')
        if i == int(4*nepochs/5):
            optimizer.param_groups[0]['lr'] = 1e-6
            print('Decrease decoder learning rate to 1e-6!')

        fp.flush()

    # model.eval()
    # f1score, tmc, pmc = space_metric(valid_loader_metric, model, truth_df,
    #                                  tile_size=(exp['model']['size'], exp['model']['size']),
    #                                  tile_step=(224, 224),
    #                                  weight='mean')
    # model.train()
    # print_log_lines(fp, i, train_logs, valid_logs, f1score, tmc, pmc)

    fp.close()
    return model_file

#
# Create optimizers
def create_model_optimizer(smp_model=smp.Unet,
                           encoder='se_resnext50_32x4d',
                           activation='sigmoid',
                           init_weights=None,
                           init_lr=0.0001,
                           loss_type=EXPERIMENT_TYPE_LOSS_SIMPLE):
    print(init_weights)
    ENCODER_WEIGHTS = 'imagenet'
    DEVICE = 'cuda'

    # 0. Model
    model = smp_model(
        encoder_name=encoder,
        encoder_weights=ENCODER_WEIGHTS,
        classes=1,
        activation=activation,
    )

    preprocessing_fn = smp.encoders.get_preprocessing_fn(encoder, ENCODER_WEIGHTS)
    if loss_type == EXPERIMENT_TYPE_LOSS_SIMPLE:
        bce_dice_loss = loss.BCEDiceLoss(bce_w=0.5, dice_w=0.5)
    else:
        bce_dice_loss = loss.BCEDiceLossWeighted(bce_w=0.6, dice_w=0.4)

    metrics = [
        smp.utils.metrics.IoU(threshold=0.5)
    ]
    optimizer = torch.optim.Adam([
        dict(params=model.parameters(), lr=init_lr),
    ])

    if init_weights is not None:
        model.load_state_dict(torch.load(init_weights))
        print('loading weights from ', init_weights)

    if loss_type == EXPERIMENT_TYPE_LOSS_SIMPLE:
        train_epoch = epoch.TrainEpoch(
            model,
            loss=bce_dice_loss,
            metrics=metrics,
            optimizer=optimizer,
            device=DEVICE,
            verbose=True,
        )

        valid_epoch = epoch.ValidEpoch(
            model,
            loss=bce_dice_loss,
            metrics=metrics,
            device=DEVICE,
            verbose=True
        )
    else:
        train_epoch = epoch.WeightedTrainEpoch(
            model,
            loss=bce_dice_loss,
            metrics=metrics,
            optimizer=optimizer,
            device=DEVICE,
            verbose=True,
        )

        valid_epoch = epoch.WeightedValidEpoch(
            model,
            loss=bce_dice_loss,
            metrics=metrics,
            device=DEVICE,
            verbose=True
        )
    return model, preprocessing_fn, optimizer, train_epoch, valid_epoch

#
# Create generators
def create_model_RGB_generators(df, base_path,bs=16, fold=None,
                                transform_train=None,
                                transform_valid=None,
                                transform_infer=None):
    if fold == None:
        train_dataset = datagen.SpaceNetRGB(df, base_path=base_path, transformers=transform_train,
                                    orientation=True, return_labels=False)

        valid_dataset = datagen.SpaceNetRGB(df.copy(), base_path=base_path, transformers=transform_valid,
                                    orientation=True, return_labels=False)

        valid_dataset_metric = datagen.SpaceNetRGB(df.copy(), base_path=base_path, transformers=transform_infer,
                                           orientation=True, return_labels=True, return_orientation=True)
    else:

        train_dataset = datagen.SpaceNetRGB(df[df.split != fold].copy(), base_path=base_path, transformers=transform_train,
                                    orientation=True, return_labels=False)

        valid_dataset = datagen.SpaceNetRGB(df[df.split == fold].copy(), base_path=base_path, transformers=transform_valid,
                                    orientation=True, return_labels=False)

        valid_dataset_metric = datagen.SpaceNetRGB(df[df.split == fold].copy(), base_path=base_path, transformers=transform_infer,
                                           orientation=True, return_labels=True, return_orientation=True)

    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=bs, shuffle=False, num_workers=4)
    valid_loader_metric = DataLoader(valid_dataset_metric, batch_size=bs, shuffle=False, num_workers=4)

    return train_loader, valid_loader, valid_loader_metric


def create_model_SAR_generators(df, base_path, bs=16, scale_max=True, lee=0, nsize=13, train_on_steroids=True, fold=None,
                                transform_train=None, transform_valid=None, transform_infer=None, return_contact=False):
    print('SAR genergator --- ')
    print('bs', bs)
    print( 'scale_max', scale_max)
    print('lee', lee)
    print('nsize', nsize)
    print('train on steroids', train_on_steroids) # inverse of valida!!!
    print('fold', fold)
    print('return_contact', return_contact)

    if fold == None:
        train_dataset = datagen.SpaceNetSAR2RGBSteroids(df.copy(), base_path=base_path, transformers=transform_train,
                                                orientation=True, return_labels=False, scale_max=scale_max, lee=lee,
                                                nsize=nsize, valida=(not train_on_steroids), return_contact=return_contact)

        valid_dataset = datagen.SpaceNetSAR2RGBSteroids(df.copy(), base_path=base_path,transformers=transform_valid,
                                                orientation=True, return_labels=False, scale_max=scale_max, lee=lee,
                                                nsize=nsize, valida=True, return_contact=return_contact)

        valid_dataset_metric = datagen.SpaceNetSAR2RGBSteroids(df.copy(), base_path=base_path, transformers=transform_infer,
                                                       orientation=True, return_labels=True, return_orientation=True,
                                                       scale_max=scale_max, lee=lee,  nsize=nsize, valida=True,
                                                       return_contact=return_contact)
    else:

        train_dataset = datagen.SpaceNetSAR2RGBSteroids(df[df.split != fold].copy(), base_path=base_path, transformers=transform_train,
                                                orientation=True, return_labels=False, scale_max=scale_max, lee=lee,
                                                nsize=nsize, valida=train_on_steroids, return_contact=return_contact)

        valid_dataset = datagen.SpaceNetSAR2RGBSteroids(df[df.split == fold].copy(), base_path=base_path, transformers=transform_valid,
                                                orientation=True, return_labels=False, scale_max=scale_max, lee=lee,
                                                nsize=nsize, valida=True, return_contact=return_contact)

        valid_dataset_metric = datagen.SpaceNetSAR2RGBSteroids(df[df.split == fold].copy(), base_path=base_path, transformers=transform_infer,
                                                       orientation=True, return_labels=True, return_orientation=True,
                                                       scale_max=scale_max,lee=lee,  nsize=nsize, valida=True,
                                                       return_contact=return_contact)

    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=6)
    valid_loader = DataLoader(valid_dataset, batch_size=bs, shuffle=False, num_workers=6)
    valid_loader_metric = DataLoader(valid_dataset_metric, batch_size=bs, shuffle=False, num_workers=4)

    return train_loader, valid_loader, valid_loader_metric
