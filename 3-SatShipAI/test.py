import os
import sys
import numpy as np
import pandas as pd
import geopandas as gpd
import argparse
import torch
import tqdm
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader, Dataset
import cv2
from shapely.wkt import loads as wkt_loads
import shapely.wkt
import rasterio
import shapely
from rasterio import features
import shapely.geometry
import shapely.affinity
from scipy import ndimage
import shutil
import gc

sys.path.append('.')
import solaris as sol
from albumentations.pytorch.transforms import ToTensor
from albumentations import (
    Compose,
    Normalize
)


def mask2box(mask):
    y1, y2, x1, x2 = np.where(mask == 1)[0].min(), np.where(mask == 1)[0].max(), np.where(mask == 1)[1].min(), \
                     np.where(mask == 1)[1].max()
    return y1, y2, x1, x2


def mask2box_xminyminxmaxymax(mask):
    y1, y2, x1, x2 = np.where(mask == 1)[0].min(), np.where(mask == 1)[0].max(), np.where(mask == 1)[1].min(), \
                     np.where(mask == 1)[1].max()
    return x1, y1, x2, y2


def colormask2boxes(mask):
    """
    Args:
        mask: [height,width], mask values, integers 0-255, 0=background
    Returns:
        list of bboxes (bbox is a list of 4 numbers, [xmin, ymin, xmax, ymax])
    """
    boxes = []
    if mask.sum() > 0:
        #         for i in range(1,len(np.unique(mask))):
        for i in [x for x in np.unique(mask) if x not in [0]]:
            x1y1x2y2 = mask2box_xminyminxmaxymax(mask == i)
            boxes.append([x1y1x2y2[0], x1y1x2y2[1], x1y1x2y2[2], x1y1x2y2[3]])
    return boxes


sigmoid = lambda x: 1 / (1 + np.exp(-x))


# def denormalize(x_batch):
#     #x_batch of shape batch_size,channels,height,width
#     x_batch2=x_batch.numpy().copy()
#     mean=[0.485, 0.456, 0.406]
#     std=[0.229, 0.224, 0.225]
#     for i in range(3):
#         x_batch2[:,i,...] = x_batch2[:,i,...]*std[i] + mean[i]
#     return (np.round(x_batch2*255)).astype('uint8')

def multimask2mask3d(multimask):
    num_buildings = len(np.unique(multimask))
    if num_buildings > 1:
        mask3d = np.zeros((multimask.shape[0], multimask.shape[1], num_buildings - 1))
        for i in range(1, num_buildings):
            mask3d[..., i - 1][multimask[..., 0] == i] = 1
    else:
        mask3d = np.zeros((multimask.shape[0], multimask.shape[1], 1))
    return (mask3d)


def multimask2mask3d_v2(multimask):
    num_buildings = len(np.unique(multimask))
    if multimask.sum() > 0:
        mask3d = np.zeros((multimask.shape[0], multimask.shape[1], num_buildings - 1))
        #         for i in range(1,num_buildings):
        for i in [x for x in np.unique(multimask) if x not in [0]]:
            mask3d[..., i - 1][multimask[..., 0] == i] = 1
    else:
        mask3d = np.zeros((multimask.shape[0], multimask.shape[1], 1))
    return (mask3d)


def multimask2mask3d_v3(multimask):
    num_buildings = len(np.unique(multimask))
    if multimask.sum() > 0:
        mask3d = np.zeros((multimask.shape[0], multimask.shape[1], num_buildings - 1))
        #         for i in range(1,num_buildings):
        counter = 0
        for i in [x for x in np.unique(multimask) if x not in [0]]:
            mask3d[..., counter][multimask == i] = 1
            counter += 1
    else:
        mask3d = np.zeros((multimask.shape[0], multimask.shape[1], 1))
    return (mask3d.astype('uint8'))


def mask2buildings(mask):
    maskC = mask.copy()
    maskC_output = np.zeros_like(maskC)  # .astype('int32')
    contours, hierarchy = cv2.findContours(maskC, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for i in range(len(contours)):
        cnt = contours[i]
        maskC_output += (cv2.drawContours(maskC, [cnt], -1, 1, cv2.FILLED) > 127.5).astype('uint8')
    uns = np.unique(maskC_output).copy()
    for ii in range(len(uns)):
        maskC_output[maskC_output == uns[ii]] = ii

    return maskC_output


def masks2masknum_v2(masks):
    outmask = np.zeros(masks.shape[1:])
    add = masks.shape[0]
    for m in range(len(masks)):
        outmask += masks[m] * (m + 1 + add)
    un_masks = np.unique(outmask)
    for mm in range(len(un_masks)):
        outmask[outmask == un_masks[mm]] = mm
    return outmask  # .astype('uint8')


def masks2masknum(masks):
    outmask = np.zeros(masks.shape[1:])
    for m in range(len(masks)):
        outmask += masks[m] * (m + 1)
    return outmask


def mask_to_polygon(mask):
    all_polygons = []
    lens=[]
    for shape, value in features.shapes(mask.astype(np.int16), mask=(mask == 1), transform=rasterio.Affine(1.0, 0, 0, 0, 1.0, 0)):
#         print(value)
#         print(len(shape['coordinates'][0]))
        all_polygons.append(shapely.geometry.shape(shape))
        lens.append(len(shape['coordinates'][0]))
#     print(np.argmax(lens))
    all_polygons = shapely.geometry.Polygon(all_polygons[np.argmax(lens)])
    if not all_polygons.is_valid:
        all_polygons = all_polygons.buffer(0)
        # Sometimes buffer() converts a simple Multipolygon to just a Polygon,
        # need to keep it a Multi throughout
#         if all_polygons.type == 'Polygon':
#             all_polygons = shapely.geometry.MultiPolygon([all_polygons])
    return all_polygons

def _convert_coordinates_to_raster(coords, img_size, xymax):
    x_max, y_max = xymax
    height, width = img_size
    W1 = 1.0 * width * width / (width + 1)
    H1 = 1.0 * height * height / (height + 1)
    xf = W1 / x_max
    yf = H1 / y_max
    coords[:, 1] *= yf
    coords[:, 0] *= xf
    coords_int = np.round(coords).astype(np.int32)
    return coords_int


def _plot_mask_from_contours(raster_img_size, contours, class_value=1):
    img_mask = np.zeros(raster_img_size, np.int8)
    if contours is None:
        return img_mask
    perim_list, interior_list = contours
#     print(interior_list)
    cv2.fillPoly(img_mask, perim_list, class_value)
#     img_mask[np.array(list(proposalcsv.PolygonWKT_Pix.values[-1].exterior.coords)).astype(int)]=0
    cv2.fillPoly(img_mask, interior_list, 0)
    return img_mask

def _get_and_convert_contours(onepolygon, raster_img_size, xymax):
    perim_list = []
    interior_list = []
#     if onepolygon is None:
#         return None
#     for k in range(len(onepolygon)):
    poly = onepolygon
#     for ppp in poly.interiors:
#         print(ppp)
    perim = np.array(list(poly.exterior.coords))
    perim_c = _convert_coordinates_to_raster(perim, raster_img_size, xymax)
    perim_list.append(perim_c)
    for pi in poly.interiors:
        interior = np.array(list(pi.coords))
        interior_c = _convert_coordinates_to_raster(interior, raster_img_size, xymax)
        interior_list.append(interior_c)
    return perim_list, interior_list

def polygon2mask(polygon, width, height):
    xymax = (900,900)

    mask = np.zeros(( width, height))

#     for i, p in enumerate(polygons):
    i=0
    polygon_list = wkt_loads(str(polygon))
#     if polygon_list.length == 0:
#         continue
    contours = _get_and_convert_contours(polygon_list, (width, height), xymax)
    mask = _plot_mask_from_contours((width, height), contours, 1)
    return mask


def read_jpg(tile_id, data_folder='test_path'):
    image_name = tile_id
    image_path = data_folder + image_name
    img = cv2.imread(image_path)
    return img


def jpg_to_tensor(img, transforms, preprocessing=None):
    augmented = transforms(image=img)
    img = augmented['image']
    if preprocessing is not None:
        preprocessed = preprocessing(image=img, mask=np.zeros_like(img).astype('uint8'))
    img = preprocessed['image']
    return img


def patch_left_right_fixed(im1, mask1, im2, mask2):
    r = 0.5
    mid = max(2, int(im1.shape[0] * r))
    img_new = np.zeros_like(im1)
    img_new[:, :mid, :] = im1[:, -mid:, :]
    img_new[:, mid:, :] = im2[:, :-mid, :]
    mask_new = np.zeros_like(mask1)
    mask_new[:, :mid] = mask1[:, -mid:]
    mask_new[:, mid:] = mask2[:, :-mid]

    return img_new, mask_new


def patch_top_down_fixed(im1, mask1, im2, mask2):
    r = 0.5
    mid = max(2, int(im1.shape[0] * r))
    img_new = np.zeros_like(im1)
    img_new[:mid, :, :] = im1[-mid:, :, :]
    img_new[mid:, :, :] = im2[:-mid, :, :]
    mask_new = np.zeros_like(mask1)
    mask_new[:mid, :] = mask1[-mid:, :]
    mask_new[mid:, :] = mask2[:-mid, :]

    return img_new, mask_new


class BuildingsDatasetInferenceCombined(Dataset):
    def __init__(self, img_ids: np.array = None, combImages=None,
                 transforms=None,
                 preprocessing=None):
        self.combImages = combImages
        self.img_ids = img_ids
        self.transforms = transforms
        self.preprocessing = preprocessing

    def __getitem__(self, idx):
        img = self.combImages[idx]

        augmented = self.transforms(image=img, mask=np.zeros_like(img).astype('uint8'))
        img = augmented['image']
        mask = augmented['mask']
        if self.preprocessing:
            preprocessed = self.preprocessing(image=img, mask=np.zeros_like(img).astype('uint8'))
            img = preprocessed['image']
            mask = preprocessed['mask']

            return img, mask

    def __len__(self):
        return len(self.img_ids)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SpaceNet 6 Baseline Algorithm')
    parser.add_argument('--testdata',
                        help='BaseDir')
    parser.add_argument('--outputfile',
                        help='Output directory')


    args = parser.parse_args(sys.argv[1:])

    print ('torch', torch.__version__)
    print ('gpd', gpd.__version__)
    print ("solaris", sol.__version__)

    test_data_path = args.testdata
    output_file = args.outputfile
    spacenet_out_dir = os.path.join(os.path.curdir, 'data/')

    spacenet_test_sar_path = os.path.join(test_data_path , 'SAR-Intensity/')

    print ('Base dir        :', spacenet_test_sar_path)
    print ('Output dir      :', spacenet_out_dir)

    #
    # Copy orientation to output as well...
    orientation_file = os.path.join('./', 'SAR_orientations.txt')
    if os.path.exists(orientation_file):
        print('SAR_orientations.txt exists')
    else:
        print ('FATAL SAR_orientations.txt missing')
        exit(1)

    import datagen
    from nasiosdataprocess import createdir, write_test_images

    #
    # 0. Nasios pipeline data prep
    test_save_path = os.path.join(spacenet_out_dir, 'test_sar_productscale_orient/')

    spacenet_test_sar = os.listdir(spacenet_test_sar_path)
    spacenet_test_sar = np.sort(spacenet_test_sar)
    orient_df = pd.read_csv(orientation_file, header=None, sep=" ")
    orient_df.columns = ['date', 'orient']
    testtifsdates = [('_').join(x.split('-')[1][10:].split('_')[:2]) for x in spacenet_test_sar]
    mines = np.load('productminesAllBoth.npy')
    maxes = np.load('productmaxesAllBoth.npy')

    if not os.path.exists(test_save_path):
        createdir(test_save_path)
        write_test_images(orient_df, test_save_path, spacenet_test_sar_path, testtifsdates, mines, maxes)
        tmp = os.listdir(test_save_path)
        print('nasios test images created', len(tmp))
    else:
        tmp = os.listdir(test_save_path)
        print('nasios test images exist', len(tmp) )
        shutil.rmtree(test_save_path)
        createdir(test_save_path)
        write_test_images(orient_df, test_save_path, spacenet_test_sar_path, testtifsdates, mines, maxes)
        tmp = os.listdir(test_save_path)
        print('nasios test images created', len(tmp))

    #
    # 2. Test on experiments
    from experiments import infer_one, create_model_optimizer
    from experiments import experiments as exps_vog
    from nasios import experiments1 as exp_nas1
    from nasios import experiments2 as exp_nas2
    exp_nas = exp_nas1 + exp_nas2

    from nasios import BuildingsDatasetBorders, get_preprocessing, get_validation_augmentation

    test_ids = [x[:-4] for x in os.listdir(test_save_path)]
    test_tiles = ['_'.join(x.split('_')[-4:-1]) for x in test_ids]
    test_tiles_nums = [int(x.split('_')[-1]) for x in test_ids]
    sortorder = np.argsort(test_tiles_nums)
    test_ids = list((np.array(test_ids)[sortorder]))


    test_ids2 = []
    for untile in np.unique(test_tiles):
        test_images_part = [x for x in test_ids if untile in x]
        test_ids2.extend(test_images_part)
    test_ids = test_ids2[:]
    test_ids_jpg = [x + '.jpg' for x in test_ids]

    test_ids_vog = ['_'.join(f.split('_')[-4:]) for f in test_ids]
    pream = '_'.join(test_ids[0].split('_')[:-4]) + '_'
    print('pream', pream)

    test_tiles_nums_nums = []
    for untile in np.unique(test_tiles):
        num = 0
        test_images_part = [x for x in test_ids if untile in x]
        test_tiles_nums = np.array([int(x.split('_')[-1]) for x in test_images_part])
        test_tiles_nums2 = ((test_tiles_nums - test_tiles_nums[0]) / 2).astype('int')
        test_tiles_nums_nums.extend(list(test_tiles_nums2))

    #pd.DataFrame(test_ids).to_csv("test_ids.csv")
    #pd.DataFrame(test_tiles_nums).to_csv("test_tiles_nums.csv")
    #pd.DataFrame(test_tiles_nums_nums).to_csv("test_tiles_nums_nums.csv")

    #
    # Accumulate all preds here
    final_preds = np.zeros((len(test_ids), 900, 900), dtype='float32')
    final_w = 0
    final_preds_borders = np.zeros((len(test_ids), 900, 900), dtype='float32')
    final_w_borders = 0


    test_df = pd.DataFrame({'ImageId': test_ids_vog, 'FullImageId':test_ids_jpg})
    test_df['date'] = test_df.apply(lambda row: row.ImageId.split("_")[0] + "_" + row.ImageId.split("_")[1], axis=1)
    test_df['tile'] = test_df.apply(lambda row: row.ImageId.split("_")[-1], axis=1)

    orient_df = pd.read_csv(spacenet_out_dir + '/SAR_orientations.txt', header=None, sep=" ")
    orient_df.columns = ['date', 'orient']
    test_df = pd.merge(test_df, orient_df, on='date')

    # Create pairs
    date_id = np.sort(np.unique(test_df.date.values))
    len(date_id)
    df_grp = test_df.groupby('date')
    pairs_D = {}
    pairs_L = {}
    for dat in tqdm.tqdm(date_id):
        df = df_grp.get_group(dat)

        for im1, orient1, tile_id1 in zip(df.FullImageId, df.orient, df.tile):
            my_ud = []
            my_lf = []
            for im2, orient2, tile_id2 in zip(df.FullImageId, df.orient, df.tile):

                if orient1 == 1:
                    if (int(tile_id1) % 2 == 1) and (int(tile_id1) == int(tile_id2) - 1):
                        my_ud.append(im2)
                    if True and (int(tile_id1) == int(tile_id2) - 2):
                        # if  (int(tile_id1)%2 == 1) and (int(tile_id1) == int(tile_id2)-2):
                        my_lf.append(im2)
                elif orient1 == 0:
                    if (int(tile_id1) % 2 == 1) and (int(tile_id1) == int(tile_id2) - 1):
                        my_ud.append(im2)
                    if True and (int(tile_id1) == int(tile_id2) - 2):
                        # if (int(tile_id1)%2 == 1) and (int(tile_id1) == int(tile_id2)-2):
                        my_lf.append(im2)

            pairs_D[im1] = my_ud
            pairs_L[im1] = my_lf

    # Calculate pair stats
    ico = 0
    for key in pairs_D.keys():
        if len(pairs_D[key]) > 0:
            ico = ico + 1
    print('Pairs found ', ico / len(pairs_D.keys()), ico, len(pairs_D.keys()))
    ico = 0
    for key in pairs_L.keys():
        if len(pairs_L[key]) > 0:
            ico = ico + 1
    print('Pairs found ',ico / len(pairs_L.keys()), ico, len(pairs_L.keys()))
    #
    # Create map from tile_id to position
    DD = dict(zip(test_ids, range(len(test_ids))))

    #
    # Create new images from left-right half/half
    ico = 0
    for key in pairs_L.keys():
        if len(pairs_L[key]) > 0:
            ico += 1
    print('keys LR', ico)

    if ico > 0:
        combImages = np.zeros((ico, 900, 900, 3), dtype='uint8')
    else:
        combImages = np.zeros((1, 900, 900, 3), dtype='uint8')

    counter = 0
    orients = []
    testid1testid2 = []
    for k in tqdm.tqdm(pairs_L.keys()):
        right = pairs_L[k]
        if len(right) > 0:
            key_L = k
            key_R = right[0]
            date_id = key_L.split('_')[0] + '_' + key_L.split('_')[1]

            orient_L = orient_df.orient.loc[orient_df.date == ('_').join(key_L.split('_')[-4:][:2])].values[0]
            orient_R  = orient_df.orient.loc[orient_df.date == ('_').join(key_R.split('_')[-4:][:2])].values[0]
            sar_L = read_jpg(key_L, data_folder=test_save_path)
            sar_R = read_jpg(key_R, data_folder=test_save_path)

            assert(orient_L==orient_R)
            if orient_L == 0:
                sar_final, _ = patch_left_right_fixed(sar_L, sar_L, sar_R, sar_R)
            else:
                sar_final, _ = patch_left_right_fixed(sar_R, sar_R, sar_L, sar_L)
            combImages[counter] = sar_final
            orients.append(orient_L)
            testid1testid2.append(key_L[:-4] + '--' + key_R[:-4])
            counter += 1

    #
    # 70/30 weight of new inference on the half of the image
    combW = 0.3
    origW = 1 - combW

    #
    # Accumulate all seconday preds here
    if len(testid1testid2) > 0:
        print ('we have ', len(testid1testid2), 'cases')
        final_preds_2 = np.zeros((len(testid1testid2), 900, 900), dtype='float32')
        final_w_2 = 0
        final_preds_borders_2 = np.zeros((len(testid1testid2), 900, 900), dtype='float32')
        final_w_borders_2 = 0

    # ########################################
    # ### nasios
    # ########################################
    print("-------------- nas inference ---------------- ")

    firsttime = True
    total_w = 0

    for exp in exp_nas:
        out_dir = os.path.join(spacenet_out_dir, exp['exp_id'])
        for i in range(len(exp['sizes'])):
            sz = exp['sizes'][i]
            w = exp['weights'][i]

            model_file = out_dir + '/' + str(exp['exp_id']) + '_' + str(exp['sizes'][i]) + '_' + str(
                exp['mode'][i]) + '.pth'
            flag = os.path.exists(model_file)
            print(exp['exp_id'] + ' ' + str(sz) + ' ' + str(w), '-> ', model_file), flag

            if flag and w > 0:
                ACTIVATION = 'sigmoid'  # None#'sigmoid'
                ENCODER = exp['encoder']
                ENCODER_WEIGHTS = 'imagenet'
                DEVICE = 'cuda'

                preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

                model = smp.Unet(
                    encoder_name=ENCODER,
                    encoder_weights=ENCODER_WEIGHTS,
                    classes=2,
                    activation=ACTIVATION,
                    decoder_attention_type='scse'
                )

                model.cuda()
                num_workers = 4
                bs = 3
                test_dataset = BuildingsDatasetBorders(datatype='test', data_folder=test_save_path, img_ids=test_ids,
                                                       transforms=get_validation_augmentation(sz),
                                                       preprocessing=get_preprocessing(preprocessing_fn))
                test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=False, num_workers=num_workers)

                model.load_state_dict(torch.load(model_file))
                model.eval()

                test_preds = np.zeros((len(test_ids), 900, 900), dtype='float32')
                test_preds_borders = np.zeros((len(test_ids), 900, 900), dtype='float32')
                for i, (x_batch, y_batch) in enumerate(tqdm.tqdm(test_loader)):
                    preds = model(x_batch.cuda()).detach().cpu()
                    mask = preds[:, 0, ...]
                    borders = preds[:, 1, ...]
                    for j in range(len(preds)):
                        test_preds[i * bs + j, ...] = w * cv2.resize((mask[j, ...]).numpy().astype('float32'),
                                                                     (900, 900))
                        test_preds_borders[i * bs + j, ...] = w * cv2.resize(
                            (borders[j, ...]).numpy().astype('float32'), (900, 900))

                final_w = final_w + w
                final_w_borders = final_w_borders + w
                final_preds += test_preds
                final_preds_borders += test_preds_borders

                if len(testid1testid2) > 0:
                    test_dataset_2 = BuildingsDatasetInferenceCombined(combImages=combImages,
                                                                       img_ids=testid1testid2,
                                                                       transforms=get_validation_augmentation(sz),
                                                                       preprocessing=get_preprocessing(preprocessing_fn))
                    test_loader_2 = DataLoader(test_dataset_2, batch_size=bs, shuffle=False, num_workers=num_workers)
                    test_preds_2 = np.zeros((len(testid1testid2), 900, 900), dtype='float32')
                    test_preds_borders_2 = np.zeros((len(testid1testid2), 900, 900), dtype='float32')
                    for i, (x_batch, y_batch) in enumerate(tqdm.tqdm(test_loader_2)):
                        preds = model(x_batch.cuda()).detach().cpu()
                        mask = preds[:, 0, ...]
                        borders = preds[:, 1, ...]
                        for j in range(len(preds)):
                            test_preds_2[i * bs + j, ...] = w * cv2.resize((mask[j, ...]).numpy().astype('float32'), (900, 900))
                            test_preds_borders_2[i * bs + j, ...] = w * cv2.resize((borders[j, ...]).numpy().astype('float32'), (900, 900))

                    final_w_2 = final_w_2 + w
                    final_w_borders_2 = final_w_borders_2 + w
                    final_preds_2 += test_preds_2
                    final_preds_borders_2 += test_preds_borders_2

                    del test_preds_2
                    del test_preds_borders_2
                    del test_dataset_2
                    del test_loader_2
                    gc.collect()
            else:
                if w == 0:
                    print('skipping model: due to weight')
                else:
                    print('skipping model: file not found')

    # merge preds
    if len(testid1testid2) > 0:
        print('new cases applied', len(testid1testid2), len(final_preds_2), len(orients))
        for i in tqdm.tqdm(range(len(final_preds_2))):
            num1 = DD[testid1testid2[i].split('--')[0]]
            num2 = DD[testid1testid2[i].split('--')[1]]

            if orients[i] == 1:
                final_preds[num1, :, :450] = origW * final_preds[num1, :, :450] + combW * final_preds_2[i, :, 450:]
                final_preds_borders[num1, :, :450] = origW * final_preds_borders[num1, :, :450] + combW * final_preds_borders_2[
                                                                                                        i, :, 450:]

                final_preds[num2, :, 450:] = origW * final_preds[num2, :, 450:] + combW * final_preds_2[i, :, :450]
                final_preds_borders[num2, :, 450:] = origW * final_preds_borders[num2, :, 450:] + combW * final_preds_borders_2[
                                                                                                        i, :, :450]

            else:
                final_preds[num1, :, 450:] = origW * final_preds[num1, :, 450:] + combW * final_preds_2[i, :, :450]
                final_preds_borders[num1, :, 450:] = origW * final_preds_borders[num1, :, 450:] + combW * final_preds_borders_2[
                                                                                                        i, :, :450]

                final_preds[num2, :, :450] = origW * final_preds[num2, :, :450] + combW * final_preds_2[i, :, 450:]
                final_preds_borders[num2, :, :450] = origW * final_preds_borders[num2, :, :450] + combW * final_preds_borders_2[
                                                                                                        i, :, 450:]
        del final_preds_2
        del final_preds_borders_2
        gc.collect()
    ########################################
    ### voglis
    ########################################
    print("-------------- vog inference ---------------- ")


    for exp in exps_vog:
        print (exp['exp_id'], exp['sar']['train']['mean'], exp['sar']['train']['std'])

        transform_infer = Compose(
            [Normalize(mean=exp['sar']['train']['mean'],
                       std=exp['sar']['train']['std']),
             ToTensor()
             ])

        inference_dataset = datagen.SpaceNetSAR2RGBSteroidsInference(test_df, transformers=transform_infer, sar_base_path=spacenet_test_sar_path,
                                                             orientation=True, return_orientation=True,
                                                             return_labels=True, sar_preampl=pream,
                                                             scale_max=True, lee=0)
        inference_loader = DataLoader(inference_dataset, batch_size=exp['sar']['train']['batch_size'], shuffle=False, num_workers=6)
        ENCODER = exp['sar']['model']['encoder']
        models = []
        for f in range(exp['nfolds']):
            print('************************************')
            print('*********** ' + str(f) + '***************')
            print('************************************')
            model, preprocessing_fn, optimizer, train_epoch, valid_epoch = create_model_optimizer(
                smp_model=exp['sar']['model']['model_type'],
                encoder=exp['sar']['model']['encoder'],
                activation=exp['sar']['model']['activation'],
                init_weights=None,
                init_lr=exp['sar']['train']['init_lr'],
                loss_type=exp['sar']['train']['loss'])

            model_dir = exp['sar']['id']  + '_' + str(f) + '/' + exp['sar']['id'] + '_' + str(f) + '.pth'
            MODEL_FILE = spacenet_out_dir + "/" +  model_dir
            print (MODEL_FILE)
            flag = os.path.exists(MODEL_FILE)
            if flag:
                print('(2) weight exists')
            if os.path.exists(MODEL_FILE):
                print(MODEL_FILE)
                model.load_state_dict(torch.load(MODEL_FILE))
                model.eval();
                models.append(model)

        dl = inference_loader

        PREDS = []
        with torch.no_grad():
            for (x_batch, tile_batch, orient_batch) in tqdm.tqdm(dl):
                for i in range(len(x_batch)):
                    preds = np.zeros((900, 900))
                    for m in models:
                        pred = infer_one(m, x_batch[i, ...], tile_size=(512, 512), tile_step=(224, 224), weight='pyramid')
                        preds = preds + pred
                    preds = preds / len(models)

                    PREDS.append(preds.astype('float16'))
        PREDS = np.array(PREDS)

        #np.saved_compressed('PREDS_' + exp['exp_id'], a=PREDS)

        final_preds = final_preds + exp['weight'] * PREDS
        final_w = final_w + exp['weight']




    final_preds  = final_preds / final_w
    final_preds_borders = final_preds_borders / final_w_borders

    final_preds = final_preds - final_preds_borders

    print('Final weights: ', final_w)
    print('Final border weights: ', final_w_borders)


    firstfile = True
    tile_ids = []
    min_mask_size = 160
    ocounter = 0
    kernel = np.ones((5, 5), np.uint8)
    counter2 = 0

    for imn in tqdm.tqdm(test_ids):
        # for imn in tqdm.tqdm_notebook(os.listdir('/var/data/spacenet/detectron/test_sar_productscale_orient')):
        #     THRESH=0.5
        THRESH = 0.45 - test_tiles_nums_nums[counter2] / 32
        DIFFTHRESH = 0.15
        A = 0.4

        orientvar = orient_df.orient.loc[orient_df.date == ('_').join(imn.split('-')[1][10:-4].split('_')[:2])].values[0]
        tile_id = imn.split('-')[1][10:]
        #print (orientvar,  tile_id, ('_').join(imn.split('-')[1][10:-4].split('_')[:2]))

        imgl = ndimage.label((final_preds[counter2] > THRESH).astype('uint16'))[0]

        pred_masks = multimask2mask3d_v3(imgl)
        pred_masks = np.rollaxis(pred_masks, 2, 0)

        #     if len(pred_masks)>0:
        if pred_masks.sum() > 0:
            tempscores = pred_masks.sum((1, 2))  # .shape
            if len(tempscores[tempscores > min_mask_size]) > 0:
                pred_masks = pred_masks[tempscores > min_mask_size]

            else:
                if len(pred_masks.shape) == 2:
                    pred_masks = np.expand_dims(pred_masks, 0)


            if pred_masks.sum() > 0:
                for g in range(pred_masks.shape[0]):
                    pred_masks[g, ...] = cv2.dilate(pred_masks[g, ...], kernel, iterations=1)

            # remove masks of low probability
            keepmask = []
            for g in range(pred_masks.shape[0]):
                maskprob = pred_masks[g, ...] * final_preds[counter2]
                #         if maskprob.mean()>0.55:
                #             keepmask.append(g)
                if (maskprob > THRESH + DIFFTHRESH).sum() > A * (maskprob > THRESH).sum():
                    keepmask.append(g)
            keepmask = np.array(keepmask)
            if len(keepmask) > 0:
                pred_masks = pred_masks[keepmask, ...]

            sortpreds = np.argsort(pred_masks.sum((1, 2)))[::-1]
            pred_masks = pred_masks[sortpreds, ...]

            counter = 0
            for mi in range(pred_masks.shape[0]):

                m = pred_masks[mi, ...]
                if orientvar == 1:
                    m = np.fliplr(np.flipud(m))


                if m.sum() > 0:
                    vectordata = mask_to_polygon(m)

                    csvaddition = pd.DataFrame({'ImageId': tile_id,
                                                'BuildingId': 0,
                                                'PolygonWKT_Pix': vectordata,
                                                'Confidence': 1
                                                }, index=[ocounter])
                    # csvaddition.to_csv('/home/voglis/SpaceNet6/tmp_proposal.csv', index=False)

                    tile_ids.append(tile_id)
                    if firstfile:
                        proposalcsv = csvaddition
                        firstfile = False
                    else:
                        proposalcsv = proposalcsv.append(csvaddition)

                    counter += 1
                    ocounter += 1

                else:
                    csvaddition = pd.DataFrame({'ImageId': tile_id,
                                                'BuildingId': 0,
                                                'PolygonWKT_Pix': ['POLYGON EMPTY'],
                                                'Confidence': 1.
                                                }, index=[ocounter])

                    if firstfile:
                        proposalcsv = csvaddition
                        firstfile = False
                    else:
                        proposalcsv = proposalcsv.append(csvaddition)
                    counter += 1
                    ocounter += 1


        else:
            csvaddition = pd.DataFrame({'ImageId': tile_id,
                                        'BuildingId': 0,
                                        'PolygonWKT_Pix': ['POLYGON EMPTY'],
                                        'Confidence': 1.
                                        }, index=[ocounter])

            if firstfile:
                proposalcsv = csvaddition
                firstfile = False
            else:
                proposalcsv = proposalcsv.append(csvaddition)
            counter += 1
            ocounter += 1

        counter2 += 1

    proposalcsv.loc[:, ['ImageId', 'PolygonWKT_Pix', 'Confidence']].to_csv(
        output_file, index=False)
