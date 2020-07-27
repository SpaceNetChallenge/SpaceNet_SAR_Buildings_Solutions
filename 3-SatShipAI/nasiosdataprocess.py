import os
from torchvision import utils
from matplotlib import pyplot as plt
from tqdm import tqdm
import numpy as np
import pandas as pd
import gdal
import cv2
import shutil
import tqdm
import geopandas as gpd
import shapely


# input data

def prepare_train(spacenet_train_base_path='/var/data/spacenet/AOI_11_Rotterdam/',
               outpath_train='/var/data/spacenet/detectron/train_sar_productscale_orient/',
               outpath_masks_train='/home/gnas/SpaceNet6/masksandweights.npz',
               outpath_image_train='/home/gnas/SpaceNet6/allimages.npz'
               ):
    spacenet_sar_path = os.path.join(spacenet_train_base_path, 'SAR-Intensity/')
    spacenet_rgb_path = os.path.join(spacenet_train_base_path, 'PS-RGB/')
    spacenet_orientation_file = os.path.join(spacenet_train_base_path, 'SummaryData/SAR_orientations.txt')
    spacenet_masks_file = os.path.join(spacenet_train_base_path, 'SummaryData/SN6_Train_AOI_11_Rotterdam_Buildings.csv')

    spacenet_labels_path = os.path.join(spacenet_train_base_path, 'geojson_buildings/')

    # save rgb sar images paths
    TRAIN_SAVE_PATH = outpath_train
    MASKS_FROM_POLYGONS = outpath_masks_train  # .npz will be completed automatically
    IMAGES_NUMPY = outpath_image_train
    print ('TRAIN_SAVE_PATH', TRAIN_SAVE_PATH)
    print('MASKS_FROM_POLYGONS', MASKS_FROM_POLYGONS)
    print('IMAGES_NUMPY', IMAGES_NUMPY)

    if (not os.path.exists(IMAGES_NUMPY)) or (not os.path.exists(MASKS_FROM_POLYGONS)):
        # create and save masks on disk path - ONLY DATA WITH MAKS, IMAGES WITH NO BUILDING IS REJECTED
        print('create all')
        orient_df = pd.read_csv(spacenet_orientation_file, header=None, sep=" ")
        orient_df.columns = ['date', 'orient']

        orient_df, mines, maxes, traintifsdates = create_max_min_train(spacenet_sar_path, spacenet_orientation_file)
        write_train_images(orient_df, TRAIN_SAVE_PATH, spacenet_sar_path, traintifsdates, mines, maxes)
        #write_test_images(orient_df, TEST_SAVE_PATH, spacenet_test_sar, traintifsdates, mines, maxes)
        train_images_with_mask = create_train_masks(TRAIN_SAVE_PATH, spacenet_masks_file, orient_df, MASKS_FROM_POLYGONS)

        allimages = np.zeros((len(train_images_with_mask), 900, 900, 3), dtype='uint8')
        counter = 0
        for image_name in tqdm.tqdm(train_images_with_mask):
            image_path = TRAIN_SAVE_PATH + image_name + '.jpg'
            img = cv2.imread(image_path)

            allimages[counter, ...] = img
            counter += 1
        np.savez_compressed(IMAGES_NUMPY, a=allimages)
    else:
        # Remove images train if there is no mask
        train_images_with_mask = []
        image_names = np.array(os.listdir(TRAIN_SAVE_PATH))
        train_builds = gpd.read_file(spacenet_masks_file)
        for tile in image_names:
            if train_builds[train_builds.ImageId == tile[:-4]]['PolygonWKT_Pix'].values[0] != 'POLYGON EMPTY':
                train_images_with_mask.append(tile[:-4])

    return train_images_with_mask


def prepare_test(spacenet_test_base_path='/var/data/spacenet/AOI_11_Rotterdam/', output_dir='',
               outpath_test='/var/data/spacenet/detectron/train_sar_productscale_orient/',
               ):
    spacenet_sar_path = os.path.join(spacenet_test_base_path, 'SAR-Intensity/')
    spacenet_orientation_file = os.path.join(output_dir, 'SAR_orientations.txt')


    # save rgb sar images paths
    TEST_SAVE_PATH  = outpath_test

    # create and save masks on disk path - ONLY DATA WITH MAKS, IMAGES WITH NO BUILDING IS REJECTED
    orient_df = pd.read_csv(spacenet_orientation_file, header=None, sep=" ")
    orient_df.columns = ['date', 'orient']

    orient_df, mines, maxes, traintifsdates = create_max_min_train(spacenet_sar_path, spacenet_orientation_file)
    write_test_images(orient_df, TEST_SAVE_PATH, spacenet_sar_path, traintifsdates, mines, maxes)
    return

def write_train_images(orient_df, TRAIN_SAVE_PATH, spacenet_sar_path, traintifsdates, mines, maxes):

    createdir(TRAIN_SAVE_PATH)

    sortedtrainimagetiles = np.sort([x for x in os.listdir(spacenet_sar_path) if traintifsdates[0] in x])

    counter = 0
    for f in tqdm.tqdm(np.unique(traintifsdates)):
        #     print(f)
        sortedtrainimagetiles = np.sort([x for x in os.listdir(spacenet_sar_path) if f in x])
        #     sortedtestimagetiles = np.sort([x for x in os.listdir(spacenet_test_sar) if f in x])

        for file in sortedtrainimagetiles:
            #         print(file)
            sarpath = spacenet_sar_path + file
            tilefile = gdal.Open(sarpath)
            data = []
            banddata = tilefile.GetRasterBand(1).ReadAsArray()
            data.append(banddata)

            # Channel 2
            banddata1 = tilefile.GetRasterBand(2).ReadAsArray()
            banddata2 = tilefile.GetRasterBand(3).ReadAsArray()
            #     banddata = banddata2
            banddata = (banddata1 + banddata2) / 2
            data.append(banddata)

            # Channel 3
            banddata = tilefile.GetRasterBand(4).ReadAsArray()
            data.append(banddata)

            data = np.stack(data, axis=-1)

            #         data=(data*255).astype('uint8')

            data = 255 * (data - mines[counter]) / (maxes[counter] - mines[counter])

            data = np.clip(data, 0, 255)
            data = data.astype('uint8')

            if orient_df.orient.loc[orient_df.date == ('_').join(file.split('-')[1][10:].split('_')[:2])].values[0] == 1:
                data = np.fliplr(np.flipud(data))
            #         cv2.imwrite(TRAIN_SAVE_PATH + file[:-3]+'jpg', data,  [int(cv2.IMWRITE_JPEG_QUALITY), 100])
            cv2.imwrite(TRAIN_SAVE_PATH + file.split('-')[1][10:-3] + 'jpg', data, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        counter += 1

def write_test_images(orient_df, TEST_SAVE_PATH, spacenet_test_sar, testtifsdates, mines, maxes):
    counter = 0
    for f in tqdm.tqdm(np.unique(testtifsdates)):
        #     print(f)
        #     sortedtrainimagetiles = np.sort([x for x in os.listdir(spacenet_sar_path) if f in x])
        sortedtestimagetiles = np.sort([x for x in os.listdir(spacenet_test_sar) if f in x])

        for file in sortedtestimagetiles:
            sarpath = spacenet_test_sar + file
            tilefile = gdal.Open(sarpath)
            data = []
            banddata = tilefile.GetRasterBand(1).ReadAsArray()
            data.append(banddata)

            # Channel 2
            banddata1 = tilefile.GetRasterBand(2).ReadAsArray()
            banddata2 = tilefile.GetRasterBand(3).ReadAsArray()
            #     banddata = banddata2
            banddata = (banddata1 + banddata2) / 2
            data.append(banddata)

            # Channel 3
            banddata = tilefile.GetRasterBand(4).ReadAsArray()
            data.append(banddata)
            data = np.stack(data, axis=-1)

            #         data=(data*255).astype('uint8')
            data = 255 * (data - mines[counter]) / (maxes[counter] - mines[counter])


            data = np.clip(data, 0, 255)
            data = data.astype('uint8')
            if orient_df.orient.loc[orient_df.date == ('_').join(file.split('-')[1][10:].split('_')[:2])].values[0] == 1:
                data = np.fliplr(np.flipud(data))
            cv2.imwrite(TEST_SAVE_PATH + file[:-3] + 'jpg', data, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

        counter += 1

int_coords = lambda x: np.array(x).round().astype(np.int32)

def polyCoors2mask(poly_coords, shape=(900,900)):
    mask = np.zeros(shape).astype('uint16')
    cv2.fillPoly(mask, [poly_coords], 1)
    mask = mask.astype('uint16')
    return mask

def tilemask_border(tilename, train_builds, build_dist=5):
    kernel = np.ones((build_dist,build_dist),np.uint8)

    temp=train_builds[train_builds.ImageId==tilename]
    temp.reset_index(inplace=True)
    mask = np.zeros((900,900)).astype('uint16')
    borders = np.zeros((900,900)).astype('uint16')
    buildingsG80=0

    if temp.PolygonWKT_Pix.values[0]!='POLYGON EMPTY':
        for i in range(len(temp)):
            P = shapely.wkt.loads(temp.PolygonWKT_Pix.values[i])

            coords=int_coords(P.exterior.coords)
            oneB1 = polyCoors2mask(coords)
            oneB = cv2.dilate(oneB1,kernel)
    #         oneB=oneB-oneB1
            if oneB1.sum()>80:
                buildingsG80 +=1
            mask+=oneB1
            borders+=oneB-oneB1
        mask[mask>0]=1
        borders[ borders>0]=1
        mask = np.stack((mask,borders),-1)
    else:
        mask=np.zeros((900,900,2))
    mask=mask.astype('uint8')
    return mask,  len(temp), buildingsG80

def create_train_masks(TRAIN_SAVE_PATH, spacenet_masks_file, orient_df, MASKS_FROM_POLYGONS):
    image_names = np.array(os.listdir(TRAIN_SAVE_PATH))

    train_builds = gpd.read_file(spacenet_masks_file)
    # Remove images train if there is no mask
    train_images_with_mask = []
    for tile in image_names:
        if train_builds[train_builds.ImageId == tile[:-4]]['PolygonWKT_Pix'].values[0] != 'POLYGON EMPTY':
            train_images_with_mask.append(tile[:-4])


    allmasks = np.zeros((len(train_images_with_mask), 900, 900, 2)).astype('uint8')
    sample_weights = np.zeros(len(train_images_with_mask))
    counter = 0
    for image_name in tqdm.tqdm(train_images_with_mask):
        orientvar = orient_df.orient.loc[orient_df.date == ('_').join(image_name.split('_')[:2])].values[0]
        mask, buildings, buildingsG80 = tilemask_border(image_name, train_builds)
        if orientvar == 1:
            mask = np.fliplr(np.flipud(mask))
        allmasks[counter, ...] = mask
        sample_weights[counter] = buildingsG80
        counter += 1
    np.savez_compressed(MASKS_FROM_POLYGONS, a=allmasks, b=sample_weights)
    return train_images_with_mask

def create_max_min_train(spacenet_sar_path, spacenet_orientation_file):
    train_sars = os.listdir(spacenet_sar_path)
    # Read orientation per date
    orient_df = pd.read_csv(spacenet_orientation_file, header=None, sep=" ")
    orient_df.columns = ['date', 'orient']

    traintilespath = os.listdir(spacenet_sar_path)
    traintifsdates = [('_').join(x.split('-')[1][10:].split('_')[:2]) for x in os.listdir(spacenet_sar_path)]

    sortedtrainimagetiles = np.sort([x for x in os.listdir(spacenet_sar_path) if traintifsdates[0] in x])

    mines=np.load('./productminesAllBoth.npy')
    maxes=np.load('./productmaxesAllBoth.npy')



    return orient_df, mines, maxes, traintifsdates


def createdir(new_dir_path):
    print ('check if exists ', new_dir_path)
    if not os.path.exists(new_dir_path):
        os.mkdir(new_dir_path)
        print('1st time')
    else:
        print('Already exists ')
        shutil.rmtree(new_dir_path)
        os.mkdir(new_dir_path)

