import shutil
import os
import fire
import gdal
import numpy as np
import cv2
from tqdm import tqdm
from osgeo import ogr
from scipy import ndimage as ndi
from skimage import measure
from skimage.morphology import dilation, square, watershed
from scipy.ndimage import binary_erosion
from multiprocessing.pool import Pool


def create_separation(labels):
    tmp = dilation(labels > 0, square(12))
    tmp2 = watershed(tmp, labels, mask=tmp, watershed_line=True) > 0
    tmp = tmp ^ tmp2
    tmp = dilation(tmp, square(3))

    props = measure.regionprops(labels)

    msk1 = np.zeros_like(labels, dtype='bool')

    for y0 in range(labels.shape[0]):
        for x0 in range(labels.shape[1]):
            if not tmp[y0, x0]:
                continue
            if labels[y0, x0] == 0:
                sz = 5
            else:
                sz = 7
                if props[labels[y0, x0] - 1].area < 300:
                    sz = 5
                elif props[labels[y0, x0] - 1].area < 2000:
                    sz = 6
            uniq = np.unique(labels[max(0, y0 - sz):min(labels.shape[0], y0 + sz + 1),
                             max(0, x0 - sz):min(labels.shape[1], x0 + sz + 1)])
            if len(uniq[uniq > 0]) > 1:
                msk1[y0, x0] = True
    return msk1


def mask_fro_id(param):
    _id, labels_path, rasters_path, result_path = param
    label_path = os.path.join(labels_path, 'SN6_Train_AOI_11_Rotterdam_Buildings_' + _id + '.geojson')
    raster_path = os.path.join(rasters_path, 'SN6_Train_AOI_11_Rotterdam_SAR-Intensity_' + _id + '.tif')
    tileHdl = gdal.Open(raster_path, gdal.GA_ReadOnly)
    tileGeoTransformationParams = tileHdl.GetGeoTransform()
    projection = tileHdl.GetProjection()
    width = tileHdl.RasterXSize
    height = tileHdl.RasterYSize

    tileHdl = None

    rasterDriver = gdal.GetDriverByName('MEM')

    final_mask = rasterDriver.Create('',
                                     height,
                                     width,
                                     1,
                                     gdal.GDT_Byte)

    final_mask.SetGeoTransform(tileGeoTransformationParams)
    final_mask.SetProjection(projection)
    tempTile = final_mask.GetRasterBand(1)
    tempTile.Fill(0)
    tempTile.SetNoDataValue(0)

    Polys_ds = ogr.Open(label_path)
    Polys = Polys_ds.GetLayer()
    gdal.RasterizeLayer(final_mask, [1], Polys, burn_values=[255])
    mask = final_mask.ReadAsArray()
    final_mask = None

    rasterDriver = gdal.GetDriverByName('GTiff')

    out_path = os.path.join(result_path, _id + '.tif')

    final_mask = rasterDriver.Create(out_path,
                                     height,
                                     width,
                                     3,
                                     gdal.GDT_Byte)

    final_mask.SetGeoTransform(tileGeoTransformationParams)
    final_mask.SetProjection(projection)
    tempTile = final_mask.GetRasterBand(1)
    tempTile.Fill(0)
    tempTile.SetNoDataValue(0)
    tempTile.WriteArray(mask[:, :])
    h, w = mask.shape
    all_contours = np.zeros((h, w), dtype=np.uint8)
    labels = ndi.label(mask, output=np.uint32)[0]
    ships_num = np.max(labels)

    if ships_num > 0:
        for i in range(1, ships_num + 1):
            ship_mask = np.zeros_like(labels, dtype='bool')
            ship_mask[labels == i] = 1

            area = np.sum(ship_mask)
            if area < 200:
                contour_size = 1
            elif area < 500:
                contour_size = 2
            else:
                contour_size = 3
            eroded = binary_erosion(ship_mask, iterations=contour_size)
            countour_mask = ship_mask ^ eroded
            all_contours += countour_mask
    all_contours = (all_contours > 0).astype(np.uint8) * 255
    tempTile = final_mask.GetRasterBand(2)
    tempTile.Fill(0)
    tempTile.SetNoDataValue(0)
    tempTile.WriteArray(all_contours[:, :])

    separation = create_separation(labels)
    separation = separation > 0
    separation = separation.astype(np.uint8)
    separation = separation * 255

    tempTile = final_mask.GetRasterBand(3)
    tempTile.Fill(0)
    tempTile.SetNoDataValue(0)
    tempTile.WriteArray(separation[:, :])

    final_mask = None


def create_masks(data_root_path='/data/SN6_buildings/train/AOI_11_Rotterdam/',
                 result_path='/wdata/train_masks/'):

    if os.path.exists(result_path):
        shutil.rmtree(result_path)
    os.mkdir(result_path)

    labels_path = os.path.join(data_root_path, 'geojson_buildings')
    rasters_path = os.path.join(data_root_path, 'SAR-Intensity')

    files = sorted(os.listdir(labels_path))
    ids = ['_'.join(el.split('.')[0] .split('_')[6:])  for el in files]
    params = [(_id, labels_path, rasters_path, result_path) for _id in ids]
    pool = Pool(8)
    # pool.map(mask_fro_id, params)
    for _ in tqdm(pool.imap_unordered(mask_fro_id, params), total=len(params)):
        pass


if __name__ == '__main__':
    fire.Fire(create_masks)
