import os
import numpy as np
import rasterio.features
import shapely.ops
import shapely.wkt
import shapely.geometry
import pandas as pd
import cv2
from scipy import ndimage as ndi
from skimage.morphology import watershed
from tqdm import tqdm
from fire import Fire


def _remove_interiors(line):
    if "), (" in line:
        line_prefix = line.split('), (')[0]
        line_terminate = line.split('))",')[-1]
        line = (
            line_prefix +
            '))",' +
            line_terminate
        )
    return line


def my_watershed(what, mask1, mask2):
    markers = ndi.label(mask2, output=np.uint32)[0]
    labels = watershed(what, markers, mask=mask1, watershed_line=True)
    return labels


def wsh(mask_img, threshold, border_img, seeds, shift):
    img_copy = np.copy(mask_img)
    m = seeds * border_img

    img_copy[m <= threshold + shift] = 0
    img_copy[m > threshold + shift] = 1
    img_copy = img_copy.astype(np.bool)

    mask_img[mask_img <= threshold] = 0
    mask_img[mask_img > threshold] = 1
    mask_img = mask_img.astype(np.bool)
    labeled_array = my_watershed(mask_img, mask_img, img_copy)
    return labeled_array


def main(folds_predict='/wdata/folds_predicts',
         prob_trs=0.3,
         shift=0.4,
         min_lolygon_area=200,
         submit_path='/wdata/submits/solution.csv',
         save_path='/wdata/submit_predicts/'):

    folds = os.listdir(folds_predict)
    print(folds)
    files = sorted(os.listdir(os.path.join(folds_predict, folds[0])))[:]

    f = open(submit_path, 'w')
    f.write('ImageId,PolygonWKT_Pix,Confidence\n')
    for _file in tqdm(files):
        for fold_i, fold_name in enumerate(folds):
            file_path = os.path.join(folds_predict, fold_name, _file)
            data = cv2.imread(file_path) / 255.0
            if fold_i == 0:
                final_data = data[:, :, :]
            else:
                final_data += data[:, :, :]

        fid = '_'.join(_file.split('_')[-4:]).split('.')[0]
        pred_data = final_data / len(folds)

        file_save_path = os.path.join(save_path, _file)
        cv2.imwrite(file_save_path, (pred_data * 255).astype(np.uint8))

        labels = wsh(pred_data[:, :, 0],
                     prob_trs,
                     # ( 1 - pred_data[:, :, 2])*( 1 - pred_data[:, :, 1]),
                      1 - pred_data[:, :, 2],
                      pred_data[:, :, 0],
                     shift)
        label_numbers = list(np.unique(labels))
        all_dfs = []
        for label in label_numbers:
            if label != 0:
                submask = (labels == label).astype(np.uint8)
                if np.sum(submask) < min_lolygon_area:
                    continue
                shapes = rasterio.features.shapes(submask.astype(np.int16), submask > 0)

                mp = shapely.ops.cascaded_union(
                    shapely.geometry.MultiPolygon([
                        shapely.geometry.shape(shape)
                        for shape, value in shapes
                    ]))

                if isinstance(mp, shapely.geometry.Polygon):
                    df = pd.DataFrame({
                        'area_size': [mp.area],
                        'poly': [mp],
                    })
                else:
                    df = pd.DataFrame({
                        'area_size': [p.area for p in mp],
                        'poly': [p for p in mp],
                    })
                df = df[df.area_size > min_lolygon_area]
                df = df.reset_index(drop=True)
                # print(df)
                if len(df) > 0:
                    all_dfs.append(df.copy())
        if len(all_dfs) > 0:
            df_poly = pd.concat(all_dfs)
            df_poly = df_poly.sort_values(by='area_size', ascending=False)
            df_poly.loc[:, 'wkt'] = df_poly.poly.apply(lambda x: shapely.wkt.dumps(x, rounding_precision=0))
            df_poly.loc[:, 'area_ratio'] = df_poly.area_size / df_poly.area_size.max()
            for i, row in df_poly.iterrows():

                line = "{},\"{}\",{:.6f}\n".format(
                    fid,
                    row.wkt,
                    row.area_ratio)
                line = _remove_interiors(line)
                f.write(line)
        else:
            f.write("{},{},0\n".format(
                fid,
                "POLYGON EMPTY"))
    f.close()


if __name__ == '__main__':
    Fire(main)
