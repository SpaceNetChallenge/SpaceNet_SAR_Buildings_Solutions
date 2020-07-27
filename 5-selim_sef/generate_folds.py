import argparse
import json

import pandas as pd
import numpy as np
from numpy.random.mtrand import RandomState
from shapely.geometry import Polygon
from shapely.ops import cascaded_union
from sklearn.model_selection import KFold

box_lon1, box_lat1 =  4.385218620300293, 51.85078428900754
box_lon2, box_lat2 = 4.404788017272949, 51.86644856213264

border_box_lon1, border_box_lat1 = 4.385776519775391, 51.850837307588726
border_box_lon2, border_box_lat2 = 4.407234191894531, 51.864938032294326
def iou(poly_true: Polygon, poly_pred: Polygon):
    int_area = poly_pred.intersection(poly_true).area
    polygons = [poly_pred, poly_true]
    u = cascaded_union(polygons)
    return float(int_area / u.area)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', default="coordinates.csv")

    args = parser.parse_args()
    df = pd.read_csv(args.csv)
    tile_coords = df.values
    data = []
    fold_tiles = []
    for i in range(2):
        fold_tiles.append({
            "type": "FeatureCollection",
            "features": []
        })
    train_polygons = []
    for row in tile_coords:
        fold = -1
        id, lon1, lat1, lon2, lat2 = row
        is_on_border = False
        lon = lon1 + (lon2 - lon1) / 2
        lat = lat1 + (lat2 - lat1) / 2
        if box_lon1 < lon < box_lon2  and box_lat1 < lat < box_lat2:
            fold = 0
        else:
            fold = 1
        if fold == 1:
            data.append([id, fold, is_on_border])
            coords = [[[lon1, lat1], [lon2, lat1], [lon2, lat2], [lon1, lat2], [lon1, lat1]]]
            train_polygons.append(Polygon(coords[0]))
            fold_tiles[fold]["features"].append(
                {
                    "type": "Feature",
                    "properties": {},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": coords
                    }
                }
            )
    for row in tile_coords:
        fold = -1
        id, lon1, lat1, lon2, lat2 = row
        is_on_border = False
        lon = lon1 + (lon2 - lon1) / 2
        lat = lat1 + (lat2 - lat1) / 2
        if box_lon1 < lon < box_lon2  and box_lat1 < lat < box_lat2:
            fold = 0
        else:
            fold = 1
        if fold == 0:
            coords = [[[lon1, lat1], [lon2, lat1], [lon2, lat2], [lon1, lat2], [lon1, lat1]]]
            polygon = Polygon(coords[0])
            for tp in train_polygons:
                if tp.intersects(polygon) and iou(tp, polygon)> 0.02:
                    is_on_border = True
                    break
            data.append([id, fold, is_on_border])

            fold_tiles[fold]["features"].append(
                {
                    "type": "Feature",
                    "properties": {},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": coords
                    }
                }
            )
    df = pd.DataFrame(data, columns=["id", "fold", "onborder"])
    df.to_csv("folds.csv", index=False)
    for i in range(2):
        with open("fold{}.json".format(i), "w") as f:
            json.dump(fold_tiles[i], f)
    print(df["fold"].value_counts())