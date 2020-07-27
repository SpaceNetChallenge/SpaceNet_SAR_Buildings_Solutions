import argparse
import json

import pandas as pd
import numpy as np
from numpy.random.mtrand import RandomState
from shapely.geometry import Polygon
from shapely.ops import cascaded_union
from sklearn.model_selection import KFold

center_lon, center_lat = 4.369125366210937, 51.888676633322035


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
    for i in range(4):
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
        #top left
        if lon < center_lon and lat > center_lat:
            fold = 0
            is_on_border = lat1 < center_lat or lon2 > center_lon
        #top right
        elif lon > center_lon and lat > center_lat:
            fold = 1
            is_on_border = lat1 < center_lat or lon1 < center_lon
        #bottom left
        elif lon < center_lon and lat < center_lat:
            fold = 2
            is_on_border = lat2 > center_lat or lon2 > center_lon
        #bottom right
        elif lon > center_lon and lat < center_lat:
            fold = 3
            is_on_border = lat2 > center_lat or lon1 < center_lon
        assert fold >= 0
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
    df = pd.DataFrame(data, columns=["id", "fold", "onborder"])
    df.to_csv("folds4.csv", index=False)
    for i in range(4):
        with open("fold{}.json".format(i), "w") as f:
            json.dump(fold_tiles[i], f)
    print(df["fold"].value_counts())
    print(df[df.onborder == False]["fold"].value_counts())
