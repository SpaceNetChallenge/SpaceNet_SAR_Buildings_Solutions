import json

from osgeo import osr, gdal
geofeatures = []
feature_collection = {
    "type": "FeatureCollection",
    "features": geofeatures
}

import os
# get the existing coordinate system
data = []
for f in list((os.listdir('/mnt/sota/datasets/spacenet/train/AOI_11_Rotterdam/PS-RGB/'))):
    ds = gdal.Open('/mnt/sota/datasets/spacenet/train/AOI_11_Rotterdam/PS-RGB/' + f)
    print(f)
    old_cs= osr.SpatialReference()
    old_cs.ImportFromWkt(ds.GetProjectionRef())

    # create the new coordinate system
    wgs84_wkt = """
    GEOGCS["WGS 84",
        DATUM["WGS_1984",
            SPHEROID["WGS 84",6378137,298.257223563,
                AUTHORITY["EPSG","7030"]],
            AUTHORITY["EPSG","6326"]],
        PRIMEM["Greenwich",0,
            AUTHORITY["EPSG","8901"]],
        UNIT["degree",0.01745329251994328,
            AUTHORITY["EPSG","9122"]],
        AUTHORITY["EPSG","4326"]]"""
    new_cs = osr.SpatialReference()
    new_cs .ImportFromWkt(wgs84_wkt)

    # create a transform object to convert between coordinate systems
    transform = osr.CoordinateTransformation(old_cs,new_cs)

    #get the point to transform, pixel (0,0) in this case
    width = ds.RasterXSize
    height = ds.RasterYSize
    gt = ds.GetGeoTransform()


    minx = gt[0]
    maxy = gt[3]
    maxx = minx + gt[1] * ds.RasterXSize
    miny = maxy + gt[5] * ds.RasterYSize

    #get the coordinates in lat long
    lon1, lat1, _ = transform.TransformPoint(minx, miny)
    lon2, lat2, _ = transform.TransformPoint(maxx, maxy)
    data.append(["_".join(f.split("_")[-4:])[:-4], lon1, lat1, lon2, lat2])

    building = {
        "type": "Feature",
        "properties": {},
        "geometry": {
            "type": "Polygon",
            "coordinates": [[[lon1, lat1], [lon2, lat1], [lon2, lat2], [lon1, lat2],[lon1, lat1]]]
        }
    }
    geofeatures.append(building)
import pandas as pd
pd.DataFrame(data, columns=["id", "xmin", "ymin", "xmax", "ymax"]).to_csv("coordinates.csv", index=False)
with open("tiles.json", "w") as f:
    json.dump(geofeatures, f)