import os.path
import pandas as pd


def read_orientation_file(path):
    """Reads SAR_orientations file, which lists whether each strip was imaged
    from the north (denoted by 0) or from the south (denoted by 1).
    """
    rotationdf = pd.read_csv(
        path,
        sep=' ',
        index_col=0,
        names=['strip', 'direction'],
        header=None
    )
    rotationdf['direction'] = rotationdf['direction'].astype(int)
    return rotationdf


def lookup_orientation(tilepath, rotationdf):
    """Looks up the SAR_orientations value for a tile based on its filename
    """
    tilename = os.path.splitext(os.path.basename(tilepath))[0]
    stripname = '_'.join(tilename.split('_')[-4:-2])
    rotation = rotationdf.loc[stripname].squeeze()
    return rotation
