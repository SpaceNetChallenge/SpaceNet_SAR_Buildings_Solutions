import argparse
import pandas as pd
import numpy as np
from numpy.random.mtrand import RandomState
from sklearn.model_selection import KFold

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv',
                        default="/mnt/sota/datasets/spacenet/train/AOI_11_Rotterdam/SummaryData/SN6_Train_AOI_11_Rotterdam_Buildings.csv")
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--seed", type=int, default=777)

    args = parser.parse_args()
    df = pd.read_csv(args.csv)
    tiles = np.unique(df["ImageId"].values)

    kfold = KFold(n_splits=args.folds, shuffle=True, random_state=RandomState(args.seed))
    data = []
    for i, (train_idx, test_idx) in  enumerate(kfold.split(tiles)):
        for idx in test_idx:
            data.append([tiles[idx], i])
    pd.DataFrame(data, columns=["id", "fold"]).to_csv("folds.csv", index=False)

