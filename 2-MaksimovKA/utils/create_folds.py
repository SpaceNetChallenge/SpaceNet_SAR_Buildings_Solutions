import os
import fire
import pandas as pd
from sklearn.model_selection import KFold


def create_folds(images_path='/data/SN6_buildings/train/AOI_11_Rotterdam/',
                 seed=769,
                 n_folds=8,
                 out_file='/wdata/folds.csv'):

    files = sorted(os.listdir(os.path.join(images_path, 'geojson_buildings')))
    files = ['_'.join(el.split('.')[0].split('_')[6:]) for el in files]

    target_df = {'image_name': [], 'fold_number': []}
    target_df['image_name'] += files
    target_df['fold_number'] += [-1 for el in files]
    
    target_df = pd.DataFrame(target_df)
    kf = KFold(n_splits=n_folds, random_state=seed, shuffle=True)
    for i, (train_index, evaluate_index) in enumerate(kf.split(target_df.index.values)):
        target_df['fold_number'].iloc[evaluate_index] = i + 1
        target_df = target_df[['image_name', 'fold_number']] 
    
    target_df.to_csv(out_file, index=False)


if __name__ == '__main__':
    fire.Fire(create_folds)
