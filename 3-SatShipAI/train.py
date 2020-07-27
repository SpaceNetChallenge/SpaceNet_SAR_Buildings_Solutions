import os
import sys
import shutil
import argparse
import numpy as np
import pandas as pd
import geopandas as gpd
import torch

sys.path.append('.')
import solaris as sol

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SpaceNet 6 Baseline Algorithm')
    parser.add_argument('--basedir',
                        help='BaseDir')


    args = parser.parse_args(sys.argv[1:])

    print ('torch', torch.__version__)
    print ('gpd', gpd.__version__)
    print ("solaris", sol.__version__)

    spacenet_base_path = args.basedir
    spacenet_out_dir = os.path.join(os.path.curdir, 'data/')

    spacenet_sar_path = os.path.join(spacenet_base_path , 'SAR-Intensity/')
    spacenet_rgb_path =  os.path.join(spacenet_base_path , 'PS-RGB/')
    spacenet_orientation_file =  os.path.join(spacenet_base_path , 'SummaryData/SAR_orientations.txt')
    spacenet_masks_file =  os.path.join(spacenet_base_path, 'SummaryData/SN6_Train_AOI_11_Rotterdam_Buildings.csv')

    spacenet_labels_path =  os.path.join(spacenet_base_path, 'geojson_buildings/')

    print ('Base dir        :', spacenet_base_path)
    print ('Base sar dir    :', spacenet_sar_path)
    print ('Base rgb sar    :', spacenet_rgb_path)
    print ('Orientation file:', spacenet_orientation_file)
    print ('Masks file      :', spacenet_masks_file)
    print ('Output dir      :', spacenet_out_dir)

    #
    # Copy orientation to output as well...
    shutil.copyfile(spacenet_orientation_file, spacenet_out_dir+'/SAR_orientations.txt')

    from datagen import prepare_df

    #
    # 0. Create preparatory DataFrame
    df = prepare_df(spacenet_masks_file, spacenet_orientation_file)
    print(df.head())
    truth_df = pd.read_csv(spacenet_masks_file)


    #
    # 2. Train on experiments
    from experiments import experiments, creat_rgb_experiment, train_rgb, creat_sar_experiment, train_sar

    for exps in experiments:
        print (exps['exp_id'])



        nfolds = exps['nfolds']
        if nfolds > 0:
            #
            # 1. Continuous folded scheme
            image_ids = np.sort(np.unique(df['date'].values))
            for im in image_ids:
                aDf = df.loc[df.date == im, :]
                aDf = aDf.sort_values(by='tile_id')

                for f in range(nfolds):
                    low = int(f * len(aDf) / nfolds)
                    hig = int((f + 1) * len(aDf) / nfolds)
                    aDf.iloc[low:hig].split = f

                le = len(aDf)
                df.loc[df.date == im, ['split']] = aDf.split
        if nfolds > 3:
            print(len(df[df.split == 0]), len(df[df.split == 1]), len(df[df.split == 2]), len(df[df.split == 3]))

        if exps['rgb'] is not None:
            model, _, optimizer, train_epoch, valid_epoch, train_loader, valid_loader, valid_loader_metric = creat_rgb_experiment(df,
                                                                                                                exps['rgb'],
                                                                                                                fold=None,
                                                                                                                base_path=spacenet_base_path)
            rgb_model_file = train_rgb(exps['rgb'],
                                       spacenet_out_dir,
                                       truth_df,
                                       model,
                                       None,
                                       optimizer,
                                       train_epoch,
                                       valid_epoch,
                                       train_loader,
                                       valid_loader,
                                       valid_loader_metric,
                                       None)
        else:
            rgb_model_file = None

        if nfolds > 0:
            for f in range(nfolds):
                model, _, optimizer, train_epoch, valid_epoch, train_loader, valid_loader, valid_loader_metric = creat_sar_experiment(df,
                                                                                                                    exps['sar'],
                                                                                                                    fold=f,
                                                                                                                    init_weights=rgb_model_file,
                                                                                                                    base_path=spacenet_base_path)
                sar_model_file = train_sar(exps['sar'],
                                           spacenet_out_dir,
                                           truth_df,
                                           model,
                                           None,
                                           optimizer,
                                           train_epoch,
                                           valid_epoch,
                                           train_loader,
                                           valid_loader,
                                           valid_loader_metric,
                                           f)
        else:
            model, _, optimizer, train_epoch, valid_epoch, train_loader, valid_loader, valid_loader_metric = creat_sar_experiment(df,
                                                                                                                exps['sar'],
                                                                                                                fold=None,
                                                                                                                init_weights=rgb_model_file
                                                                                                                )
            sar_model_file = train_sar(exps['sar'],
                                       spacenet_out_dir,
                                       truth_df,
                                       model,
                                       None,
                                       optimizer,
                                       train_epoch,
                                       valid_epoch,
                                       train_loader,
                                       valid_loader,
                                       valid_loader_metric,
                                       None)



