import os
import sys
import shutil
import argparse
import numpy as np
import torch
import geopandas as gpd

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

    import nasios

    #
    # train_images_with_mask = prepare_train(spacenet_train_base_path=spacenet_base_path,
    #            outpath_train=os.path.join(spacenet_out_dir, './train_sar_productscale_orient/'),
    #            outpath_masks_train=os.path.join(spacenet_out_dir, 'masksandweights.npz'),
    #            outpath_image_train=os.path.join(spacenet_out_dir, 'allimages.npz'))
    spacenet_masks_file = os.path.join(spacenet_base_path, 'SummaryData/SN6_Train_AOI_11_Rotterdam_Buildings.csv')
    train_images_with_mask = []

    image_names = np.array(os.listdir(os.path.join(spacenet_out_dir, './train_sar_productscale_orient/')))
    train_builds = gpd.read_file(spacenet_masks_file)
    for tile in image_names:
        if train_builds[train_builds.ImageId == tile[:-4]]['PolygonWKT_Pix'].values[0] != 'POLYGON EMPTY':
            train_images_with_mask.append(tile[:-4])

    masks_dir = os.path.join(spacenet_out_dir, 'masksandweights.npz')
    images_dir = os.path.join(spacenet_out_dir, 'allimages.npz')

    allimages = np.load(images_dir)['a']
    print ('Loaded images from', images_dir)
    allmasks = np.load(masks_dir)['a']
    print ('Loaded masks from', masks_dir)
    sample_weights = np.load(masks_dir)['b']
    print ('Loaded weights from', masks_dir)

    for exps in nasios.experiments2:
        print (exps['exp_id'])
        out_dir  = os.path.join(spacenet_out_dir, exps['exp_id'])

        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        for i in range(len(exps['sizes'])):
            id = exps['exp_id']
            md = exps['mode'][i]
            sz = exps['sizes'][i]
            bs = exps['bs'][i]
            encoder = exps['encoder']
            activation = exps['activation']
            init_lr = exps['init_lr']
            epochs = exps['epochs'][i]
            snapshots = exps['snapshots'][i]
            output_model_file_name = out_dir + '/' + exps['exp_id'] + '_' + str(sz) + '_' + str(md) + '.pth'
            output_txt_file_name = out_dir + '/' + exps['exp_id'] + '_' + str(sz) + '_' + str(md) + '.txt'
            print ('id', id)
            print ('size', sz)
            print ('mode', md)
            print ('batch size ', bs)
            print ('encoder', encoder)
            print ('init lr', init_lr)
            print ('epochs', epochs)
            print ('snapshots', snapshots)
            print ('output_model_file_name', output_model_file_name)
            print ('output_txt_file_name', output_txt_file_name)

            if i == 0:
                print('loading from imagenet')
                nasios.train_model(model_encoder=encoder, bs=bs, epochs=epochs, output_dir=out_dir, init_lr=init_lr,
                                                 images_dir=images_dir,masks_dir=masks_dir, name=id, SIZE=sz,
                                                 init_model=None, train_ids=train_images_with_mask,
                                                 output_model=output_model_file_name, output_txt=output_txt_file_name,
                                                 snapshots=snapshots, allimages=allimages, allmasks=allmasks,
                                                 sample_weights=sample_weights, mode=md)
            else:
                print ('->', i, str(exps['mode'][i]), i-1, str(exps['mode'][i-1]))
                init_model_file_name =  out_dir + '/' + str(exps['exp_id']) + '_' + str(exps['sizes'][i-1]) + '_' + str(exps['mode'][i-1]) + '.pth'
                print('loading from ', init_model_file_name)

                nasios.train_model(model_encoder=encoder, bs=bs, epochs=epochs, output_dir=out_dir, init_lr=init_lr,
                                                 images_dir=images_dir,  masks_dir=masks_dir, name=id, SIZE=sz,
                                                 init_model=init_model_file_name, train_ids=train_images_with_mask,
                                                 output_model=output_model_file_name, output_txt=output_txt_file_name,
                                                  snapshots=snapshots, allimages=allimages, allmasks=allmasks,
                                                 sample_weights=sample_weights, mode=md)


    exit(1)