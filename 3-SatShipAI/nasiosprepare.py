import os
import sys
import shutil
import argparse
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


    import nasiosdataprocess

    train_images_with_mask = nasiosdataprocess.prepare_train(spacenet_train_base_path=spacenet_base_path,
               outpath_train=os.path.join(spacenet_out_dir, './train_sar_productscale_orient/'),
               outpath_masks_train=os.path.join(spacenet_out_dir, 'masksandweights.npz'),
               outpath_image_train=os.path.join(spacenet_out_dir, 'allimages.npz'))

    masks_dir = os.path.join(spacenet_out_dir, 'masksandweights.npz')
    images_dir = os.path.join(spacenet_out_dir, 'allimages.npz')

