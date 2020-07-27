import os
import sys
import glob
import math
import uuid
import shutil
import pathlib
import argparse
import numpy as np
import pandas as pd
import geopandas as gpd
import skimage
import tqdm
import gdal

import solaris as sol


def makeemptyfolder(path):
    """
    Create an empty folder, deleting anything there already
    """
    shutil.rmtree(path, ignore_errors=True)
    pathlib.Path(path).mkdir(exist_ok=True)


def readrotationfile(path):
    """
    Reads SAR_orientations file, which lists whether each strip was imaged
    from the north (denoted by 0) or from the south (denoted by 1).
    """
    rotationdf = pd.read_csv(path,
                             sep=' ',
                             index_col=0,
                             names=['strip', 'direction'],
                             header=None)
    rotationdf['direction'] = rotationdf['direction'].astype(int)
    return rotationdf


def lookuprotation(tilepath, rotationdf):
    """
    Looks up the SAR_orientations value for a tile based on its filename
    """
    tilename = os.path.splitext(os.path.basename(tilepath))[0]
    stripname = '_'.join(tilename.split('_')[-4:-2])
    rotation = rotationdf.loc[stripname].squeeze()
    return rotation


def copyrotateimage(srcpath, dstpath, rotate=False, deletesource=False):
    """
    Copying with rotation:  Copies a TIFF image from srcpath to dstpath,
    rotating the image by 180 degrees if specified.
    """
    #Handles special case where source path and destination path are the same
    if srcpath==dstpath:
        if not rotate:
            #Then there's nothing to do
            return
        else:
            #Move file to temporary location before continuing
            srcpath = srcpath + str(uuid.uuid4())
            shutil.move(dstpath, srcpath)
            deletesource = True

    if not rotate:
        shutil.copy(srcpath, dstpath, follow_symlinks=True)
    else:
        driver = gdal.GetDriverByName('GTiff')
        tilefile = gdal.Open(srcpath)
        copyfile = driver.CreateCopy(dstpath, tilefile, strict=0)
        numbands = copyfile.RasterCount
        for bandnum in range(1, numbands+1):
            banddata = tilefile.GetRasterBand(bandnum).ReadAsArray()
            banddata = np.fliplr(np.flipud(banddata)) #180 deg rotation
            copyfile.GetRasterBand(bandnum).WriteArray(banddata)
        copyfile.FlushCache()
        copyfile = None
        tilefile = None

    if deletesource:
        os.remove(srcpath)


def reorderbands(srcpath, dstpath, bandlist, deletesource=False):
    """
    Copies a TIFF image from srcpath to dstpath, reordering the bands.
    """
    #Handles special case where source path and destination path are the same
    if srcpath==dstpath:
        #Move file to temporary location before continuing
        srcpath = srcpath + str(uuid.uuid4())
        shutil.move(dstpath, srcpath)
        deletesource = True

    driver = gdal.GetDriverByName('GTiff')
    tilefile = gdal.Open(srcpath)
    geotransform = tilefile.GetGeoTransform()
    projection = tilefile.GetProjection()
    numbands = len(bandlist)
    shape = tilefile.GetRasterBand(1).ReadAsArray().shape
    copyfile = driver.Create(dstpath, shape[1], shape[0],
                             numbands, gdal.GDT_Byte)
    for bandnum in range(1, numbands+1):
        banddata = tilefile.GetRasterBand(bandlist[bandnum-1]).ReadAsArray()
        copyfile.GetRasterBand(bandnum).WriteArray(banddata)
    copyfile.SetGeoTransform(geotransform)
    copyfile.SetProjection(projection)
    copyfile.FlushCache()
    copyfile = None
    tilefile = None

    if deletesource:
        os.remove(srcpath)


def pretrain(args):
    """
    Creates rotated versions of imagery used for training
    as well as raster label masks.  Optical bands are re-ordered to
    mimic SAR bands.
    """
    print('Pretrain')
    assert(args.sardir is not None and args.labeldir is not None and args.maskdir is not None and args.sarprocdir is not None)

    #Save local copy of rotation file
    if args.rotate:
        shutil.copy(args.rotationfile, args.rotationfilelocal, follow_symlinks=True)

    #Get paths to relevant files
    sarpaths = glob.glob(os.path.join(args.sardir, '*.tif'))
    sarpaths.sort()
    labelpaths = glob.glob(os.path.join(args.labeldir, '*.geojson'))
    labelpaths.sort()
    maskpaths = [os.path.join(args.maskdir, os.path.basename(sarpath)) for sarpath in sarpaths]
    sarprocpaths = [os.path.join(args.sarprocdir, os.path.basename(sarpath)) for sarpath in sarpaths]
    if args.opticaldir is not None:
        opticalpaths = glob.glob(os.path.join(args.opticaldir, '*.tif'))
        opticalpaths.sort()
        opticalprocpaths = [os.path.join(args.opticalprocdir, os.path.basename(opticalpath)) for opticalpath in opticalpaths]
    else:
        opticalpaths = [''] * len(sarpaths)
        opticalprocpaths = [''] * len(sarpaths)

    #Create empty folders to hold masks, processed SAR, & processed optical
    folders = [args.maskdir, args.sarprocdir]
    if args.opticalprocdir is not None:
        folders.append(args.opticalprocdir)
    for folder in folders:
        makeemptyfolder(folder)
    pathlib.Path(args.modeldir).mkdir(exist_ok=True)

    #Look up how to rotate masks and images, if enabled
    if args.rotate:
        assert(args.rotationfilelocal is not None)
        rotationdf = readrotationfile(args.rotationfilelocal)

    #Create masks, with optional rotation and optional size threshold
    #Also copy SAR and optical imagery to local folder, with optional rotation
    #Optical files are optionally modified to more closely resemble SAR
    #Also create Pandas dataframe of training data
    combodf = pd.DataFrame(columns=['opticalimage',
                                    'sarimage',
                                    'label',
                                    'group'])
    ledge = 591550 #Approximate west edge of training data area
    redge = 596250 #Approximate east edge of training data area
    numgroups = 5
    reorganizeoptical = True
    for i, (sarpath, opticalpath, labelpath, maskpath, sarprocpath, opticalprocpath) in tqdm.tqdm(enumerate(zip(sarpaths, opticalpaths, labelpaths, maskpaths, sarprocpaths, opticalprocpaths)), total=len(sarpaths)):
        #Generate mask
        gdf = gpd.read_file(labelpath)
        if args.mintrainsize is not None:
            cut = gdf.area > float(args.mintrainsize)
            gdf = gdf.loc[cut]
        maskdata = code.solaris.vector.mask.footprint_mask(
            df=gdf,
            reference_im=sarpath,
            out_file=maskpath
        )
        #Optionally rotate mask
        if args.rotate:
            rotationflag = lookuprotation(sarpath, rotationdf)
        else:
            rotationflag = 0
        if rotationflag==1:
            copyrotateimage(maskpath, maskpath, rotate=True)
        #Copy SAR and optical imagery, with optional rotation
        rotationflagbool = rotationflag == 1
        copyrotateimage(sarpath, sarprocpath, rotate=rotationflagbool)
        if args.opticaldir is not None:
            copyrotateimage(opticalpath, opticalprocpath,
                            rotate=rotationflagbool)
            if reorganizeoptical:
                reorderbands(opticalprocpath, opticalprocpath, [3,1,1,2])

        #Assign the tile to one of a small number of groups, for setting
        #aside validation data (or for k-fold cross-validation, not used here).
        #Caveats: These groups slightly overlap each other.  Also, they are
        #not of equal size.
        sarfile = gdal.Open(sarpath)
        sartransform = sarfile.GetGeoTransform()
        sarx = sartransform[0]
        groupnum = min(numgroups-1, max(0, math.floor((sarx-ledge) / (redge-ledge) * numgroups)))
        combodf = combodf.append({
            'sarimage': sarprocpath,
            'opticalimage': opticalprocpath,
            'label': maskpath,
            'group': groupnum}, ignore_index=True)

        #Optionally end loop early (for debugging purposes)
        if args.earlycutoff is not None:
            if i >= int(args.earlycutoff) - 1:
                break

    #Write reference CSVs for training
    for i in range(numgroups+1):
        print( '%i: %i' % (i, len(combodf[combodf['group']==i])))
    validationgroup = numgroups - 1
    traindf = combodf[combodf['group'] != validationgroup]
    validdf = combodf[combodf['group'] == validationgroup]
    sartraindf = traindf.loc[:, ['sarimage', 'label']].rename(columns={'sarimage':'image'})
    sarvaliddf = validdf.loc[:, ['sarimage', 'label']].rename(columns={'sarimage':'image'})
    opticaltraindf = traindf.loc[:, ['opticalimage', 'label']].rename(columns={'opticalimage':'image'})
    opticalvaliddf = validdf.loc[:, ['opticalimage', 'label']].rename(columns={'opticalimage':'image'})
    sartraindf.to_csv(args.traincsv, index=False)
    sarvaliddf.to_csv(args.validcsv, index=False)
    opticaltraindf.to_csv(args.opticaltraincsv, index=False)
    opticalvaliddf.to_csv(args.opticalvalidcsv, index=False)


#Custom model dictionaries, defined globally
sar_dict = {
    'model_name': 'unet11',
    'weight_path': None,
    'weight_url': None,
    'arch': model.UNet11
}

optical_dict = {
    'model_name': 'unet11',
    'weight_path': None,
    'weight_url': None,
    'arch': model.UNet11
}


def defineyaml(args):
    #YAML
    yamlcontents = """
model_name: unet11

model_path:
train: true
infer: true

pretrained:
nn_framework:  torch
batch_size: 8

data_specs:
  width: 512
  height: 512
  dtype:
  image_type: 32bit
  rescale: false
  rescale_minima: auto
  rescale_maxima: auto
  channels: 4
  label_type: mask
  is_categorical: false
  mask_channels: 1
  val_holdout_frac:
  data_workers:

training_data_csv: '$TRAINCSV'
validation_data_csv: '$VALIDCSV'
inference_data_csv: '$TESTCSV'

training_augmentation:
  augmentations:
    HorizontalFlip:
      p: 0.5
    RandomCrop:
      height: 512
      width: 512
      p: 1.0
    Normalize:
      mean:
        - 0.5
      std:
        - 0.125
      max_pixel_value: 255.0
      p: 1.0
  p: 1.0
  shuffle: true
validation_augmentation:
  augmentations:
    CenterCrop:
      height: 512
      width: 512
      p: 1.0
    Normalize:
      mean:
        - 0.5
      std:
        - 0.125
      max_pixel_value: 255.0
      p: 1.0
  p: 1.0
inference_augmentation:
  augmentations:
    Normalize:
      mean:
        - 0.5
      std:
        - 0.125
      max_pixel_value: 255.0
      p: 1.0
  p: 1.0
training:
  epochs: 50
  steps_per_epoch:
  optimizer: AdamW
  lr: 1e-4
  opt_args:
  loss:
    dice:
        logits: true
    focal:
        logits: true
  loss_weights:
    dice: 1.0
    focal: 10.0
  metrics:
    training:
    validation:
  checkpoint_frequency: 10
  callbacks:
    model_checkpoint:
      filepath: '$MODELDIR/best.model'
      monitor: val_loss
  model_dest_path: '$MODELDIR/last.model'
  verbose: true

inference:
  window_step_size_x: 512
  window_step_size_y: 512
  output_dir: '$TESTOUTDIR'
"""
    if args.traincsv is not None:
        yamlcontents = yamlcontents.replace('$TRAINCSV', args.traincsv)
    if args.validcsv is not None:
        yamlcontents = yamlcontents.replace('$VALIDCSV', args.validcsv)
    if args.testcsv is not None:
        yamlcontents = yamlcontents.replace('$TESTCSV', args.testcsv)
    if args.modeldir is not None:
        yamlcontents = yamlcontents.replace('$MODELDIR', args.modeldir)
    if args.testoutdir is not None:
        yamlcontents = yamlcontents.replace('$TESTOUTDIR', args.testoutdir)
    yamlfile = open(args.yamlpath, 'w')
    yamlfile.write(yamlcontents)
    yamlfile.close()



def defineopticalyaml(args):
    #Optical YAML
    yamlcontents = """
model_name: unet11

model_path:
train: true
infer: false

pretrained: false
nn_framework:  torch
batch_size: 8

data_specs:
  width: 512
  height: 512
  dtype:
  image_type: 32bit
  rescale: false
  rescale_minima: auto
  rescale_maxima: auto
  channels: 4
  label_type: mask
  is_categorical: false
  mask_channels: 1
  val_holdout_frac:
  data_workers:

training_data_csv: '$OPTICALTRAINCSV'
validation_data_csv: '$OPTICALVALIDCSV'
inference_data_csv:

training_augmentation:
  augmentations:
    HorizontalFlip:
      p: 0.5
    RandomRotate90:
      p: 1.0
    RandomCrop:
      height: 512
      width: 512
      p: 1.0
    Normalize:
      mean:
        - 0.5
      std:
        - 0.125
      max_pixel_value: 255.0
      p: 1.0
  p: 1.0
  shuffle: true
validation_augmentation:
  augmentations:
    CenterCrop:
      height: 512
      width: 512
      p: 1.0
    Normalize:
      mean:
        - 0.5
      std:
        - 0.125
      max_pixel_value: 255.0
      p: 1.0
  p: 1.0
inference_augmentation:
  augmentations:
    Normalize:
      mean:
        - 0.5
      std:
        - 0.125
      max_pixel_value: 255.0
      p: 1.0
  p: 1.0
training:
  epochs: 150
  steps_per_epoch:
  optimizer: AdamW
  lr: 1e-4
  opt_args:
  loss:
    dice:
        logits: true
    focal:
        logits: true
  loss_weights:
    dice: 1.0
    focal: 10.0
  metrics:
    training:
    validation:
  checkpoint_frequency: 10
  callbacks:
    model_checkpoint:
      filepath: '$MODELDIR/opticalbest.model'
      monitor: val_loss
  model_dest_path: '$MODELDIR/opticallast.model'
  verbose: true
"""
    yamlcontents = yamlcontents.replace('$OPTICALTRAINCSV',
                                        args.opticaltraincsv)
    yamlcontents = yamlcontents.replace('$OPTICALVALIDCSV',
                                        args.opticalvalidcsv)
    yamlcontents = yamlcontents.replace('$MODELDIR', args.modeldir)
    yamlfile = open(args.opticalyamlpath, 'w')
    yamlfile.write(yamlcontents)
    yamlfile.close()


def train(args):
    """
    Trains the model.
    """
    print('Train')
    
    #Create YAML files
    defineyaml(args)
    defineopticalyaml(args)

    #Optionally start by training on optical imagery for transfer learning
    if args.transferoptical:
        print('Training on Optical: Start')
        config = code.solaris.utils.config.parse(args.opticalyamlpath)
        trainer = sol.nets.train.Trainer(config,
                                         custom_model_dict=optical_dict)
        trainer.train()

        #Select best-performing optical imagery model
        if not args.uselastmodel:
            modelfiles = sorted(glob.glob(os.path.join(args.modeldir,
                                                       'opticalbest*.model')))
            timestamps = [os.path.getmtime(modelfile)
                          for modelfile in modelfiles]
            latestindex = timestamps.index(max(timestamps))
            modelfile = modelfiles[latestindex]
        else:
            modelfile = os.path.join(args.modeldir, 'opticallast.model')
        print(modelfile)
        destfile = os.path.join(args.modeldir, 'optical.model')
        shutil.copyfile(modelfile, destfile, follow_symlinks=True)
        print('Training on Optical: End')

    #Instantiate trainer and train on SAR imagery
    config = code.solaris.utils.config.parse(args.yamlpath)
    if args.transferoptical:
        config['pretrained'] = True
        sar_dict['weight_path'] = os.path.join(args.modeldir, 'optical.model')
    else:
        config['pretrained'] = False
    trainer = sol.nets.train.Trainer(config, custom_model_dict=sar_dict)
    trainer.train()


def pretest(args):
    """
    Create rotated versions of imagery used for testing.
    """
    print('Pretest')
    assert(args.testdir is not None and args.testprocdir is not None)

    #Get paths to relevant files
    sarpaths = glob.glob(os.path.join(args.testdir, '*.tif'))
    sarpaths.sort()
    sarprocpaths = [os.path.join(args.testprocdir, os.path.basename(sarpath)) for sarpath in sarpaths]

    #Create empty folder to hold processed test SAR images
    makeemptyfolder(args.testprocdir)

    #Look up how to rotate masks and images, if enabled
    if args.rotate:
        assert(args.rotationfilelocal is not None)
        rotationdf = readrotationfile(args.rotationfilelocal)

    #Copy SAR test images to local folder, with optional rotation
    #Also create Pandas dataframe of testing data
    testdf = pd.DataFrame(columns=['image'])
    for i, (sarpath, sarprocpath) in tqdm.tqdm(enumerate(zip(sarpaths, sarprocpaths)), total=len(sarpaths)):
        #Copy SAR test imagery, with optional rotation
        if args.rotate:
            rotationflag = lookuprotation(sarpath, rotationdf)
        else:
            rotationflag = 0
        rotationflagbool = rotationflag == 1
        copyrotateimage(sarpath, sarprocpath, rotate=rotationflagbool)

        #Add row to Pandas dataframe of testing data
        testdf = testdf.append({
            'image': sarprocpath
        }, ignore_index=True)

        #Optionally end loop early (for debugging purposes)
        if args.earlycutoff is not None:
            if i >= int(args.earlycutoff) - 1:
                break

    #Write reference CSVs for testing
    testdf.to_csv(args.testcsv, index=False)

def test(args):
    """
    Uses the trained model to conduct inference on the test dataset.
    Outputs are a continuously-varying pixel map, a binary pixel map,
    and a CSV file of vector labels for evaluation.
    """
    print('Test')

    #Create SAR YAML file if absent
    if not os.path.exists(args.yamlpath):
        defineyaml(args)

    #Overwrite last model with best model
    modelfiles = sorted(glob.glob(os.path.join(args.modeldir, 'best*.model')))
    timestamps = [os.path.getmtime(modelfile) for modelfile in modelfiles]
    latestindex = timestamps.index(max(timestamps))
    modelfile = modelfiles[latestindex]
    print(modelfile)
    if not args.uselastmodel:
        destfile = os.path.join(args.modeldir, 'last.model')
        shutil.copyfile(modelfile, destfile, follow_symlinks=True)

    #Create empty folders to hold various inference outputs
    folders = [args.testoutdir, args.testbinarydir, args.testvectordir]
    for folder in folders:
        makeemptyfolder(folder)

    #Run inference on the test data
    config = code.solaris.utils.config.parse(args.yamlpath)
    inferer = sol.nets.infer.Inferer(config, custom_model_dict=sar_dict)
    print('Start inference.')
    inferer()
    print('Finished inference.')

    #Binary and vector inference output
    driver = gdal.GetDriverByName('GTiff')
    firstfile = True
    sourcefolder = config['inference']['output_dir']
    sourcefiles = sorted(glob.glob(os.path.join(sourcefolder, '*')))
    if args.rotate:
        rotationdf = readrotationfile(args.rotationfilelocal)
    minbuildingsize = float(args.mintestsize) if args.mintestsize is not None else 0
    for sourcefile in tqdm.tqdm(sourcefiles, total=len(sourcefiles)):
        filename = os.path.basename(sourcefile)
        destfile = os.path.join(args.testbinarydir, filename)

        #Create binary array
        cutoff = 0.
        sourcedataorig = gdal.Open(sourcefile).ReadAsArray()
        sourcedata = np.zeros(np.shape(sourcedataorig), dtype='int')
        sourcedata[np.where(sourcedataorig > cutoff)] = 255
        sourcedata[np.where(sourcedataorig <= cutoff)] = 0

        #Remove small buildings
        if minbuildingsize>0:
            regionlabels, regioncount = skimage.measure.label(sourcedata, background=0, connectivity=1, return_num=True)
            regionproperties = skimage.measure.regionprops(regionlabels)
            for bl in range(regioncount):
                if regionproperties[bl].area < minbuildingsize:
                    sourcedata[regionlabels == bl+1] = 0

        #Save binary image
        destdata = driver.Create(destfile, sourcedata.shape[1], sourcedata.shape[0], 1, gdal.GDT_Byte)
        destdata.GetRasterBand(1).WriteArray(sourcedata)
        del destdata

        #Rotate source data back to real-world orientation before vectorizing
        if args.rotate:
            rotationflag = lookuprotation(filename, rotationdf)
        else:
            rotationflag = 0
        rotationflagbool = rotationflag == 1
        if rotationflag:
            sourcedatarotated = np.fliplr(np.flipud(sourcedata))
        else:
            sourcedatarotated = sourcedata

        #Save vector file (CSV)
        vectorname = '.'.join(filename.split('.')[:-1]) + '.csv'
        vectorfile = os.path.join(args.testvectordir, vectorname)
        referencefile = os.path.join(args.testprocdir, filename)
        vectordata = code.solaris.vector.mask.mask_to_poly_geojson(
            sourcedatarotated,
            output_path=vectorfile,
            output_type='csv',
            min_area=0,
            bg_threshold=128,
            do_transform=False,
            simplify=True
        )

        #Add to the cumulative inference CSV file
        tilename = '_'.join(os.path.splitext(filename)[0].split('_')[-4:])
        csvaddition = pd.DataFrame({'ImageId': tilename,
                                    'BuildingId': 0,
                                    'PolygonWKT_Pix': vectordata['geometry'],
                                    'Confidence': 1
        })
        csvaddition['BuildingId'] = range(len(csvaddition))
        if firstfile:
            proposalcsv = csvaddition
            firstfile = False
        else:
            proposalcsv = proposalcsv.append(csvaddition)

    proposalcsv.to_csv(args.outputcsv, index=False)


def evaluation(args):
    """
    Compares infered test data vector labels to ground truth.
    """
    truthpath = os.path.join(os.path.dirname(args.outputcsv), 'SN6_Test_Public_AOI_11_Rotterdam_Buildings.csv')
    proposalpath = args.outputcsv
    minevalsize = 80

    evaluator = code.solaris.eval.base.Evaluator(truthpath)
    evaluator.load_proposal(proposalpath, proposalCSV=True, conf_field_list=[])
    report = evaluator.eval_iou_spacenet_csv(miniou=0.5, min_area=minevalsize)

    tp = 0
    fp = 0
    fn = 0
    for entry in report:
        tp += entry['TruePos']
        fp += entry['FalsePos']
        fn += entry['FalseNeg']
    f1score = (2*tp) / (2*tp + fp + fn)
    print('Vector F1: {}'.format(f1score))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SpaceNet 6 Baseline Algorithm')
    #Which operations to carry out
    parser.add_argument('--pretrain', action='store_true',
                        help='Whether to format training data')
    parser.add_argument('--train', action='store_true',
                        help='Whether to train model')
    parser.add_argument('--pretest', action='store_true',
                        help='Whether to format testing data')
    parser.add_argument('--test', action='store_true',
                        help='Whether to test model')
    parser.add_argument('--eval', action='store_true',
                        help='Whether to evaluate test output')
    #Training: Input file paths
    parser.add_argument('--sardir',
                        help='Folder of SAR training imagery files')
    parser.add_argument('--opticaldir',
                        help='Folder of optical imagery files')
    parser.add_argument('--labeldir',
                        help='Folder of building footprint vector files')
    parser.add_argument('--rotationfile',
                        help='File of data acquisition directions')
    #Training: Preprocessed file paths
    parser.add_argument('--rotationfilelocal',
                        help='Where to save a copy of directions file')
    parser.add_argument('--maskdir',
                        help='Where to save building footprint masks')
    parser.add_argument('--sarprocdir',
                        help='Where to save preprocessed SAR training files')
    parser.add_argument('--opticalprocdir',
                        help='Where to save preprocessed optical image files')
    #Training and inference: YAML and Reference CSV file paths
    parser.add_argument('--traincsv',
                        help='Where to save reference CSV of training data')
    parser.add_argument('--validcsv',
                        help='Where to save reference CSV of validation data')
    parser.add_argument('--opticaltraincsv',
                        help='Where to save CSV of optical training data')
    parser.add_argument('--opticalvalidcsv',
                        help='Where to save CSV of optical validation data')
    parser.add_argument('--testcsv',
                        help='Where to save reference CSV of testing data')
    parser.add_argument('--yamlpath',
                        help='Where to save YAML file')
    parser.add_argument('--opticalyamlpath',
                        help='Where to save transfer learning YAML file')
    #Training and inference: Model weights file path
    parser.add_argument('--modeldir',
                        help='Where to save model weights')
    #Inference (testing) file paths
    parser.add_argument('--testdir',
                        help='Folder of SAR testing imagery files')
    parser.add_argument('--testprocdir',
                        help='Where to save preprocessed SAR testing files')
    parser.add_argument('--testoutdir',
                        help='Where to save test continuous segmentation maps')
    parser.add_argument('--testbinarydir',
                        help='Where to save test binary segmentation maps')
    parser.add_argument('--testvectordir',
                        help='Where to save test vector label output')
    parser.add_argument('--outputcsv',
                        help='Where to save labels inferred from test data')
    #Algorithm settings
    parser.add_argument('--rotate', action='store_true',
                        help='Rotate tiles to align imaging direction')
    parser.add_argument('--transferoptical', action='store_true',
                        help='Train model on optical before training on SAR')
    parser.add_argument('--mintrainsize',
                        help='Minimum building size (m^2) for training')
    parser.add_argument('--mintestsize',
                        help='Minimum size to output during testing')
    parser.add_argument('--uselastmodel', action='store_true',
                        help='Do not overwrite last model with best model')
    parser.add_argument('--earlycutoff',
                        help='Limit tiles used, for debugging purposes')
    args = parser.parse_args(sys.argv[1:])

    if args.pretrain:
        pretrain(args)
    if args.train:
        train(args)
    if args.pretest:
        pretest(args)
    if args.test:
        test(args)
    if args.eval:
        evaluation(args)
