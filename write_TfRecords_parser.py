import os
import sys
import glob
import time
import json
import SimpleITK as sitk
from tqdm import tqdm
import argparse
import numpy as np
import tensorflow as tf
from pathlib import Path
import nibabel as nib
from nibabel.funcs import concat_images
from collections import defaultdict

DESCRIPTION = """For example:
$ python write_TfRecords_args.py -d list of derivatives\
                                        -t float \
                                        -v float
"""

options = ['alff', 'degree_binarize', 'degree_weighted',
            'dual_regression', 'eigenvector_binarize',
            'eigenvector_weighted', 'falff',  'lfcd', 'reho', 'vmhc']

def build_parser():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    optional = parser.add_argument_group('optional arguments')
    required.add_argument('-d', nargs='+',
                        help='one or more derivatives seperated with space',
                        required=True, choices=options)
    optional.add_argument('-t', type=float,
                        default=0.7, help='train size, default=0.7')
    optional.add_argument('-v', type=float,
                        default=0.2, help='validation size, default=0.2')

    return parser



def split_paths(input_dir, split_sizes, derivatives):
    class_dict = defaultdict()

    if sum(list(split_sizes.values())) > 1 :
            raise Exception("Train and Validation sizes should be < 1.0!")
    else :
        train_size, val_size = split_sizes.values()
        val_size += train_size


    paths_ASD = np.array([])
    paths_CON = np.array([])

    for derivative in derivatives :
        for class_name in os.listdir(f'{input_dir}/{derivative}'):
            image_paths = np.array(sorted(glob.glob(f'{input_dir}{derivative}/{class_name}/*')))
            image_paths = np.reshape(image_paths, (image_paths.shape[0], 1))
            if class_name == 'ASD':
                paths_ASD = np.concatenate([paths_ASD, image_paths], axis=1) if paths_ASD.size else image_paths
            else:
                paths_CON = np.concatenate([paths_CON, image_paths], axis=1) if paths_ASD.size else image_paths


    class_dict['train_ASD'], class_dict['valid_ASD'], class_dict['test_ASD'] =\
                    np.split(paths_ASD,indices_or_sections = [int(train_size * paths_ASD.shape[0]),
                     int(val_size*paths_ASD.shape[0])],
                                axis=0)

    class_dict['train_CON'], class_dict['valid_CON'], class_dict['test_CON'] =\
                    np.split(paths_CON,indices_or_sections = [int(train_size * paths_CON.shape[0]),
                    int(val_size*paths_CON.shape[0])],
                                axis=0)
    return class_dict

## TF features
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    # Since this will be used to convert an np.array we don't use []
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def compute_n_files(nImages):
    # Determine number of TFRecord files
    maxImagesPerFile = 32
    nImagesPerFile = []
    temp = nImages
    while temp >= 0:
        if temp > maxImagesPerFile:
            nImagesPerFile.append(maxImagesPerFile)
            temp -= maxImagesPerFile
        else:
            nImagesPerFile.append(temp)
            break

    return nImagesPerFile


def permute_data(X, y, nImages):

    perm = np.random.permutation(np.arange(nImages))
    X = [X[i] for i in perm]
    y = y[perm]

    return X, y
    

def load_concat_single_volume(paths):
    np_concat =  np.array([])
    for path in paths:
        img_ = sitk.ReadImage(path)
        img = sitk.GetArrayFromImage(img_)
        if len(img.shape) == 3 :
            img = img[np.newaxis, :, :, :]
            np_concat = np.concatenate([np_concat, img], axis=0) if np_concat.size else img
        else :
            np_concat = np.concatenate([np_concat, img], axis=0) if np_concat.size else img

    final_img = np.moveaxis(np_concat, 0, -1)
    return final_img

## Write TFRecord files
def write_tfrecord_files(X, Y, nImagesPerFile, fileOutRoot):

    imIndex = 0

    for fileIndex, nImagesFile in enumerate(tqdm(nImagesPerFile)):

        fileName = '%s_%04i.tfrecord' % (fileOutRoot, fileIndex)
        print(fileName)
        print('Done %i files' % imIndex)

        writer = tf.io.TFRecordWriter(fileName)

        for volumeFile, label in zip(X[imIndex:imIndex+nImagesFile], Y[imIndex:imIndex+nImagesFile]):

            # Load volumes
            # image = load_single_volume(volumeFile)

            image = load_concat_single_volume(volumeFile)
            #print(f'DIMS: x: {x_dim}, y: {y_dim}, z: {z_dim}')
            #image = image / 60 - 1
            # image = (255 * (image - np.min(image)) / (np.max(image) - np.min(image)) )
            # image = image / 127.5 - 1
            # image = image[:,:,:,np.newaxis]
            #image = np.clip(image, -1, 1)
            # x_dim = image.shape[0]
            # y_dim = image.shape[1]
            # z_dim = image.shape[2]

            img_shape = image.shape
            # print(img_shape)
            # Define features
            feature = {
            # 'x_dim': _int64_feature(int(x_dim)),
            # 'y_dim': _int64_feature(int(y_dim)),
            # 'z_dim': _int64_feature(int(z_dim)),

            'x_dim': _int64_feature(int(img_shape[0])),
            'y_dim': _int64_feature(int(img_shape[1])),
            'z_dim': _int64_feature(int(img_shape[2])),
            'channels': _int64_feature(int(img_shape[3])),
            'image': _bytes_feature(tf.compat.as_bytes(image.tostring())),
            'label': _int64_feature(int(label))
            }

            tf_example = tf.train.Example(features=tf.train.Features(feature=feature))

            writer.write(tf_example.SerializeToString())

        imIndex += nImagesFile

def main():

    parser = build_parser()
    options = parser.parse_args()

    if options.d==None:
        print("No derivatives are given")
        parser.print_help()
        sys.exit(1)

    derivatives = options.d

    split_dict={'train': options.t,
             'validation': options.v}

    dataset_path = f'{os.getcwd()}/DATA/'
    print(dataset_path)
    filename = '_'.join(derivatives)
    if os.path.exists(f'{os.getcwd()}/TfRecords/{filename}'):
        print(f'The combination of {", ".join(derivatives)} already exists!')
        sys.exit(1)

    out_path = f'{os.getcwd()}/TfRecords/{filename}/'

    if not os.path.isdir(dataset_path):
        sys.exit(' Dataset ' + dataset_path + ' does not exist')

    get_paths = split_paths(dataset_path, split_dict, derivatives)


    controls_train_path = get_paths['train_CON'].tolist()
    asds_train_path = get_paths['train_ASD'].tolist()

    controls_valid_path = get_paths['valid_CON'].tolist()
    asds_valid_path = get_paths['valid_ASD'].tolist()

    controls_test_path = get_paths['test_CON'].tolist()
    asds_test_path = get_paths['test_ASD'].tolist()


    Xtrain = controls_train_path + asds_train_path
    Xvalid = controls_valid_path + asds_valid_path
    Xtest = controls_test_path + asds_test_path

    nA_train = len(controls_train_path)
    nB_train = len(asds_train_path)
    nImages_train = nA_train + nB_train

    nA_valid = len(controls_valid_path)
    nB_valid = len(asds_valid_path)
    nImages_valid = nA_valid + nB_valid

    nA_test = len(controls_test_path)
    nB_test = len(asds_test_path)
    nImages_test = nA_test + nB_test

    print('Training: %i, %i, total: %i' % (nA_train, nB_train, nImages_train))
    print('Validation: %i, %i, total: %i' % (nA_valid, nB_valid, nImages_valid))
    print('Test: %i, %i, total: %i' % (nA_test, nB_test, nImages_test))

    Ytrain = np.concatenate( (np.zeros(nA_train), np.ones(nB_train)), axis=None )
    Yvalid = np.concatenate( (np.zeros(nA_valid), np.ones(nB_valid)), axis=None )
    Ytest = np.concatenate( (np.zeros(nA_test), np.ones(nB_test)), axis=None )

    ## Permute data

    Xtrain, Ytrain = permute_data(Xtrain, Ytrain, nImages_train)
    Xvalid, Yvalid = permute_data(Xvalid, Yvalid, nImages_valid)
    Xtest, Ytest = permute_data(Xtest, Ytest, nImages_test)

    folderOut = out_path
    if not os.path.exists(folderOut):
        os.makedirs(folderOut)


    nImagesPerFileTrain = compute_n_files(nImages_train)
    nFilesTrain = len(nImagesPerFileTrain)

    nImagesPerFileValid = compute_n_files(nImages_valid)
    nFilesValid = len(nImagesPerFileValid)

    nImagesPerFileTest = compute_n_files(nImages_test)
    nFilesTest = len(nImagesPerFileTest)

    print('nFiles: %i, %i, %i' % (nFilesTrain, nFilesValid, nFilesTest))
    print('\n\nStart writing TfRecords')

    fileOutRoot = os.path.join(folderOut, 'train')
    write_tfrecord_files(Xtrain, Ytrain, nImagesPerFileTrain, fileOutRoot)
    fileOutRoot = os.path.join(folderOut, 'valid')
    write_tfrecord_files(Xvalid, Yvalid, nImagesPerFileValid, fileOutRoot)

    fileOutRoot = os.path.join(folderOut, 'test')
    write_tfrecord_files(Xtest, Ytest, nImagesPerFileTest, fileOutRoot)

    if 'dual_regression' not in derivatives :
        nChannels = len(derivatives)
    else :
        nChannels = len(derivatives) + 9

    metadata = {
    "augmentation": f'{derivatives}',
    "TfRecords": [{'train':f'nFilesTrain','valid':f'nFilesValid','test':f'nFilesTest'}],
    "nChannels": f'{nChannels}'
    }

    with open(f'{out_path}/metadata.txt', 'w') as json_file:
        json.dump(metadata, json_file)
    print('TfRecords created!')


if __name__ == '__main__':
    main()

    # THIS CODE USED IN THE SPLIT PATHS TO CREATE ARRAYS FROM THE DERIVATIVES
    # BUT LOOPING AND CONCATENATE ARRAYS IS VERY TIME CONSUMING
    # for i in np.ndindex(paths_ASD.shape[0]):
    #     print (i, concat_images(paths_ASD[i]).get_fdata())
    #
    #
    # np_ASD = np.array([])
    # np_CON = np.array([])
    # for derivative in derivatives:
    #     print(derivative)
    #
    #     for class_name in os.listdir(f'{input_dir}/{derivative}'):
    #         image_paths = np.array(glob.glob(f'{input_dir}/{derivative}/{class_name}/*'))
    #
    #         image_arrays = [nib.load(file).get_fdata() for file in image_paths]
    #         #print(image_arrays[0].shape)
    #         #sys.exit(1)
    #         image_arrays = np.array([array[:,:,:,np.newaxis] for array in image_arrays])
    #         if class_name == 'ASD' :
    #             np_ASD =np.concatenate([np_ASD, image_arrays], axis=4) if np_ASD.size else image_arrays
    #         else :
    #             np_CON =np.concatenate([np_CON, image_arrays], axis=4) if np_ASD.size else image_arrays
    #
    # np_ASD = np.take(np_ASD,np.random.rand(np_ASD.shape[0]).argsort(),axis=0,out=np_ASD)
    # np_CON = np.take(np_ASD,np.random.rand(np_ASD.shape[0]).argsort(),axis=0,out=np_ASD)
    #
    # class_dict['train_ASD'], class_dict['validate_ASD'], class_dict['test_ASD'] = np.split(np_ASD,
    #             indices_or_sections = [int(train_size * len(image_paths)), int(val_size*len(image_paths))],
    #                     axis=0)
    # class_dict['train_CON'], class_dict['validate_CON'], class_dict['test_CON'] = np.split(np_CON,
    #             indices_or_sections = [int(train_size * len(image_paths)), int(val_size*len(image_paths))],
    #                     axis=0)
    #
    # return class_dict
