import os
import sys
import glob
import time
import json
from tqdm import tqdm
import argparse
import numpy as np
from collections import Counter
import tensorflow as tf
from pathlib import Path
import nibabel as nib
from sklearn.utils import shuffle
import multiprocess as mp
import numba
from numba import jit
from collections import defaultdict
from multiprocessing import Process, Pool
from sklearn.model_selection import train_test_split
from preprocessing import whitening_3D, normalise_zero_one_3D, normalise_one_one_3D


DESCRIPTION = """For example:
$ python write_TfRecords_args.py -d list of derivatives\
                                        -t float \
                                        -v float \
                                        -n str
"""


deriv_options = ['alff', 'degree_binarize', 'degree_weighted',
            'dual_regression', 'eigenvector_binarize',
            'eigenvector_weighted', 'falff',  'lfcd', 'reho', 'vmhc']

norm_options = ['whitening', 'zero_one', 'one_one']


def build_parser():
    '''Arguments parser
       use write_TfRecords_parser_v1.py -h on terminal
    '''
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    optional = parser.add_argument_group('optional arguments')
    required.add_argument('-d', nargs='+',
                        help='one or more derivatives seperated with space',
                        required=True, choices=deriv_options)
    optional.add_argument('-t', type=float,
                        default=0.7, help='train size, default=0.7')
    optional.add_argument('-v', type=float,
                        default=0.15, help='valid size, default=0.15')
    optional.add_argument('-n', type=str,
                        default=None, help='normalization method, default=None',
                        choices = norm_options)

    return parser
def get_paths(input_dir, derivatives):
    ''' Get the paths of the requested derivatives from user arguments
            [INPUT]
    input_dir -> parent directory of the derivatives
    derivatives -> list of requested derivatives to combine
            [OUTPUT]
    paths_array -> an array with the paths (columns : number of derivatives, rows: number of subjects i.e 1035)
    labels -> corresponding label for each subject
    '''

    nDerivatives = len(derivatives)
    nASD = 505 # number of ASD
    nCON = 530 # number of CONTROL

    paths_ASD = np.empty( (nASD, ) + (nDerivatives, ), dtype ='object' )
    paths_CON = np.empty( (nCON, ) + (nDerivatives, ), dtype ='object' )

    labels_ASD = np.ones(nASD)
    labels_CON = np.zeros(nCON)

    i = 0
    # iterate over derivatives
    for derivative in derivatives :
        # iterate over classes (ASD, CON)
        for class_name in os.listdir(f'{input_dir}/{derivative}'):
            # get the paths of the derivate and class
            image_paths = np.array(sorted(glob.glob(f'{input_dir}{derivative}/{class_name}/*')))
            # fill the columns of the paths array
            if class_name == 'ASD':

                paths_ASD[:, i] = image_paths
            else:
                paths_CON[:, i] = image_paths
        i+=1

    # stuck the arrays vertically
    paths_array = np.vstack((paths_ASD, paths_CON))
    labels = np.concatenate( (np.ones(nASD), np.zeros(nCON)), axis=None )

    return paths_array, labels

def volume_from_paths(paths):
    '''Get volume from a list of paths
            [INPUT]
    paths -> an arary of paths
            [OUTPUT]
    Image -> concatenated image
    '''

    # load each image from the paths array
    im = [nib.load(path).get_fdata(dtype = 'float32') for path in paths]
    # expand the dimensions if ndim == 3
    im = [x[:, :, :, np.newaxis] if x.ndim ==3 else x for x in im ]
    # concatenate images on channel axis i.e 3
    concat_im = np.concatenate(im, axis=3)
    return concat_im

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

#
# def permute_data(X, y, nImages):
#
#     perm = np.random.permutation(np.arange(nImages))
#     X = [X[i] for i in perm]
#     y = y[perm]
#
#     return X, y

# def load_concat_single_volume(paths):
#     np_concat =  np.array([])
#     for path in paths:
#         img_ = sitk.ReadImage(path)
#         img = sitk.GetArrayFromImage(img_)
#         if len(img.shape) == 3 :
#             img = img[np.newaxis, :, :, :]
#             np_concat = np.concatenate([np_concat, img], axis=0) if np_concat.size else img
#         else :
#             np_concat = np.concatenate([np_concat, img], axis=0) if np_concat.size else img
#
#     final_img = np.moveaxis(np_concat, 0, -1)
#     return final_img



## Write TFRecord files
def write_tfrecord_files(X, Y, nImagesPerFile, fileOutRoot, norm_method):

    imIndex = 0
    k = 0
    for fileIndex, nImagesFile in enumerate(tqdm(nImagesPerFile)):

        fileName = '%s_%04i.tfrecord' % (fileOutRoot, fileIndex)
        print(fileName)
        print('Done %i files' % imIndex)

        writer = tf.io.TFRecordWriter(fileName)

        for volumeFile, label in zip(X[imIndex:imIndex+nImagesFile], Y[imIndex:imIndex+nImagesFile]):
            # print(label)
            # print(volumeFile)
            # print('\n')

            # Load volumes
            image = volume_from_paths(volumeFile)
            # print(image.shape)

            # Normalise or not according to user input
            if norm_method == None:
                img = image
            elif norm_method == 'whitening':
                img = whitening_3D(image)
            elif norm_method == 'one_one':
                img = normalise_one_one_3D(image)
            else :
                img = normalise_zero_one_3D(image)

            #image = image / 60 - 1
            # image = (255 * (image - np.min(image)) / (np.max(image) - np.min(image)) )
            # image = image / 127.5 - 1
            # image = image[:,:,:,np.newaxis]
            #image = np.clip(image, -1, 1)
            x_dim = img.shape[0]
            y_dim = img.shape[1]
            z_dim = img.shape[2]
            nChannels = img.shape[3]
            # print(f'DIMS: x: {x_dim}, y: {y_dim}, z: {z_dim}, channels:{nChannels}')

            # Define features
            feature = {
            'x_dim': _int64_feature(int(x_dim)),
            'y_dim': _int64_feature(int(y_dim)),
            'z_dim': _int64_feature(int(z_dim)),
            'channels': _int64_feature(int(nChannels)),
            'image': _bytes_feature(tf.compat.as_bytes(img.tostring())),
            'label': _int64_feature(int(label))
            }

            tf_example = tf.train.Example(features=tf.train.Features(feature=feature))

            writer.write(tf_example.SerializeToString())
        k += len(X[imIndex:imIndex+nImagesFile])

        imIndex += nImagesFile

    print(f'Total number of images on {fileOutRoot} files : -->{k}')
    return k





def check_options(parser, options):
    '''
    function to check the Arguments pased on parser
    '''
    derivatives = options.d
    norm_method = options.n
    split_sizes={'train_size': options.t,
             'valid_size': options.v}
    percentage_sum = sum(list(split_sizes.values()))

    if  percentage_sum >1:
        raise Exception("Train and valid sizes should be < 1.0!")

    # if not os.path.isdir(dataset_path):
    #     sys.exit(' Dataset ' + dataset_path + ' does not exist')
    split_sizes['test_size'] = np.round(1.0 - percentage_sum, 2)

    return derivatives, split_sizes , norm_method


def print_classes(Y, title):
    classes_count = Counter(Y)
    print(f'Number of {title} classes --> ASD: {classes_count[1.0]}, CON: {classes_count[0.0]}')


def main():
    # bluild the parser
    parser = build_parser()
    # take the arguments
    options = parser.parse_args()
    # check the options and return
    derivatives, split_dict, norm_method = check_options(parser, options)
    # parent directory where the derivatives are named DATA
    dataset_path = f'{os.getcwd()}/DATA/'
    print(dataset_path)
    # name of the output files
    filename = '_'.join(derivatives)
    # path to write the derivatives
    out_path = f'{os.getcwd()}/TfRecords/{filename}_no_aug/'
    folderOut = out_path

    try :
        os.makedirs(folderOut)
    except FileExistsError:
         print(f'The combination of {", ".join(derivatives)} already exists!\nRemove old combination.\n')
         os.system(f'rm -rf {folderOut}')
         os.makedirs(folderOut)

    # get paths array and labels_ASD
    X, y = get_paths(dataset_path, derivatives)

    train_ratio = split_dict['train_size']
    validation_ratio = split_dict['valid_size']
    test_ratio = split_dict['test_size']
    print(f'train ratio: {train_ratio}--valid ratio: {validation_ratio}--test_ratio: {test_ratio}')
    # shuffle the data
    X, y = shuffle(X, y, random_state=10)
    # split to train test split
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, y,
                                                    test_size = (1 - train_ratio))

    Xvalid, Xtest, Yvalid, Ytest = train_test_split(Xtest, Ytest,
                                                    test_size = (test_ratio/(test_ratio + validation_ratio)))

    print_classes(Ytrain, 'train')
    print_classes(Yvalid, 'valid')
    print_classes(Ytest, 'test')

    nImages_train = len(Ytrain)
    nImages_valid = len(Yvalid)
    nImages_test = len(Ytest)

    nImagesPerFileTrain = compute_n_files(nImages_train)
    nFilesTrain = len(nImagesPerFileTrain)

    nImagesPerFileValid = compute_n_files(nImages_valid)
    nFilesValid = len(nImagesPerFileValid)

    nImagesPerFileTest = compute_n_files(nImages_test)
    nFilesTest = len(nImagesPerFileTest)

    print('nFiles train : %i, valid : %i, test : %i' % (nFilesTrain, nFilesValid, nFilesTest))
    print('\n\nStart writing TfRecords')

    # count the number of channels
    if 'dual_regression' not in derivatives :
        nChannels = len(derivatives)
    else :
        nChannels = len(derivatives) + 9
    # write the TfRecords
    fileOutRoot = os.path.join(folderOut, 'train')
    totImgTrain = write_tfrecord_files(Xtrain, Ytrain, nImagesPerFileTrain, fileOutRoot, norm_method)

    fileOutRoot = os.path.join(folderOut, 'valid')
    totImgValid = write_tfrecord_files(Xvalid, Yvalid, nImagesPerFileValid, fileOutRoot, norm_method)

    fileOutRoot = os.path.join(folderOut, 'test')
    totImgTest = write_tfrecord_files(Xtest, Ytest, nImagesPerFileTest, fileOutRoot, norm_method)

    # create a metadata file
    metadata = {
    "Derivatives": f'{derivatives}',
    "TfRecords": [{'train':f'{totImgTrain}','valid':f'{totImgValid}','test':f'{totImgTest}'}],
    "nChannels": f'{nChannels}',
    "Normalise" : f'{norm_method}'
    }
    with open(f'{out_path}/metadata.txt', 'w') as json_file:
        json.dump(metadata, json_file)
    print(f'TfRecords created for derivatives: {" ".join(derivatives)}!')


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
