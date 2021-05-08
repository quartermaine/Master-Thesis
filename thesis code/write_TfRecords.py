import glob
import os
import sys
import nibabel as nib
import json
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from datasets import flip_volume

## Read image paths

dataset_path1 = f'{os.getcwd()}/DATA/dual_regression/ASD'
dataset_path2 = f'{os.getcwd()}/DATA/dual_regression/CONTROL'
out_path = f'{os.getcwd()}/TfRecords/dual_regression/'

if not os.path.isdir(dataset_path1):
    sys.exit(' Dataset ' + subfolder + ' does not exist')

if not os.path.isdir(dataset_path2):
    sys.exit(' Dataset ' + subfolder + ' does not exist')

# volume paths

paths1 = sorted(glob.glob(os.path.join(dataset_path1, '*.nii.gz')))
paths2 = sorted(glob.glob(os.path.join(dataset_path2, '*.nii.gz')))

X = paths1 + paths2

print(len(X))

## Create ground truth labels
y = np.zeros(len(X))


## TF features
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

## Permute data

perm = np.random.permutation(np.arange(len(y)))

X = [X[i] for i in perm]
y = y[perm]


Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, y, test_size = (1 - 0.7))

Xvalid, Xtest, Yvalid, Ytest = train_test_split(Xtest, Ytest, test_size = (0.15/(0.3)))


nImages_train  = len(Ytrain)

nImages_valid = len(Yvalid)

nImages_test = len(Ytest)

print(f'Images in train-->{nImages_train}, valid-->{nImages_valid}, test-->{nImages_test}')

## Determine number of TFRecord files

maxImagesPerFile = 32

folderOut = out_path
if not os.path.exists(folderOut):
    os.makedirs(folderOut)

def compute_n_files(nImages):
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

nImagesPerFileTrain = compute_n_files(nImages_train)
nFilesTrain = len(nImagesPerFileTrain)

nImagesPerFileValid = compute_n_files(nImages_valid)
nFilesValid = len(nImagesPerFileValid)

nImagesPerFileTest = compute_n_files(nImages_test)
nFilesTest = len(nImagesPerFileTest)

print('nFiles TfRecords: %i, %i, %i' % (nFilesTrain, nFilesValid, nFilesTest))


## Write TFRecord files
## Write TFRecord files

#sys.exit()

def write_tfrecord_files(X, Y, nImagesPerFile, fileOutRoot):

    imIndex = 0

    for fileIndex, nImagesFile in enumerate(nImagesPerFile):

        fileName = '%s_%04i.tfrecord' % (fileOutRoot, fileIndex)
        print(fileName)
        print('Done %i files' % imIndex)

        writer = tf.io.TFRecordWriter(fileName)

        for volumeFile, label in zip(X[imIndex:imIndex+nImagesFile], Y[imIndex:imIndex+nImagesFile]):

            # Load volumes
            img = nib.load(volumeFile).get_fdata(dtype='float32')

            #image = image / 60 - 1
            # image = image[:,:,:,np.newaxis]
            #image = np.clip(image, -1, 1)

            # Define features
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



            flipped = flip_volume(img)

            feature_aug = {
                'x_dim': _int64_feature(int(x_dim)),
                'y_dim': _int64_feature(int(y_dim)),
                'z_dim': _int64_feature(int(z_dim)),
                'channels': _int64_feature(int(nChannels)),
                'image': _bytes_feature(tf.compat.as_bytes(flipped.tostring())),
                'label': _int64_feature(int(1))
                }


            tf_example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(tf_example.SerializeToString())

            tf_example_aug = tf.train.Example(features=tf.train.Features(feature=feature_aug))
            writer.write(tf_example_aug.SerializeToString())

        imIndex += nImagesFile

fileOutRoot = os.path.join(folderOut, 'train')
write_tfrecord_files(Xtrain, Ytrain, nImagesPerFileTrain, fileOutRoot)

fileOutRoot = os.path.join(folderOut, 'valid')
write_tfrecord_files(Xvalid, Yvalid, nImagesPerFileValid, fileOutRoot)

fileOutRoot = os.path.join(folderOut, 'test')
write_tfrecord_files(Xtest, Ytest, nImagesPerFileTest, fileOutRoot)

metadata = {
    "Derivatives": f'dual_regression',
    "TfRecords": [{'train':f'{nImages_train*2}','valid':f'{nImages_valid*2}','test':f'{nImages_test}'}],
    "nChannels": f'10',
    "Normalise" : f'None'
    }

with open(f'{out_path}/metadata.txt', 'w') as json_file:
    json.dump(metadata, json_file)
print('TfRecords created!')
