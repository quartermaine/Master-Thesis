import os
import warnings
import glob
import sys
import tqdm
import json
import time
import csv
import datetime
import numpy as np

import keras.backend as K
import tensorflow as tf
import kerastuner as kt
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Conv3D, BatchNormalization, MaxPooling3D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam as Adam
from tensorflow.keras.losses import binary_crossentropy as BC
import matplotlib.pyplot as plt
from custom_utils import plot_history, myDropout
from datasets import flip_3D, rotation_3D
from models import resnet50


K.set_image_data_format('channels_last')
# Ignore FutureWarning from numpy
warnings.simplefilter(action='ignore', category=FutureWarning)

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
# The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"]="0,1";
# Allow growth of GPU memory, otherwise it will always look like all the memory is being used
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
tf.config.experimental.set_memory_growth(physical_devices[1], True)

global nChannels

def parse_example(serialized):
    '''Decode examples stored in TFRecords'''
    features = {
                'x_dim': tf.io.FixedLenFeature([], tf.int64),
                'y_dim': tf.io.FixedLenFeature([], tf.int64),
                'z_dim': tf.io.FixedLenFeature([], tf.int64),
                'channels': tf.io.FixedLenFeature([], tf.int64),
                'image': tf.io.FixedLenFeature([], tf.string),
                'label': tf.io.FixedLenFeature([], tf.int64)}
    # Parse the serialized data so we get a dict with our data.
    parsed_example = tf.io.parse_single_example(serialized=serialized, features=features)

    x_dim = parsed_example['x_dim']
    y_dim = parsed_example['y_dim']
    z_dim = parsed_example['z_dim']
    channels = parsed_example['channels']

    im_shape = [x_dim, y_dim, z_dim, channels]


    label = parsed_example['label']
    image_raw = parsed_example['image']

    image = tf.cast(tf.io.decode_raw(image_raw, tf.float32), tf.float32)
    image = tf.reshape(image, [x_dim, y_dim, z_dim, channels])

    label = tf.cast(label, tf.int64)
    label = tf.reshape(label, [1])

    return image, label


def input_fn(filenames, subset, batch_size, buffer_size=512, data_augmentation=True):
    # Args:
    # filenames:   Filenames for the TFRecords files.
    # subset:      Subset to make either train, valid, test.
    # batch_size:  Return batches of this size.
    # buffer_size: Read buffers of this size. The random shuffling
    #              is done on the buffer, so it must be big enough.

    # Create a TensorFlow Dataset-object which has functionality
    # for reading and shuffling data from TFRecords files.
    dataset = tf.data.TFRecordDataset(filenames=filenames)

    # Parse the serialized data in the TFRecords files.
    # This returns TensorFlow tensors for the image and labels.
    dataset = dataset.map(parse_example, num_parallel_calls=8)

    if subset == 'train' or subset =='valid':
        # Allow infinite reading of the data.
        dataset = dataset.repeat()
    else :
        dataset = dataset.repeat(1)

    if subset == 'train':
        dataset = dataset.shuffle(buffer_size=buffer_size)

    ''' DATA AUGMENTATION '''
    if (subset != 'test' and data_augmentation == True):
        dataset = dataset.map(flip_3D)
        dataset = dataset.map(rotation_3D)

    # Get a batch of data with the given size.
    dataset = dataset.batch(batch_size)

    if subset == 'train':
        dataset = dataset.prefetch(64)

    return dataset

def train_input_fn():
    return input_fn(filenames=path_tfrecords_train, subset='train', batch_size=batch_size)

def valid_input_fn():
    return input_fn(filenames=path_tfrecords_valid, subset='valid', batch_size=batch_size)

def test_input_fn():
    return input_fn(filenames=path_tfrecords_test, subset='test', batch_size=batch_size)


CURRENT_DIR = os.getcwd()
os.system(f'rm -rf {CURRENT_DIR}/results.csv')
# train params
n_GPUs = 2
batch_size = 4 #64 * n_GPUs
learning_rate = 1e-6

with open(f'{CURRENT_DIR}/metadata.txt') as f:
    json_data = json.load(f)


n_im_train = int(json_data["TfRecords"][0]["train"]) # 724

n_im_valid = int(json_data["TfRecords"][0]["valid"]) # 206

n_im_test = int(json_data["TfRecords"][0]["test"]) # 105

nChannels = int(json_data['nChannels'])

# Tensorbord
os.system(f'rm -rf {CURRENT_DIR}/logs/')
logdir = f'{CURRENT_DIR}/logs/fit_{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}'

# Read TfRecords
tfrecord_files_train = sorted(glob.glob(f'{CURRENT_DIR}/train*.tfrecord'))
tfrecord_files_valid = sorted(glob.glob(f'{CURRENT_DIR}/valid*.tfrecord'))
tfrecord_files_test = sorted(glob.glob(f'{CURRENT_DIR}/test*.tfrecord'))

model = resnet50(1, (61, 73, 61, nChannels))
model.compile(loss=BC, optimizer=Adam(lr=learning_rate), metrics=['accuracy'])
model.summary()

dataset_train = input_fn(tfrecord_files_train, 'train', batch_size)
dataset_valid = input_fn(tfrecord_files_valid, 'valid', batch_size)
dataset_test = input_fn(tfrecord_files_test, 'test', batch_size)

## Train
start_time = time.time()
early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy',
                                                                patience=30, restore_best_weights=True, mode='max')
                                                                
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=f'{logdir}', histogram_freq=1)

history = model.fit(dataset_train, epochs=100,
                                            steps_per_epoch=n_im_train//batch_size,
                                            validation_data=dataset_valid,
                                            validation_steps=n_im_valid//batch_size,
                                            callbacks=[early_stopping_callback, tensorboard_callback])
elapsed_time = time.time() - start_time
elapsed_time_string = str(datetime.timedelta(seconds=round(elapsed_time)))
print('Training time: ', elapsed_time_string)

score = model.evaluate(dataset_test, steps=n_im_test//batch_size)
print('Test loss: %.4f' % score[0])
print('Test accuracy: %.4f' % score[1])
