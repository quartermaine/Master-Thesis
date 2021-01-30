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

def input_fn(filenames, subset, batch_size, buffer_size=512):
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
    dataset = dataset.map(parse_example, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    if subset == 'train' or subset =='valid':
        # Allow infinite reading of the data.
        dataset = dataset.repeat()
    else :
        dataset = dataset.repeat(1)

    if subset == 'train':
        dataset = dataset.shuffle(buffer_size=buffer_size)

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
# CNN params
nFilts = [32, 64, 128]
dropoutRates = [0.5, 0.6]
denseNodes = [25, 50, 100, 200]
doubleFirst = True
convLayers = [4, 5, 6]
learning_rate = 0.0001

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

# Loop over settings

file = open('results.csv', 'w', newline='')
with file :
    header = ['convLayers', 'nFilts', 'dropoutRate', 'denseNodes', 'loss', 'accuracy']
    writer = csv.DictWriter(file, fieldnames=header)
    writer.writeheader()

    for convLayer in convLayers:
        for nFilt in nFilts:
            for dropoutRate in dropoutRates:
                for dense in denseNodes:
                    strategy = tf.distribute.MirroredStrategy()
                    with strategy.scope():
                        NAME = f'{convLayer}-conv-{nFilt}-filtrs-{dropoutRate}-dropout-{dense}-dense'

                        input = Input(shape=[61,73,61,nChannels])
                        x = input

                        x = Conv3D(nFilt, kernel_size=(3,3,3), activation='relu', padding='same',
                                                strides = (1, 1, 1), kernel_regularizer = tf.keras.regularizers.l2(l=0.01))(x)
                        x = BatchNormalization()(x)


                        if doubleFirst:
                            x = Conv3D(nFilt, kernel_size=(3,3,3), activation='relu', padding='same',
                                                strides = (1, 1, 1), kernel_regularizer = tf.keras.regularizers.l2(l=0.01))(x)
                            x = BatchNormalization()(x)

                        x = MaxPooling3D(pool_size=(2,2,2))(x)

                        for i in range(convLayer):
                        # x = Conv3D(nFilt*(2**(i+1)), kernel_size=(3,3,3), activation='relu', padding='same')(x)
                            x = Conv3D(nFilt*(2**(i+1)), kernel_size=(3,3,3), activation='relu', padding='same',
                                                strides = (1, 1, 1), kernel_regularizer = tf.keras.regularizers.l2(l=0.01))(x)
                            x = BatchNormalization()(x)
                            x = MaxPooling3D(pool_size=(2,2,2))(x)

                        x = Flatten()(x)

                        for _ in range(1):
                            x = Dense(dense, activation='relu')(x)
                            x = BatchNormalization()(x)
                            x = Dropout(dropoutRate)(x)

                        output = Dense(1, activation='sigmoid')(x)

                        model = Model(inputs=input, outputs=output)
                        model.compile(loss=BC, optimizer=Adam(lr=learning_rate), metrics=['accuracy'])


                        print(f'Training model with parameters {NAME}')
                        model.summary()

                        dataset_train = input_fn(tfrecord_files_train, 'train', batch_size)
                        dataset_valid = input_fn(tfrecord_files_valid, 'valid', batch_size)
                        dataset_test = input_fn(tfrecord_files_test, 'test', batch_size)

                        ## Train
                        start_time = time.time()
                        early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True, mode='max')
                        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=f'{logdir}/{NAME}', histogram_freq=1)

                        try:

                            history = model.fit(dataset_train, epochs=30,
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

                            # Save results to text file
                            # accuracy = np.zeros((1,1))
                            writer.writerow({'convLayers': convLayer ,
                                'nFilts': nFilt, 'dropoutRate': dropoutRate,
                                'denseNodes': dense, 'loss': score[0],
                                'accuracy':score[1]})

                        except:

                            print('Training failed reporting zero accuracy/loss')
                            # accuracy = np.zeros((1,1))
                            accuracy = 0.0
                            writer.writerow({'convLayers': convLayer ,
                                'nFilts': nFilt, 'dropoutRate': dropoutRate,
                                'denseNodes': dense, 'loss': 0.0,
                                'accuracy':0.0})
                        finally:

                            del x
                            del model
                            del strategy
                            del dataset_train
                            del dataset_valid
                            del dataset_test
