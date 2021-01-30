import os
import warnings
import glob
import sys
import tqdm
import json
import datetime
import numpy as np

import keras.backend as K
import tensorflow as tf
import kerastuner as kt
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa

from tensorflow.keras.optimizers import Adam, Nadam
from tensorflow.keras.losses import binary_crossentropy as BC
import matplotlib.pyplot as plt
from custom_utils import plot_history


# Ignore FutureWarning from numpy
warnings.simplefilter(action='ignore', category=FutureWarning)

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
# The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"]="0,1";
# Allow growth of GPU memory, otherwise it will always look like all the memory is being used
physical_devices = tf.config.experimental.list_physical_devices('GPU')
print("Num GPUs Available: ", len(physical_devices))
# tf.config.experimental.set_memory_growth(physical_devices[0], True)

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

    image = tf.cast(tf.io.decode_raw(image_raw, tf.float64), tf.float64)
    image = tf.reshape(image, [x_dim, y_dim, z_dim, channels])

    label = tf.cast(label, tf.int64)
    label = tf.reshape(label, [1])

    return image, label

def input_fn(filenames, subset, batch_size=32, buffer_size=2048):
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

    if subset == 'train':
        dataset = dataset.shuffle(buffer_size=2048)

    # Get a batch of data with the given size.
    dataset = dataset.batch(batch_size)

    if subset == 'train':
        dataset = dataset.prefetch(64)

    return dataset

def train_input_fn():
    return input_fn(filenames=path_tfrecords_train, subset='train')

def valid_input_fn():
    return input_fn(filenames=path_tfrecords_valid, subset='valid')

def test_input_fn():
    return input_fn(filenames=path_tfrecords_test, subset='test')


def build_model(hp):
    """Builds a convolutional model with hyperparameters"""

    hp_activations = hp.Choice('activation' , ['relu', 'elu'])
    hp_dense_layer = hp.Choice('dense', [25, 50, 100, 200, 500])
    hp_droput_rate = hp.Choice('droput_rate', [0.5, 0.6])
    hp_filters = hp.Int('filters_cnn', 8, 32, step =4, default=4)

    # define model
    input = keras.Input(shape=[61, 73, 61, nChannels])
    x = input

    x = layers.Conv3D(filters=4, kernel_size=(3,3,3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)

    for i in range(hp.Int('conv_layers', 2, 8, step=2, default=4)):
        x = layers.Conv3D(
            filters= hp_filters,
            kernel_size=(3, 3, 3),
            activation= 'relu',
            padding='same')(x)

        x = layers.BatchNormalization()(x)
        x = layers.MaxPool3D(pool_size=(2,2,2), padding='same')(x)


        # if hp.Choice('pooling' + str(i), ['max', 'avg']) == 'max':
        #     x = tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 2))(x)
        # else:
        #     x = tf.keras.layers.AveragePooling3D(pool_size=(2, 2, 2))(x)

    # if hp.Choice('global_pooling', ['max', 'avg']) == 'max':
    #     x = tf.keras.layers.GlobalMaxPooling3D()(x)
    # else:
    #     x = tf.keras.layers.GlobalAveragePooling3D()(x)

    x = layers.Flatten()(x)

    for i in range(1):

        x = layers.Dense(units=hp_dense_layer, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(rate =hp_learning_rate)(x)

    output = layers.Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model(input, output)
    model.summary()

    hp_learning_rate = hp.Choice('learning_rate', values = [1e-2, 1e-3, 1e-4])

    model.compile(keras.optimizers.Adam(learning_rate = hp_learning_rate),
                    loss='binary_crossentropy', metrics=['accuracy'])

    return model

###############################################################################

CURRENT_DIR = os.getcwd()

n_GPUs = 2
# batch_size = 64 * n_GPUs

# get the number of channels from txt file
with open(f'{CURRENT_DIR}/metadata.txt') as f:
    json_data = json.load(f)

nChannels = int(json_data['nChannels'])

# Read TfRecords
path_tfrecords_train = sorted(glob.glob(f'{CURRENT_DIR}/train*.tfrecord'))
path_tfrecords_valid = sorted(glob.glob(f'{CURRENT_DIR}/valid*.tfrecord'))
path_tfrecords_test = sorted(glob.glob(f'{CURRENT_DIR}/test*.tfrecord'))

train_dataset = train_input_fn()
validation_dataset = valid_input_fn()
test_dataset = test_input_fn()

os.system(f'rm -rf {CURRENT_DIR}/logs/')
 # log_dir = f'{CURRENT_DIR}/logs'
log_dir = f'{CURRENT_DIR}/logs/fit_{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}'

# Early stopping
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir, histogram_freq=1, write_graph=True)

# Train
tuner = kt.Hyperband(
    hypermodel=build_model,
    objective='val_accuracy',
    max_epochs=10,
    factor=2,
    hyperband_iterations=3,
    distribution_strategy=tf.distribute.MirroredStrategy(),
    directory=log_dir,
    project_name='hp_turner')


tuner.search(train_dataset,
                 steps_per_epoch=600,
                 validation_data=validation_dataset,
                 validation_steps=100,
                 epochs=3,
                 callbacks=[early_stopping_callback, tensorboard_callback])
