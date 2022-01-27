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
import multiprocessing as mp

# import keras
from tensorflow.keras import backend as K
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa
# from tensorflow.keras.utils import plot_model

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Input, Conv3D, BatchNormalization, MaxPooling3D, Flatten, Dense,
    Dropout, ZeroPadding3D, SpatialDropout3D, GlobalAveragePooling3D )
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.optimizers import Adam as Adam
from tensorflow.keras.losses import binary_crossentropy as BC
import matplotlib.pyplot as plt
from custom_utils import plot_history, myDropout, MC_eval
from datasets import flip3D, rotation3D, blur3D, center_crop3D, elastic3D


# tf.config.optimizer.set_jit(True)
# set the channels last format on the neural network
K.set_image_data_format('channels_last')
# Ignore FutureWarning from numpy
warnings.simplefilter(action='ignore', category=FutureWarning)

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
# The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"]="0, 1";
# Allow growth of GPU memory, otherwise it will always look like all the memory is being used
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
tf.config.experimental.set_memory_growth(physical_devices[1], True)
os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'

# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#   # Create 2 virtual GPUs with 1GB memory each
#   try:
#     tf.config.experimental.set_virtual_device_configuration(
#         gpus[0],
#         [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5000),
#          tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5000)])
#     logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#     print(len(gpus), "Physical GPU,", len(logical_gpus), "Logical GPUs")
#   except RuntimeError as e:
#     # Virtual devices must be set before GPUs have been initialized
#     print(e)

os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'

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
    # get the dimensions stored in the TFRecord
    x_dim = parsed_example['x_dim']
    y_dim = parsed_example['y_dim']
    z_dim = parsed_example['z_dim']
    channels = parsed_example['channels']
    # set image shape
    im_shape = [x_dim, y_dim, z_dim, channels]
    # get raw image, label
    label = parsed_example['label']
    image_raw = parsed_example['image']
    # decode the raw image, label
    image = tf.cast(tf.io.decode_raw(image_raw, tf.float32), tf.float32)
    image = tf.reshape(image, [x_dim, y_dim, z_dim, channels])

    label = tf.cast(label, tf.int64)
    label = tf.reshape(label, [1,])

    return image, label


def input_fn(filenames, subset, batch_size, buffer_size=256, data_augmentation=False):
    # Args:
    # filenames:   Filenames for the TFRecords files.
    # subset:      Subset to make either train, valid, test.
    # batch_size:  Return batches of this size.
    # buffer_size: Read buffers of this size. The random shuffling
    #              is done on the buffer, so it must be big enough.
    # data_augmentation : to perform augmentation on the fly or not

    # Create a TensorFlow Dataset-object which has functionality
    # for reading and shuffling data from TFRecords files.

    # dataset = tf.data.Dataset.from_tensor_slices(filenames)
    #
    # dataset = dataset.interleave(lambda x: tf.data.TFRecordDataset(x),
    #                              cycle_length=4,
    #                              num_parallel_calls=AUTO,
    #                              deterministic=False)


    #AUTO = tf.data.experimental.AUTOTUNE
    nCores = mp.cpu_count()
    ignore_order = tf.data.Options()
    ignore_order.experimental_deterministic = False

    dataset = tf.data.TFRecordDataset(filenames=filenames)
    dataset = dataset.with_options(ignore_order)
    # # Parse the serialized data in the TFRecords files.
    # # This returns TensorFlow tensors for the image and labels.
    dataset = dataset.map(parse_example, num_parallel_calls = nCores)


    # make the training dataset to iterate forever
    if subset == 'train':
        dataset = dataset.repeat()

    # shuffle the training dataset
    if subset != 'test':
        dataset = dataset.shuffle(buffer_size=buffer_size)

    # ''' PATCH EXTRACTION '''
    # if dataset_type == 'test':
    #     dataset = dataset.map(lambda image, label: patch_extraction_tf(image, label, sizePatches=patch_size, Npatches=1, zero_centered=True))
    # else:
    #     dataset = dataset.map(lambda image, label: patch_extraction_tf(image, label, sizePatches=patch_size, Npatches=1, zero_centered=False))

    # dataset = dataset.map(parse_example, num_parallel_calls = AUTO)

    ''' DATA AUGMENTATION '''
    if (subset != 'test' and data_augmentation == True):
        dataset = dataset.map(elastic3D,  num_parallel_calls = nCores)
        dataset = dataset.map(flip3D,  num_parallel_calls = nCores)
        dataset = dataset.map(rotation3D,  num_parallel_calls = nCores)
        dataset= dataset.map(blur3D,  num_parallel_calls = nCores)

    # set bach_size
    dataset = dataset.batch(batch_size=batch_size)

    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return dataset

def train_input_fn():
    return input_fn(filenames=path_tfrecords_train, subset='train', batch_size=batch_size)

def valid_input_fn():
    return input_fn(filenames=path_tfrecords_valid, subset='valid', batch_size=batch_size)

def test_input_fn():
    return input_fn(filenames=path_tfrecords_test, subset='test', batch_size=batch_size)

def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.1)


CURRENT_DIR = os.getcwd()

# Clear results previous file
os.system(f'rm -rf {CURRENT_DIR}/results.csv')
# Clear log previous file
os.system(f'rm -rf {CURRENT_DIR}/logs/')
logdir = f'{CURRENT_DIR}/logs/fit_{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}'
# train params
n_GPUs = 2
batch_size = 12 #64 * n_GPUs
# CNN params
nFilts = [32, 64, 128]
dropoutRates = [0.4, 0.5, 0.6]
denseNodes = [128, 256, 512, 1024]
doubleFirst = True
convLayers = [2, 3, 4]
learning_rate = 1e-5

# open the metadata file writern when writing TfRecords
with open(f'{CURRENT_DIR}/metadata.txt') as f:
    json_data = json.load(f)

# get derivative names
res = json_data['Derivatives'].strip('][''').split(', ')
res = [x.replace("'", "") for x in res]
derivative_names = '_'.join(res)

# number of train images
n_im_train = int(json_data["TfRecords"][0]["train"]) # 724
# number of valid images
n_im_valid = int(json_data["TfRecords"][0]["valid"]) # 206
# number of test images
n_im_test = int(json_data["TfRecords"][0]["test"]) # 105
# number of channels
nChannels = int(json_data['nChannels'])

# Read TfRecords paths
tfrecord_files_train = sorted(glob.glob(f'{CURRENT_DIR}/train*.tfrecord'))
tfrecord_files_valid = sorted(glob.glob(f'{CURRENT_DIR}/valid*.tfrecord'))
tfrecord_files_test = sorted(glob.glob(f'{CURRENT_DIR}/test*.tfrecord'))
# begin training
best_accuracy = 0
# create a csv to store the results of accuracy, loss for every combination
file = open('results.csv', 'w', newline='')
with file :
    header = ['convLayers', 'nFilts', 'dropoutRate', 'denseNodes', 'loss', 'accuracy', 'n_params', 'train_time']
    writer = csv.DictWriter(file, fieldnames=header)
    writer.writeheader()
    # Loop over settings
    for convLayer in convLayers:
        for nFilt in nFilts:
            for dropoutRate in dropoutRates:
                for dense in denseNodes:
                    strategy = tf.distribute.MirroredStrategy()
                    with strategy.scope():
                        NAME = f'{convLayer}-conv-{nFilt}-filtrs-{dropoutRate}-dropout-{dense}-dense'
                        # tf.keras.backend.clear_session()
                        input = Input(shape=[61,73,61,nChannels])
                        x = input

                        x = Conv3D(nFilt,
                                   kernel_size = (3, 3, 3),
                                   padding='same',
                                   activation = 'relu',
                                   kernel_initializer = 'he_normal', name = 'conv1')(x)
                        x = BatchNormalization()(x)
                        x = MaxPooling3D(pool_size = (2, 2, 2))(x)


                        for i in range(convLayer):

                            # filter = nFilt*(2**(i+1))
                            # x = conv_block(x, 3, [filter, filter, filter], i+1, 'a', s = 2)
                            # x = identity_block(x, 3, [filter, filter, filter], i+1, 'b')

                            # conv1
                            x = Conv3D(nFilt*(2**(i+1)),
                                       kernel_size = (3, 3, 3),
                                       padding='same',
                                       activation = 'relu',
                                        kernel_initializer = 'he_normal', name = 'conv2_' + str(i))(x)
                            x = BatchNormalization()(x)
                            x = MaxPooling3D(pool_size= (2, 2, 2))(x)

                            # conv2 |<------ this block did not used in thesis due to memory error ------>|
                            x = Conv3D(nFilt*(2**(i+1)),
                                       kernel_size = (3, 3, 3),
                                       padding='same',
                                       activation = 'relu',
                                        kernel_initializer = 'he_normal', name ='conv3_' + str(i))(x)
                            x = BatchNormalization()(x)
                            x = MaxPooling3D(pool_size= (2, 2, 2))(x)

                        # x = Conv3D(2, (3, 3), activation = 'relu', padding = 'same')(x) # used in different model 
                        x = GlobalAveragePooling3D()(x)
                        # x = Activation('softmax')(x) # used in different model 
                        x = Flatten()(x)

                        for _ in range(1):
                             x = Dense(dense, activation='relu', name ='dense')(x)
                             x = BatchNormalization()(x)
                             x = myDropout(dropoutRate, name = 'mydropout')(x) # MC dropout

                        output = Dense(1, activation='sigmoid', name = 'out')(x)

                        model = Model(inputs=input, outputs=output)
                        model.compile(loss=BC,
                                      optimizer=Adam(lr=learning_rate),
                                      metrics=['accuracy'])


                        print(f'Training model with parameters {NAME}')
                        model.summary()
                        n_params = model.count_params()

                        dataset_train = input_fn(tfrecord_files_train, 'train', batch_size)
                        dataset_valid = input_fn(tfrecord_files_valid, 'valid', batch_size)
                        dataset_test = input_fn(tfrecord_files_test, 'test', batch_size)

                        ## Train
                        start_time = time.time()

                        early_stopping_callback = tf.keras.callbacks.EarlyStopping(
                            monitor='val_accuracy',
                            patience=15,
                            restore_best_weights=True,
                            mode='max')

                        tensorboard_callback = tf.keras.callbacks.TensorBoard(
                            log_dir=f'{logdir}/{NAME}',
                            histogram_freq=1)

                        lr_callback = tf.keras.callbacks.LearningRateScheduler(
                            scheduler)

                        try:

                            history = model.fit(dataset_train,
                                                epochs = 200,
                                                batch_size = batch_size,
                                                steps_per_epoch=n_im_train//batch_size,
                                                validation_data=dataset_valid,
                                                validation_steps=n_im_valid//batch_size,
                                                callbacks=[early_stopping_callback,
                                                tensorboard_callback],
                                                shuffle =True)

                            elapsed_time = time.time() - start_time
                            elapsed_time_string = str(datetime.timedelta(seconds=round(elapsed_time)))
                            print('Training time: ', elapsed_time_string)

                            # score = model.evaluate(dataset_test, steps=n_im_test//batch_size)
                            mc_accuracies, mc_losses = MC_eval(model, 1000, dataset_test, n_im_test//batch_size)

                            print('Test loss: %.4f' % np.mean(mc_losses))
                            print('Test accuracy: %.4f' % np.mean(mc_accuracies))

                            if np.mean(mc_accuracies)>best_accuracy:
                                print('Best mean accuracy changed writing MC accuracies to file mc_accuracies')
                                np.save(f'mc_accuracies_{derivative_names}', mc_accuracies)
                                model.save(f'best_model_{derivative_names}.h5')
                                # plot_model(model, to_file=f'best_model_plot_{derivative_names}.png') # used to save png of best model

                            # Save results to text file
                            # accuracy = np.zeros((1,1))
                            writer.writerow({'convLayers': convLayer ,
                                             'nFilts': nFilt,
                                             'dropoutRate': dropoutRate,
                                             'denseNodes': dense,
                                             'loss': np.mean(mc_losses),
                                             'accuracy': np.mean(mc_accuracies),
                                             'n_params': n_params,
                                             'train_time': elapsed_time_string})

                        except:
                            print('Training failed reporting zero accuracy/loss')
                            accuracy = np.zeros((1,1))
                            accuracy = 0.0
                            writer.writerow({'convLayers': convLayer ,
                                            'nFilts': nFilt,
                                            'dropoutRate': dropoutRate,
                                            'denseNodes': dense,
                                            'loss': 0.0,
                                            'accuracy':0.0,
                                            'n_params': n_params,
                                            'train_time': 0.0})

                        finally:
                            del x
                            del model
                            del strategy
                            del dataset_train
                            del dataset_valid
                            del dataset_test
