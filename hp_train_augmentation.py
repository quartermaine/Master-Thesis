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
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
import tensorflow_addons as tfa
from tensorboard.plugins.hparams import api as hp

from tensorflow.keras.optimizers import Adam, Nadam
from tensorflow.keras.losses import binary_crossentropy as BC
import matplotlib.pyplot as plt
from custom_utils import plot_history, myDropout
from datasets import flip3D, rotation3D, center_crop3D, blur3D, elastic3D
from resnet_blocks import identity_block, conv_block

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

    dataset = dataset.map(parse_example, num_parallel_calls = tf.data.experimental.AUTOTUNE).cache()

    if subset == 'train' or subset =='valid':
        # Allow infinite reading of the data.
        dataset = dataset.repeat()
    else :
        dataset = dataset.repeat(1)

    if subset != 'test':
        dataset = dataset.shuffle(buffer_size=buffer_size)

    ''' DATA AUGMENTATION '''
    if (subset != 'test' and data_augmentation == True):
        dataset = dataset.map(elastic3D,  num_parallel_calls = tf.data.experimental.AUTOTUNE)
        dataset = dataset.map(flip3D,  num_parallel_calls = tf.data.experimental.AUTOTUNE)
        dataset = dataset.map(rotation3D,  num_parallel_calls = tf.data.experimental.AUTOTUNE)
        dataset = dataset.map(blur3D,  num_parallel_calls = tf.data.experimental.AUTOTUNE)


    # Get a batch of data with the given size.
    dataset = dataset.batch(batch_size)

    if subset == 'train':
        dataset = dataset.prefetch(10)

    return dataset


def train_input_fn():
    return input_fn(filenames=path_tfrecords_train, subset='train', batch_size=batch_size)

def valid_input_fn():
    return input_fn(filenames=path_tfrecords_valid, subset='valid', batch_size=batch_size)

def test_input_fn():
    return input_fn(filenames=path_tfrecords_test, subset='test', batch_size=batch_size)



CURRENT_DIR = os.getcwd()

with open(f'{CURRENT_DIR}/metadata.txt') as f:
    json_data = json.load(f)


# train params
n_GPUs = 2
batch_size = 4 # 64 * n_GPUs
n_im_train = int(json_data["TfRecords"][0]["train"]) # 724
n_im_valid = int(json_data["TfRecords"][0]["valid"]) # 206
n_im_test = int(json_data["TfRecords"][0]["test"]) # 105
nChannels = int(json_data['nChannels'])

# CNN params
HP_LAYERS = hp.HParam('cnn_layers', hp.Discrete([4, 5, 6]))
HP_FILTER = hp.HParam('num_filters', hp.Discrete([32 , 64, 128]))
HP_DROPOUT = hp.HParam('dropout_rate', hp.Discrete([0.5, 0.6]))
HP_DENSE = hp.HParam('num_dense', hp.Discrete([256, 512, 10, 200, 500]))
# HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adam', 'Nadam']))

METRIC_ACCURACY = 'accuracy'
METRIC_LOSS = 'loss'
# METRIC_F1 = 'f1'

# Clear log previous file
os.system(f'rm -rf {CURRENT_DIR}/logs/')
os.system(f'rm -rf {CURRENT_DIR}/results.csv')
print('Removing previous log file\n')
logdir = f'{CURRENT_DIR}/logs/fit_{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}'
# hpdir = os.path.join(logdir, 'validation')
#
with tf.summary.create_file_writer(logdir).as_default():
  hp.hparams_config(
    hparams=[HP_LAYERS, HP_FILTER, HP_DROPOUT, HP_DENSE],
    metrics=[hp.Metric(METRIC_ACCURACY, display_name='Accuracy'),
            hp.Metric(METRIC_LOSS, display_name='Loss')])


# Read TfRecords
path_tfrecords_train = sorted(glob.glob(f'{CURRENT_DIR}/train*.tfrecord'))
path_tfrecords_valid = sorted(glob.glob(f'{CURRENT_DIR}/valid*.tfrecord'))
path_tfrecords_test = sorted(glob.glob(f'{CURRENT_DIR}/test*.tfrecord'))

train_dataset = train_input_fn()
valid_dataset = valid_input_fn()
test_dataset = test_input_fn()

def train_test_model(logdir, hparams):
    input = keras.Input(shape=[61,73,61,nChannels])
    x = input

    x = layers.Conv3D(hparams[HP_FILTER], kernel_size=(3,3,3), activation='relu', padding='same',
                         kernel_initializer = 'he_normal')(x)
    x = layers.BatchNormalization()(x)

    x = layers.MaxPooling3D(pool_size=(2,2,2), padding = 'same')(x)

    for i in range(hparams[HP_LAYERS]):

        # filter = hparams[HP_FILTER]*(2**(i+1))
        #
        # x = conv_block(x, 3, [filter, filter, 4*filter], strides=(1, 1, 1))
        # x = identity_block(x, 3, [filter, filter, 4*filter])
        # x = identity_block(x, 3, [filter, filter, 4*filter])

        # layer 1
        x = layers.Conv3D(hparams[HP_FILTER]*(2**(i+1)), kernel_size=(3,3,3), activation='relu', padding='same',
                            kernel_initializer = 'he_normal')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling3D(pool_size=(2,2,2), padding='same')(x)

        # layers 2
        x = layers.Conv3D(hparams[HP_FILTER]*(2**(i+1)), kernel_size=(3,3,3), activation='relu', padding='same',
                                kernel_initializer = 'he_normal')(x)

        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling3D(pool_size=(2,2,2), padding='same')(x)

    x = layers.Flatten()(x)

    for i in range(1):
        x = layers.Dense(hparams[HP_DENSE], activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(hparams[HP_DROPOUT])(x)

    output = layers.Dense(1, activation='sigmoid')(x)

    model = Model(inputs=input, outputs=output)
    model.compile(loss=BC, optimizer=Adam(lr=0.0001), metrics=['accuracy'], run_eagerly=True)

    model.summary()
    NAME = f'{hparams[HP_LAYERS]}-conv-{hparams[HP_FILTER]}-filtrs-{hparams[HP_DROPOUT]}-dropout-{hparams[HP_DENSE]}-dense'
      ## Train
    start_time = time.time()
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True, mode='max')
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=f'{logdir}/{NAME}', histogram_freq=1)
    hp_callback = hp.KerasCallback(logdir, hparams)

    history = model.fit(train_dataset,
                              epochs=100,
                              steps_per_epoch=n_im_train//batch_size,
                              validation_data=valid_dataset,
                              validation_steps=n_im_valid//batch_size,
                              callbacks=[early_stopping_callback, tensorboard_callback, hp_callback])

    elapsed_time = time.time() - start_time
    elapsed_time_string = str(datetime.timedelta(seconds=round(elapsed_time)))
    print('Training time: ', elapsed_time_string)
    mc_accuracies, mc_losses = MC_eval(model, 100, dataset_test, n_im_test//batch_size)
    print('Test loss: %.4f' % np.mean(mc_losses))
    print('Test accuracy: %.4f' % np.mean(mc_accuracies))

    return (mc_losses, mc_accuracies)

def run(run_dir, hparams):
    with tf.summary.create_file_writer(run_dir).as_default():
        hp.hparams(hparams)  # record the values used in this trial
        scores= train_test_model(logdir, hparams)
        tf.summary.scalar(METRIC_ACCURACY,np.mean(scores[0]), step=1)
        tf.summary.scalar(METRIC_LOSS, np.mean(scores[1]), step=1)

    return scores

best_accuracy = 0
file = open('results.csv', 'w', newline='')
with file :
    header = ['conv_layers', 'filters', 'dropout', 'dense', 'loss', 'accuracy']
    writer = csv.DictWriter(file, fieldnames=header)
    writer.writeheader()

    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        session_num = 0
        for num_layers in HP_LAYERS.domain.values:
            for num_filters in HP_FILTER.domain.values:
                for dropout_rate in HP_DROPOUT.domain.values:
                    for num_dense in HP_DENSE.domain.values:
                        hparams = {
                        HP_LAYERS: num_layers,
                        HP_FILTER: num_filters,
                        HP_DROPOUT: dropout_rate,
                        HP_DENSE : num_dense
                        }
                        run_name = "run-%d" % session_num
                        print('--- Starting trial: %s' % run_name)
                        print({h.name: hparams[h] for h in hparams})
                        current_params = [hparams[h] for h in hparams]
                        current_scores = run('logs/hparam_tuning/' + run_name, hparams)
                        if np.mean(current_scores[1])>best_accuracy:
                            print('Best mean accuracy changed writing MC accuracies to file mc_accuracies')
                            np.save('mc_accuracies', scores[1])
                        writer.writerow({'conv_layers': current_params[0],
                                'filters': current_params[1], 'dropout': current_params[2],
                                'dense': current_params[3], 'loss': np.mean(current_scores[0]),
                                'accuracy': np.mean(current_score[1])})
                        session_num += 1
