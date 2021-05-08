import os
import warnings
import glob
import sys
import tempfile

import keras.backend as K
import tensorflow as tf
from models import modelT1000

# Ignore FutureWarning from numpy
warnings.simplefilter(action='ignore', category=FutureWarning)

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
# The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"]="0";
# Allow growth of GPU memory, otherwise it will always look like all the memory is being used
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)



def parse_example(serialized, shape=[61, 73, 61, 1]):
    '''Decode examples stored in TFRecords'''

    features = {
                # 'x_dim': tf.io.FixedLenFeature([], tf.int64),
                # 'y_dim': tf.io.FixedLenFeature([], tf.int64),
                # 'z_dim': tf.io.FixedLenFeature([], tf.int64),
                'image_shape': tf.io.FixedLenFeature([], tf.string),
                'image': tf.io.FixedLenFeature([], tf.string),
                'label': tf.io.FixedLenFeature([], tf.int64)}
    # Parse the serialized data so we get a dict with our data.
    parsed_example = tf.io.parse_single_example(serialized=serialized, features=features)

    # x_dim = parsed_example['x_dim']
    # y_dim = parsed_example['y_dim']
    # z_dim = parsed_example['z_dim']
    label = parsed_example['label']
    image_raw = parsed_example['image']


    image = tf.cast(tf.io.decode_raw(image_raw, tf.float64), tf.float32)
    image = tf.reshape(image, shape)

    label = tf.cast(label, tf.int64)

    return image, label


def train_input_fn():
    return input_fn(filenames=path_tfrecords_train, train=True)

def valid_input_fn():
    return input_fn(filenames=path_tfrecords_valid, train=False)

def test_input_fn():
    return input_fn(filenames=path_tfrecords_test, train=False)


def input_fn(filenames, train, batch_size=32, buffer_size=2048):
    # Args:
    # filenames:   Filenames for the TFRecords files.
    # train:       Boolean whether training (True) or testing (False).
    # batch_size:  Return batches of this size.
    # buffer_size: Read buffers of this size. The random shuffling
    #              is done on the buffer, so it must be big enough.

    # Create a TensorFlow Dataset-object which has functionality
    # for reading and shuffling data from TFRecords files.
    dataset = tf.data.TFRecordDataset(filenames=filenames)

    # Parse the serialized data in the TFRecords files.
    # This returns TensorFlow tensors for the image and labels.
    dataset = dataset.map(parse_example)

    if train:
        # If training then read a buffer of the given size and
        # randomly shuffle it.
        dataset = dataset.shuffle(buffer_size=buffer_size)

        # Allow infinite reading of the data.
        num_repeat = None
    else:
        # If testing then don't shuffle the data.

        # Only go through the data once.
        num_repeat = 1

    # Repeat the dataset the given number of times.
    dataset = dataset.repeat(num_repeat)

    # Get a batch of data with the given size.
    dataset = dataset.batch(batch_size)

    # Create an iterator for the dataset and the above modifications.
    # iterator = dataset.make_one_shot_iterator()

    # Get the next batch of images and labels.
    # images_batch, labels_batch = iterator.get_next()

    images_batch, labels_batch = tf.data.experimental.get_single_element(dataset.take(1))
    # The input-function must return a dict wrapping the images.
    x = {'image': images_batch}
    y = labels_batch

    return x, y

path_tfrecords_train = sorted(glob.glob(f'{os.getcwd()}/ENCODED_DATA/train*'))
path_tfrecords_valid = sorted(glob.glob(f'{os.getcwd()}/ENCODED_DATA/valid*'))
path_tfrecords_test = sorted(glob.glob(f'{os.getcwd()}/ENCODED_DATA/test*'))

model_dir = tempfile.mkdtemp()
keras_estimator = tf.keras.estimator.model_to_estimator(
    keras_model=modelT1000(), model_dir=model_dir)

train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=1000)
eval_spec = tf.estimator.EvalSpec(input_fn=valid_input_fn)

tf.estimator.train_and_evaluate(keras_estimator, train_spec, eval_spec)

# keras_estimator.train(input_fn=train_input_fn, steps=500)
# eval_result = keras_estimator.evaluate(input_fn=valid_input_fn, steps=10)
# print('Eval result: {}'.format(eval_result))
