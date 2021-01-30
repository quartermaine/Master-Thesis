import keras.backend as K
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def identity_block(input_tensor, kernel_size, filters):

    filters1, filters2, filters3 = filters
    # if K.image_data_format() == 'channels_last':
    #     bn_axis = 3
    # else:
    #     bn_axis = 1

    x = layers.Conv3D(filters1, (1, 1, 1), use_bias=False,
                      kernel_initializer='he_normal')(input_tensor)

    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv3D(filters2, kernel_size,
                      padding='same', use_bias=False,
                      kernel_initializer='he_normal')(x)

    x = layers.BatchNormalization()(x)

    x = layers.Activation('relu')(x)

    x = layers.Conv3D(filters3, (1, 1, 1), use_bias=False,
                      kernel_initializer='he_normal')(x)

    x = layers.BatchNormalization()(x)

    x = layers.add([x, input_tensor])
    x = layers.Activation('relu')(x)
    return x

def conv_block(input_tensor, kernel_size, filters, strides=(2, 2, 2)):

    filters1, filters2, filters3 = filters

    # if K.image_data_format() == 'channels_last':
    #     bn_axis = 3
    # else:
    #     bn_axis = 1

    # conv1
    x = layers.Conv3D(filters1, (1, 1, 1), use_bias=False,
                      kernel_initializer='he_normal')(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    # conv 2
    x = layers.Conv3D(filters2, kernel_size, strides=strides, padding='same',
                      use_bias=False, kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    # conv 3
    x = layers.Conv3D(filters3, (1, 1, 1), use_bias=False,
                      kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    # conv 4
    shortcut = layers.Conv3D(filters3, (1, 1, 1), strides=strides, use_bias=False,
                             kernel_initializer='he_normal')(input_tensor)
    shortcut = layers.BatchNormalization()(shortcut)

    x = layers.add([x, shortcut])
    x = layers.Activation('relu')(x)
    return x


def resnet50(num_classes, input_shape):
    img_input = layers.Input(shape=input_shape)

    # if K.image_data_format() == 'channels_first':
    #     x = layers.Lambda(lambda x: backend.permute_dimensions(x, (0, 3, 1, 2)),
    #                       name='transpose')(img_input)
    #     bn_axis = 1
    # else:  # channels_last
    #     x = img_input
    #     bn_axis = 3

    # Conv1 (7x7,64,stride=2)
    x = layers.ZeroPadding3D(padding=(3, 3, 3))(x)

    x = layers.Conv3D(64, (7, 7, 7),
                      strides=(2, 2, 2),
                      padding='valid', use_bias=False,
                      kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.ZeroPadding3D(padding=(1, 1, 1))(x)

    # 3x3 max pool,stride=2
    x = layers.MaxPooling3D((3, 3, 1), strides=(2, 2, 2))(x)

    # Conv2_x

    # 1×?1, 64
    # 3×?3, 64
    # 1×?1, 256

    x = conv_block(x, 3, [64, 64, 256], strides=(1, 1, 1))
    x = identity_block(x, 3, [64, 64, 256])
    x = identity_block(x, 3, [64, 64, 256])

    # Conv3_x
    #
    # 1×?1, 128
    # 3×?3, 128
    # 1×?1, 512

    x = conv_block(x, 3, [128, 128, 512])
    x = identity_block(x, 3, [128, 128, 512])
    x = identity_block(x, 3, [128, 128, 512])
    x = identity_block(x, 3, [128, 128, 512])

    # Conv4_x
    # 1×?1, 256
    # 3×?3, 256
    # 1×?1, 1024
    x = conv_block(x, 3, [256, 256, 1024])
    x = identity_block(x, 3, [256, 256, 1024])
    x = identity_block(x, 3, [256, 256, 1024])
    x = identity_block(x, 3, [256, 256, 1024])
    x = identity_block(x, 3, [256, 256, 1024])
    x = identity_block(x, 3, [256, 256, 1024])

    # 1×?1, 512
    # 3×?3, 512
    # 1×?1, 2048
    x = conv_block(x, 3, [512, 512, 2048])
    x = identity_block(x, 3, [512, 512, 2048])
    x = identity_block(x, 3, [512, 512, 2048])

    # average pool
    x = layers.GlobalAveragePooling3D()(x)
    x = layers.Dense(
        num_classes, activation='sigmoid')(x)

    # Create model.
    return keras.Model(img_input, x, name='resnet50')
