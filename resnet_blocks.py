from tensorflow.keras import layers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Input, Add, Activation, Conv3D, BatchNormalization, MaxPooling3D, Flatten, Dense,
    Dropout, ZeroPadding3D, SpatialDropout3D, GlobalAveragePooling3D )
# from keras.models import Model, load_model
# from keras.utils import layer_utils
# from keras.utils.data_utils import get_file
# from keras.applications.imagenet_utils import preprocess_input
# from keras.utils.vis_utils import model_to_dot
from tensorflow.keras.initializers import glorot_uniform

# import keras.backend as K
# K.set_image_data_format('channels_last')
# K.set_learning_phase(1)


def identity_block(X, f, filters, stage, block):
    """
    Implementation of the identity block

    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_D_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network

    Returns:
    X -- output of the identity block, tensor of shape (n_H, n_W, n_D, n_C)
    """

    # defining names
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # Retrieve Filters
    F1, F2, F3 = filters

    # Save the input value. You'll need this later to add back to the main path.
    X_shortcut = X

    # First component of main path
    X = Conv3D(filters = F1, kernel_size = (1, 1, 1),
               strides = (1, 1, 1), padding = 'valid',
               name = conv_name_base + '2a', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 4, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)


    # Second component of main path (??3 lines)
    X = Conv3D(filters = F2, kernel_size = (f, f, f),
               strides = (1, 1, 1), padding = 'same',
               name = conv_name_base + '2b', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 4, name = bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third component of main path (??2 lines)
    X = Conv3D(filters = F3, kernel_size = (1, 1, 1),
               strides = (1, 1, 1), padding = 'valid',
               name = conv_name_base + '2c', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 4, name = bn_name_base + '2c')(X)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation (??2 lines)
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)


    return X

def conv_block(X, f, filters, stage, block, s = 2):
    """
    Implementation of the convolutional block as defined in Figure 4

    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network
    s -- Integer, specifying the stride to be used

    Returns:
    X -- output of the convolutional block, tensor of shape (n_H, n_W, n__D, n_C)
    """

    # defining names
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # Retrieve Filters
    F1, F2, F3 = filters

    # Save the input value
    X_shortcut = X


    # First component of main path
    X = Conv3D(F1, (1, 1, 1),
               strides = (s, s, s),
               name = conv_name_base + '2a', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 4, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    # Second component of main path (??3 lines)
    X = Conv3D(filters = F2, kernel_size = (f, f, f),
               strides = (1, 1, 1), padding = 'same',
               name = conv_name_base + '2b', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 4, name = bn_name_base + '2b')(X)
    X = Activation('relu')(X)


    # Third component of main path (??2 lines)
    X = Conv3D(filters = F3, kernel_size = (1, 1 , 1),
               strides = (1, 1, 1), padding = 'valid',
               name = conv_name_base + '2c', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 4, name = bn_name_base + '2c')(X)


    ##### SHORTCUT PATH #### (??2 lines)
    X_shortcut = Conv3D(filters = F3, kernel_size = (1, 1, 1),
                        strides = (s, s, s), padding = 'valid',
                        name = conv_name_base + '1',
                        kernel_initializer = glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(axis = 4, name = bn_name_base + '1')(X_shortcut)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation (??2 lines)
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    return X
