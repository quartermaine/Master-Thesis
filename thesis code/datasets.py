import numpy as np
import tensorflow as tf
import elasticdeform
from scipy import ndimage
from scipy.ndimage.interpolation import affine_transform
import random as rnd
import numba
from numba import jit

# np.random.seed(121)


###################### Functions for augmentatation TfRecords on the fly during training --> slow  ###################
# These functions use tensoflow numpy function wraped on tf functions
# and can be used to augment image on the fly when reeading TfRecords

@tf.function
def flip3D(volume, label):
    '''FLip image
            [INPUT]
    Image -> the input volume to flip. Expected channel last convention
            i.e. [x,y,z,channels]
    Label -> the label corresponding to the Image input.
            [OUTPUT]
    Image -> flipped volume
    Label -> the label corresponding to the Image input.(no transformation is done)
   '''
    def flip(volume):
        # flip randomly
        choice = np.random.randint(3)
        if choice == 0: # flip on x
            volume_flip = volume[::-1, :, :, :]
        if choice == 1: # flip on y
            volume_flip = volume[:, ::-1, :, :]
        if choice == 2: # flip on z
            volume_flip = volume[:, :, ::-1, :]

        return volume_flip

    augmented_volume = tf.numpy_function(flip, [volume], tf.float32)

    return augmented_volume, label

@tf.function
def rotation3D(volume, label):
    '''Rotation of image
            [INPUT]
    Image -> the input volume to rotate. Expected channel last convention
            i.e. [x,y,z,channels]
    Label -> the label corresponding to the Image input.
            [OUTPUT]
    Image -> rotated volume
    Label -> the label corresponding to the Image input.(no transformation is done)
    '''

    def scipy_rotate(volume):

        alpha, beta, gamma = np.random.randint(0, 31, size=3)/180*np.pi
        Rx = np.array([[1, 0, 0],
                    [0, np.cos(alpha), -np.sin(alpha)],
                    [0, np.sin(alpha), np.cos(alpha)]])

        Ry = np.array([[np.cos(beta), 0, np.sin(beta)],
                    [0, 1, 0],
                    [-np.sin(beta), 0, np.cos(beta)]])

        Rz = np.array([[np.cos(gamma), -np.sin(gamma), 0],
                    [np.sin(gamma), np.cos(gamma), 0],
                    [0, 0, 1]])

        R = np.dot(np.dot(Rx, Ry), Rz)

        volume_rot = np.empty_like(volume)
        for channel in tf.range(volume.shape[-1]):
            volume_rot[:,:,:,channel] = affine_transform(volume[:,:,:,channel], R, offset=0, order=3, mode='nearest')

        return volume_rot

    augmented_volume = tf.numpy_function(scipy_rotate, [volume], tf.float32)

    return augmented_volume, label


def center_crop3D(image, central_fraction= 0.85, random=1):
    '''Center crop of image
            [INPUT]
    Image -> the input volume to flip. Expected channel last convention
            i.e. [x,y,z,channels]
    Label -> the label corresponding to the Image input.
            [OUTPUT]
    Image -> center cropped volume
    Label -> the label corresponding to the Image input.(no transformation is done)
    '''

    # perform random flip with probability random
    # if tf.random.uniform(()) <= random:
    if np.random.rand(1) <= random:
        #seed = np.round((tf.random.uniform(())*100).numpy(), decimals=0)
        #seed = tf.random.uniform(())*100

        img = tf.image.central_crop(image, central_fraction)
        img =img.numpy()
        return img
    else:
        return image


@tf.function
def blur3D(volume, label):
    '''Bluring of image
            [INPUT]
    Image -> the input volume to flip. Expected channel last convention
            i.e. [x,y,z,channels]
    Label -> the label corresponding to the Image input.
            [OUTPUT]
    Image -> blured volume
    Label -> the label corresponding to the Image input.(no transformation is done)
    '''

    def blur(volume):

        blur_volume = ndimage.gaussian_filter(volume, sigma = 1.2)

        return blur_volume

    augmented_volume = tf.numpy_function(blur, [volume], tf.float32)

    return augmented_volume, label


@tf.function
def elastic3D(volume, label):
    '''Elastic Deformation of an image
            [INPUT]
    Image -> the input volume to flip. Expected channel last convention
            i.e. [x,y,z,channels]
    Label -> the label corresponding to the Image input.
            [OUTPUT]
    Image -> elastic deformed volume
    Label -> the label corresponding to the Image input.(no transformation is done)
    '''

    def elastic_deform(volume):

        X = [volume[:, :, :, c] for c in range(volume.shape[3])]
        Xel = elasticdeform.deform_random_grid(X, sigma=3, axis=(0, 1, 2), order = 2)
        volume_el = np.stack(Xel, axis=3)

        return volume_el

    augmented_volume = tf.numpy_function(elastic_deform, [volume], tf.float32)

    return augmented_volume, label

########################### Functions for augmentation during writing TfRecords ########################
# These functions use native numpy and scipy
# and can be used to augment image when writing TfRecords

def flip_vol(volume):
    '''FLip image
           [INPUT]
    Image -> the input volume to flip. Expected channel last convention
           i.e. [x,y,z,channels]
           [OUTPUT]
    Image -> flipped volume
    '''
    choice = np.random.randint(3)
    if choice == 0: # flip on x
        volume_flip = volume[::-1, :, :, :]
    if choice == 1: # flip on y
        volume_flip = volume[:, ::-1, :, :]
    if choice == 2: # flip on z
        volume_flip = volume[:, :, ::-1, :]

    # print('flipped')

    return volume_flip


def scipy_rotate_vol(volume):
    '''Rotation of image
            [INPUT]
    Image -> the input volume to rotate. Expected channel last convention
            i.e. [x,y,z,channels]
            [OUTPUT]
    Image -> rotated volume
    '''

    alpha, beta, gamma = np.random.randint(0, 31, size=3)/180*np.pi
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(alpha), -np.sin(alpha)],
                   [0, np.sin(alpha), np.cos(alpha)]])

    Ry = np.array([[np.cos(beta), 0, np.sin(beta)],
                    [0, 1, 0],
                    [-np.sin(beta), 0, np.cos(beta)]])

    Rz = np.array([[np.cos(gamma), -np.sin(gamma), 0],
                   [np.sin(gamma), np.cos(gamma), 0],
                   [0, 0, 1]])

    R = np.dot(np.dot(Rx, Ry), Rz)

    volume_rot = np.empty_like(volume)
    for channel in range(volume.shape[-1]):
        volume_rot[:,:,:,channel] = affine_transform(volume[:,:,:,channel], R,
                                                     offset=0,
                                                     order=3,
                                                     mode='nearest')
    # print('rotated')
    return volume_rot


def blur_vol(volume):
    '''Bluring of image
            [INPUT]
    Image -> the input volume to flip. Expected channel last convention
            i.e. [x,y,z,channels]
            [OUTPUT]
    Image -> blured volume
    '''

    blur_volume = ndimage.gaussian_filter(volume, sigma = 1.2)
    # print('blured')
    return blur_volume


def elastic_deform_vol(volume):
    '''Elastic Deformation of an image
            [INPUT]
    Image -> the input volume to flip. Expected channel last convention
            i.e. [x,y,z,channels]
            [OUTPUT]
    Image -> elastic deformed volume
    '''

    X = [volume[:, :, :, c] for c in range(volume.shape[3])]

    Xel = elasticdeform.deform_random_grid(X, sigma=3, axis=(0, 1, 2), order = 2)

    volume_el = np.stack(Xel, axis=3)

    # print('deformed')

    return volume_el

def pad(image, new_shape, border_mode="constant", value=0):
    '''
    image: [H, W, D, C] or [H, W, D]
    new_shape: [H, W, D]
    '''
    axes_not_pad = len(image.shape) - len(new_shape)

    old_shape = np.array(image.shape[:len(new_shape)])
    new_shape = np.array([max(new_shape[i], old_shape[i]) for i in range(len(new_shape))])

    difference = new_shape - old_shape
    pad_below = difference // 2
    pad_above = difference - pad_below

    pad_list = [list(i) for i in zip(pad_below, pad_above)] + [[0, 0]] * axes_not_pad

    if border_mode == 'reflect':
        res = np.pad(image, pad_list, border_mode)
    elif border_mode == 'constant':
        res = np.pad(image, pad_list, border_mode, constant_values=value)
    else:
        raise ValueError

    return res

def get_center_crop_coords(height, width, depth, crop_height, crop_width, crop_depth):
    x1 = (height - crop_height) // 2
    x2 = x1 + crop_height
    y1 = (width - crop_width) // 2
    y2 = y1 + crop_width
    z1 = (depth - crop_depth) // 2
    z2 = z1 + crop_depth
    return x1, y1, z1, x2, y2, z2


def center_crop_vol(img):
    crop_height, crop_width, crop_depth = 56, 63, 56
    height, width, depth = img.shape[:3]
    if height < crop_height or width < crop_width or depth < crop_depth:
        raise ValueError
    x1, y1, z1, x2, y2, z2 = get_center_crop_coords(height, width, depth, crop_height, crop_width, crop_depth)

    im = [img[x1:x2, y1:y2, z1:z2, c] for c in range(img.shape[3])]
    im = [pad(img, (61, 73, 61)) for img in im]

    Xcenter = np.stack(im, axis = 3)

    return Xcenter
