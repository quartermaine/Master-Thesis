import numpy as np


def whitening(image):
    """Whitening. Normalises image to zero mean and unit variance.
            [INPUT]
    Image -> the input image to whiten. Three dimensions i.e. [x,y,z]
            [OUTPUT]
    Image -> whitened volume
    """

    image = image.astype(np.float32)

    mean = np.mean(image)
    std = np.std(image)

    if std > 0:
        ret = (image - mean) / std
    else:
        ret = image * 0.
    return ret

def normalise_zero_one(image):
    """Image normalisation. Normalises image to fit [0, 1] range.
            [INPUT]
    Image -> the input image to normalize. Three dimensions i.e. [x,y,z]
            [OUTPUT]
    Image -> normalized volume
    """

    image = image.astype(np.float32)

    minimum = np.min(image)
    maximum = np.max(image)

    if maximum > minimum:
        ret = (image - minimum) / (maximum - minimum)
    else:
        ret = image * 0.
    return ret

def normalise_one_one(image):
    """Image normalization. Normalises image to fit [-1, 1] range.
            [INPUT]
    Image -> the input image to noramalize. Three dimensions i.e. [x,y,z]
            [OUTPUT]
    Image -> normalized volume
    """

    ret = normalise_zero_one(image)
    ret *= 2.
    ret -= 1.
    return ret


def whitening_3D(img):
    """Whitening. Normalises multichannel volume to zero mean and unit variance.
            [INPUT]
    Image -> the input volume to whiten. Expected channel last convention
            i.e. [x,y,z,channels]
            [OUTPUT]
    Image -> whitened volume
    """
    if len(img.shape) == 3:
        return whitening(img)
    else :

        im = [whitening(img[:, :, :, c]) for c in range(img.shape[3])]

        white_im = np.stack(im, axis=3)

        return white_im

def normalise_zero_one_3D(img):
    """Image normalisation. Normalises multichannel volume to fit [0, 1] range.
            [INPUT]
    Image -> the input volume to normalize. Expected channel last convention
            i.e. [x,y,z,channels]
            [OUTPUT]
    Image -> normalized volume
    """
    if len(img.shape) == 3:
        return normalise_zero_one(img)
    else :

        im = [normalise_zero_one(img[:, :, :, c]) for c in range(img.shape[3])]

        norm_im = np.stack(im, axis=3)

        return norm_im

def normalise_one_one_3D(img):
    """Image normalisation. Normalises multichannel volume to fit [-1, 1] range.
            [INPUT]
    Image -> the input volume to normalize. Expected channel last convention
            i.e. [x,y,z,channels]
            [OUTPUT]
    Image -> normalized volume
    """
    if len(img.shape) == 3:
        return normalise_one_one(img)
    else :

        im = [normalise_one_one(img[:, :, :, c]) for c in range(img.shape[3])]

        norm_im = np.stack(im, axis=3)
        return norm_im
