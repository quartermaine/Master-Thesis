# from pathlib import Path

import numpy as np

# import cv2

# import imutils

import pandas as pd

#from tqdm import tqdm

# import PIL.Image as Image

from IPython.display import clear_output

#import seaborn as sns

#from pylab import rcParams

import matplotlib.pyplot as plt

#from matplotlib import rc

#from matplotlib.ticker import MaxNLocator

from glob import glob

import shutil
import os
# import watermark

from collections import defaultdict
import warnings

# Ignore FutureWarning from numpy
warnings.simplefilter(action='ignore', category=FutureWarning)


def print_split(func):
    '''
    Prints spliting diagnostics

    Parameters:
    ---------------
    input_dir     see split_ files
    out_dir
    folder_names
    split_size

    Returns:
    ----------

    '''
    def inner(*args, **kwargs):
        func(*args, **kwargs)
        file_names = args[2]
        out_dir = args[1]
        innerdict = {}
        outerdict = {}
        print('Print Split\n')
        for data_path in file_names:
            print(data_path)
            for class_name in os.listdir(f'{out_dir}{data_path}'):
                work_path = os.path.join(f'{out_dir}/{data_path}/{class_name}')
                img_num = len(os.listdir(work_path))
                innerdict[class_name] = img_num
                print(f'{data_path}/{class_name} has {img_num} pictures')

            outerdict[data_path] = innerdict
            innerdict = {}


        df = pd.DataFrame(outerdict).T

        df.plot(kind="bar")
        plt.show()

    return inner




os.system('mkdir SPLIT_DATA')
os.system('mkdir SPLIT_DATA/TRAIN SPLIT_DATA/TEST SPLIT_DATA/VAL SPLIT_DATA/TRAIN/ASD SPLIT_DATA/TRAIN/CONTROL SPLIT_DATA/TEST/ASD SPLIT_DATA/TEST/CONTROL SPLIT_DATA/VAL/ASD SPLIT_DATA/VAL/CONTROL')
# os.system('tree -d')

# path = "/tmp/year/month/week/day"
#
# try:
#     os.makedirs(SPLIT_DATA)
# except OSError:
#     print ("Creation of the directory %s failed" % path)
# else:
#     print ("Successfully created the directory %s" % path)


CURRENT_DIR =os.getcwd()

IMG_PATH = f'{CURRENT_DIR}/DATA/'

# '../input/brain-mri-images-for-brain-tumor-detection/brain_tumor_dataset/'

TARGET_PATH = f'{CURRENT_DIR}/SPLIT_DATA/'

#'/kaggle/working'

FOLDER_NAMES = ['TRAIN', 'VAL', 'TEST']

SPLIT = {'train': 0.7,
         'validation' :0.2}


@print_split
def split_files(input_dir, out_dir,folder_names, split_sizes):

    if sum(list(split_sizes.values())) > 1 :
            raise Exception("Train and Validation should be < 1.0!")
    else :
        train_size, val_size = split_sizes.values()
        val_size += train_size


    for class_name in os.listdir(input_dir):

        image_paths = np.array(glob(f'{input_dir}{class_name}/*'))

        np.random.shuffle(image_paths)

        ds_split = np.split(image_paths,
                        indices_or_sections = [int(train_size * len(image_paths)), int(val_size*len(image_paths))]
                           )


        dataset_data = zip(folder_names, ds_split)

        for ds, images in dataset_data:
            for img_path in images:
                shutil.copy(img_path, f'{out_dir}{ds}/{class_name.upper()}/')
    print(f'Split files completed\n')



split_files(IMG_PATH, TARGET_PATH, FOLDER_NAMES, SPLIT)
