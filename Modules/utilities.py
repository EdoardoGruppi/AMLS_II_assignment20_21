# Import packages
import os
from tensorflow import keras
from Modules.config import *
import shutil


def download_datasets(extract=True):
    """
    Downloads and saves the datasets within the Datasets folder.

    :param extract: if True tries extracting the file as an Archive, like tar or zip.
    :return:
    """
    # Absolute path needed to save the files retrieved within the Datasets folder
    target_dir = os.path.abspath(base_dir)
    # Names of the file to download from the specified url
    filenames = ['DIV2K_valid_LR_bicubic_X4', 'DIV2K_valid_LR_unknown_X4', 'DIV2K_valid_HR',
                 'DIV2K_train_LR_bicubic_X4', 'DIV2K_train_LR_unknown_X4', 'DIV2K_train_HR']
    # Name of the new folders wherein the images will be moved.
    new_folders = ['Valid_A', 'Valid_B', 'Valid_HR', 'Train_A', 'Train_B', 'Train_HR']
    for filename, folder in zip(filenames, new_folders):
        source_url = dataset_url + filename + '.zip'
        # Download a file from a URL if not already downloaded and processed.
        origin = folder + '_cached'
        if origin not in os.listdir(target_dir):
            keras.utils.get_file(origin, source_url, cache_subdir=target_dir, extract=extract)
            # The images are in a sub-folder within the folder downloaded. To facilitate the handling of the images,
            # they are moved in a folder dedicated and easier to access.
            if 'X4' in filename:
                old_folder = os.path.join(filename[:-3], 'X4')
            else:
                old_folder = filename
            # Obtain the path to the new and old folders
            old_dir = os.path.join(target_dir, old_folder)
            new_dir = os.path.join(target_dir, folder)
            # Move the images from one folder to another
            shutil.move(old_dir, new_dir)
            # Delete the old folder
            old_dir = os.path.join(target_dir, filename.split('_X4')[0])
            if os.path.exists(old_dir):
                shutil.rmtree(old_dir)







