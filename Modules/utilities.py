# Import packages
import os
from tensorflow import keras
from Modules.config import *
import shutil
from pathlib import Path


def download_datasets(extract=True):
    """
    Downloads and saves the datasets within the Datasets folder.

    :param extract: if True tries extracting the file as an Archive, like tar or zip. default_value=True
    :return:
    """
    # Absolute path needed to save the files retrieved within the Datasets folder
    target_dir = os.path.abspath(base_dir)
    # Names of the file to download from the specified url
    filenames = ['DIV2K_valid_LR_bicubic_X4', 'DIV2K_valid_LR_unknown_X4', 'DIV2K_valid_HR',
                 'DIV2K_train_LR_bicubic_X4', 'DIV2K_train_LR_unknown_X4', 'DIV2K_train_HR']
    # Name of the new folders wherein the images will be moved
    new_folders = ['Valid_A', 'Valid_B', 'Valid_HR', 'Train_A', 'Train_B', 'Train_HR']
    for filename, folder in zip(filenames, new_folders):
        source_url = dataset_url + filename + '.zip'
        # Download a file from a URL if not already downloaded and processed
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


def split_dataset(test_size=100):
    """
    Creates the test datasets moving a number of images from the training folders.

    :param test_size: number of images to move from the training folders to the test folders. default_value=100
    :return:
    """
    train_folders = ['Train_A', 'Train_B', 'Train_HR']
    # Name of the new folders wherein the images will be moved.
    test_folders = ['Test_A', 'Test_B', 'Test_HR']
    for train_folder, test_folder in zip(train_folders, test_folders):
        # Train folder path
        train_folder = os.path.join(base_dir, train_folder)
        test_images = sorted(os.listdir(train_folder))[-test_size:]
        # Test folder path
        test_folder = os.path.join(base_dir, test_folder)
        # Division is made only if the test directory does not already exist
        if not os.path.isdir(test_folder):
            # Create a new folder for the test dataset
            Path(test_folder).mkdir(parents=True, exist_ok=True)
            for image in test_images:
                # Move the images to the created folder
                shutil.move(os.path.join(train_folder, image), test_folder)



