# Import packages
import os
from tensorflow import keras
from Modules.config import *
import shutil
from pathlib import Path
import numpy as np
from Modules.pre_processing import images_dataset
from matplotlib import pyplot as plt
from tensorflow.image import psnr
import seaborn as sn


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


def compute_dataset_mean():
    folders = ['Train_A', 'Train_B', 'Train_HR']
    for folder in folders:
        path = os.path.join(base_dir, folder)
        images_list = [os.path.join(path, item) for item in os.listdir(path)]
        data_set = images_dataset(images_list)
        mean_list = []
        for image in data_set:
            mean_list.append(np.mean(image, axis=(0, 1)))
        print(np.mean(mean_list, axis=0))


def plot_lr_hr_pair(lr_image, hr_image, ax=True, title=True):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(lr_image)
    axes[1].imshow(hr_image)
    if title:
        axes[0].set_title('Low Resolution')
        axes[1].set_title('High Resolution')
    if not ax:
        axes[0].axis('off')
        axes[1].axis('off')
    plt.tight_layout()
    plt.show()


def psnr_metric(true_img, pred_img):
    return psnr(true_img, pred_img, max_val=1)


def plot_history(metric, val_metric, loss, val_loss, title=None):
    """
    Plots the history of the training phase and validation phase. It compares in two different subplots the metric
    and the loss of the model.

    :param metric: list of values for every epoch.
    :param val_metric: list of values for every epoch.
    :param loss: list of values for every epoch.
    :param val_loss: list of values for every epoch.
    :param title: tile of the figure printed. default_value=None
    :return:
    """
    sn.set()
    fig, axes = plt.subplots(2, 1, sharex='all', figsize=(13, 8))
    x_axis = list(range(1, len(metric) + 1))
    # First subplot
    axes[0].plot(x_axis, metric)
    axes[0].plot(x_axis, val_metric)
    axes[0].set(ylabel='PSNR')
    axes[0].legend(['Train', 'Valid'], loc='lower right')
    # Second subplot
    axes[1].plot(x_axis, loss)
    axes[1].plot(x_axis, val_loss)
    axes[1].set(ylabel='Loss', xlabel='Epoch')
    # Legend
    axes[1].legend(['Train', 'Valid'], loc='upper right')
    if title is not None:
        fig.suptitle(title)
    plt.tight_layout()
    plt.show()
