# Import packages
import os
from tensorflow.keras.backend import get_value
from Modules.pre_processing import load_dataset
from tensorflow import keras, clip_by_value
from Modules.config import *
import shutil
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sn
from sys import stdout
from Modules.components import *


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
            # Download the folder
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


def download_test_datasets(scale=4):
    """
    Downloads the test datasets (Set5, Set14, Urban100, BSD100) and prepare their folders to facilitate their
    management.

    :param scale: scale difference between the two images. default_value=4
    """
    # Absolute path needed to save the files retrieved within the Datasets folder
    target_dir = os.path.abspath(base_dir)
    # URLs from which to download the test datasets
    urls = [set5_url, set14_url, urban100_url, bsd100_url]
    # Name of the content to download
    new_folders = ['Set5_SR', 'Set14_SR', 'Urban100', 'BSD100_SR']
    # List containing the final filename part of the images that will be kept
    final_part = ('HR.png', 'LR.png')
    # Name of the sub folder that from which the images will be retrieved
    sub_folder_name = f'image_SRF_{scale}'
    # Cycle across all the selected test datasets
    for url, folder in zip(urls, new_folders):
        # Name of the folder downloaded
        directory = folder.split('_')[0]
        # Download the specific folder from the URL if not already downloaded and processed
        if directory not in os.listdir(target_dir):
            # Folder where to download the content
            new_dir = target_dir
            # Unfortunately the folders downloaded have a different structure. The following if-else is needed to
            # create a folder for the 'Urban100' or 'BSD100_SR' where to insert the content downloaded. Only following
            # this procedure the final structure of all the folders becomes the same.
            if folder in new_folders[-2:]:
                # Create a new directory where to insert the content downloaded for one of the following datasets:
                # 'Urban100' and 'BSD100_SR'.
                new_dir = os.path.join(target_dir, directory)
                os.mkdir(new_dir)
            # Download the folder
            keras.utils.get_file(folder, url, cache_subdir=new_dir, extract=True)
            # Remove the additional file downloaded
            os.remove(os.path.join(new_dir, folder))
            # Get the path to the folder downloaded
            directory = os.path.join(target_dir, directory)
            # List of all the sub-folders within the folder downloaded
            sub_folders = [item for item in os.listdir(directory)]
            # Path of the sub-folder from which the images are collected
            sub_directory = os.path.join(directory, sub_folder_name)
            # List of all the images inside the sub-folder selected
            file_list = os.listdir(sub_directory)
            # Keep only all the images that end with one of the strings in final_part
            file_list = [file for file in file_list if file.endswith(final_part, 14)]
            # Move the images selected outside the sub folder but inside the folder downloaded
            for file in file_list:
                shutil.move(os.path.join(sub_directory, file), directory)
            # Delete all the sub-folders
            for sub_folder in sub_folders:
                shutil.rmtree(os.path.join(directory, sub_folder), ignore_errors=True)


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
            for img in test_images:
                # Move the images to the created folder
                shutil.move(os.path.join(train_folder, img), test_folder)


def plot_pair(lr_image, hr_image, ax=True, title=True):
    """
    Plots a comparison between the low-resolution and high-resolution images given.

    :param lr_image: low-resolution image.
    :param hr_image: high-resolution image.
    :param ax: if True the axis of the plots, i.e. images, are hidden. default_value=True
    :param title: if True the titles of the images are displayed. default_value=True
    :return:
    """
    # Change seaborn style to avoid the presence of the grid within the plots
    sn.set_style("whitegrid", {'axes.grid': False})
    # Create figure depicting the low and high resolution images side-by-side
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(lr_image)
    axes[1].imshow(hr_image)
    # If title==True insert titles
    if title:
        axes[0].set_title('Low Resolution')
        axes[1].set_title('High Resolution')
    # If ax==True insert axes
    if not ax:
        axes[0].axis('off')
        axes[1].axis('off')
    plt.tight_layout()
    plt.show()


def plot_results(lr_images, predictions, hr_images, ax=True, title=True):
    """
    Plots the low-resolution, the predicted and the high-resolution images juxtaposed for every tuple of images passed.

    :param lr_images: list of the low-resolution images.
    :param predictions: list of the predicted images.
    :param hr_images: list of the high-resolution images.
    :param ax: if True the axis of the plots, i.e. images, are hidden. default_value=True
    :param title: if True the titles of the images are displayed. default_value=True
    :return:
    """
    # Change seaborn style to avoid the presence of the grid within the plots
    sn.set_style("whitegrid", {'axes.grid': False})
    # For the same starting image display its low-resolution, predicted and high-resolution versions.
    for lr_image, pred_image, hr_image in zip(lr_images, predictions, hr_images):
        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(15, 5.3))
        axes[0].imshow(lr_image)
        axes[1].imshow(pred_image)
        axes[2].imshow(hr_image)
        # If title==True insert titles
        if title:
            axes[0].set_title('Low Resolution')
            axes[1].set_title('Prediction')
            axes[2].set_title('Ground Truth')
        # If ax==True insert axes
        if not ax:
            axes[0].axis('off')
            axes[1].axis('off')
            axes[2].axis('off')
        plt.tight_layout()
        plt.show()


def plot_results_bicubic(lr_images, predictions, hr_images, ax=True, title=True, scale=4):
    """
    Plots a comparison among the low-resolution image, the predicted image, the ground truth image and the high
    resolution image obtained through bicubic interpolation for every tuple of images passed.

    :param lr_images: list of the low-resolution images.
    :param predictions: list of the predicted images.
    :param hr_images: list of the high-resolution images.
    :param ax: if True the axis of the plots, i.e. images, are hidden. default_value=True
    :param title: if True the titles of the images are displayed. default_value=True
    :param scale: scale difference between the two images. default_value=4
    :return:
    """
    # Change seaborn style to avoid the presence of the grid within the plots
    sn.set_style("whitegrid", {'axes.grid': False})
    results_psnr = []
    results_ssim = []
    # Size of the bicubic HR image
    size = [patch_size * scale, patch_size * scale]
    # For the same starting image display its low-resolution, predicted and high-resolution versions.
    for lr_image, pred_image, hr_image in zip(lr_images, predictions, hr_images):
        # Create the bicubic image via a bicubic interpolation applied on the LR image.
        bicubic = image.resize(lr_image, size, method=image.ResizeMethod.BICUBIC)
        # Clip all the values that are not in the range [0,1] and that are created by the previous step.
        bicubic = clip_by_value(bicubic, clip_value_min=0, clip_value_max=1)
        # Compute the PSNR and SSIM metrics on every bicubic image obtained
        results_psnr.append(get_value(psnr_metric(hr_image, bicubic)))
        results_ssim.append(get_value(ssim_metric(hr_image, bicubic)))
        # Create figure
        fig, axes = plt.subplots(1, 4, figsize=(20, 5.3))
        axes[0].imshow(lr_image)
        axes[1].imshow(bicubic)
        axes[2].imshow(pred_image)
        axes[3].imshow(hr_image)
        # If title==True insert titles
        if title:
            axes[0].set_title('Low Resolution')
            axes[1].set_title('Bicubic')
            axes[2].set_title('Prediction')
            axes[3].set_title('Ground Truth')
        # If ax==True insert axes
        if not ax:
            axes[0].axis('off')
            axes[1].axis('off')
            axes[2].axis('off')
            axes[3].axis('off')
        plt.tight_layout()
        plt.show()
    return results_psnr, results_ssim


def compute_results_bicubic(lr_images, predictions, hr_images, scale=4):
    """
    Plots a comparison among the low-resolution image, the predicted image, the ground truth image and the high
    resolution image obtained through bicubic interpolation for every tuple of images passed.

    :param lr_images: list of the low-resolution images.
    :param predictions: list of the predicted images.
    :param hr_images: list of the high-resolution images.
    :param scale: scale difference between the two images. default_value=4
    :return:
    """
    results_psnr, results_ssim = [], []
    # Size of the bicubic HR image
    size = [patch_size * scale, patch_size * scale]
    # For the same starting image display its low-resolution, predicted and high-resolution versions.
    for lr_image, pred_image, hr_image in zip(lr_images, predictions, hr_images):
        # Create the bicubic image via a bicubic interpolation applied on the LR image.
        bicubic = image.resize(lr_image, size, method=image.ResizeMethod.BICUBIC)
        # Clip all the values that are not in the range [0,1] and that are created by the previous step.
        bicubic = clip_by_value(bicubic, clip_value_min=0, clip_value_max=1)
        # Compute the PSNR and SSIM metrics on every bicubic image obtained
        results_psnr.append(get_value(psnr_metric(hr_image, bicubic)))
        results_ssim.append(get_value(ssim_metric(hr_image, bicubic)))
    return results_psnr, results_ssim


def plot_history(metric, val_metric, loss, val_loss, title=None):
    """
    Plots the history of the training phase and validation phase. It compares, in two different subplots, the metric
    and the loss of the model.

    :param metric: values of the metric measured after every epoch on the training set.
    :param val_metric: values of the metric measured after every epoch on the validation set.
    :param loss: values of the loss measured after every epoch on the training set.
    :param val_loss: values of the loss measured after every epoch on the validation set.
    :param title: title of the figure displayed. default_value=None
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
    # If title==True insert titles
    if title is not None:
        fig.suptitle(title)
    plt.tight_layout()
    plt.show()


def plot_learning(metric, loss, val_metric, val_loss, title=None):
    """
    Plots the history of the training phase and validation phase for the GAN model. It compares, in two different
    subplots, the metric and the loss of the model.

    :param metric: values of the metric measured after every batch on the training set.
    :param val_metric: values of the metric measured after every epoch on the validation set.
    :param loss: values of the loss measured after every batch on the training set.
    :param val_loss: values of the loss measured after every epoch on the validation set.
    :param title: title of the figure displayed. default_value=None
    :return:
    """
    # The values of the metric and the loss are measured after every batch during the training phase. The following
    # lines enable to obtain the mean of the values related to each epoch
    step = int((800 - test_dim) / batch_dim)
    metric = [np.mean(metric[index:index + step]) for index in range(0, len(metric), step)]
    loss = [np.mean(loss[index:index + step]) for index in range(0, len(loss), step)]
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
    # If title==True insert titles
    if title is not None:
        fig.suptitle(title)
    plt.tight_layout()
    plt.show()


def progressbar(iterable_object, prefix="Status", size=30, output=stdout, iterable=False, iterations=30):
    """
    Displays a progress-bar animation, associated to a for cycle, inside a specified output "place".

    :param iterable_object: object iterated through the for cycle.
    :param prefix: string inserted before the progress-bar. default_value=""
    :param size: dimension of the progress-bar. default_value=60
    :param output: defines where to display the progress-bar. default_value=stdout
    :param iterable: if True the object length can be obtained via len(). Otherwise the length parameter must
        be passed. default_value=True
    :param iterations: number of times the progress bar must be updated. Required only if the object has not a
        length property. default_value=30
    :return:
    """
    if iterable:
        length = len(iterable_object)
    else:
        length = iterations

    def update_bar(step):
        progress = int(size * step / length)
        output.write("%s: [%s%s%s] %i/%i\r" % (prefix, "=" * progress, ">", "." * (size - progress), step, length))
        output.flush()

    update_bar(0)
    for counter, item in enumerate(iterable_object):
        yield item
        update_bar(counter + 1)
    output.write("\n")
    output.flush()


def compute_dataset_mean():
    """
    Computes the mean (channel-wise) of the starting dataset.

    :return:
    """
    # List of the folders where the training images are located
    folders = ['Train_A', 'Train_B', 'Train_HR']
    for folder in folders:
        path = os.path.join(base_dir, folder)
        # List of the paths related to all the images within the folder considered
        images_list = [os.path.join(path, item) for item in os.listdir(path)]
        # Create the dataset object (tf.data.Dataset object)
        data_set = load_dataset(images_list)
        # List of the means computed on the R, G and B channels for every image
        mean_list = []
        for img in data_set:
            mean_list.append(np.mean(img, axis=(0, 1)))
        # Print the RGB means calculated on the images of the current folder
        print(np.mean(mean_list, axis=0))
