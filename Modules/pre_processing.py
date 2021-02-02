# Import packages
import os
from Modules.config import *
from tensorflow import data, io, image, random, shape, int32
from tensorflow.python.data.experimental import AUTOTUNE


def prepare_batches(crop_size=96, batch_size=10, task='A', rotation=True, flip=True):
    """
    Prepares the train, test and valid batches.

    :param crop_size: dimension of the square images extracted from the original ones. default_value=96
    :param batch_size: size of each batch. default_value=10
    :param task: it can be equal to 'A' or 'B' according to the task performed. default_value='A'
    :param rotation: if True images can be randomly rotated. default_value=True
    :param flip: if True images can be randomly and horizontally flipped. default_value=True
    :return: the train, test and valid batches.
    """
    # Get the path of the train, valid and test folders
    train_dir = os.path.join(base_dir, 'Train_' + task)
    valid_dir = os.path.join(base_dir, 'Valid_' + task)
    test_dir = os.path.join(base_dir, 'Test_' + task)
    # Call the create_dataset function to obtain the batches from the datasets
    train_batches = create_dataset(train_dir, batch_size=batch_size, crop_size=crop_size, rotation=rotation, flip=flip)
    valid_batches = create_dataset(valid_dir, batch_size=batch_size, crop_size=crop_size, rotation=rotation, flip=flip)
    test_batches = create_dataset(test_dir, batch_size=batch_size, crop_size=crop_size, rotation=rotation, flip=flip,
                                  repetition=False)
    return train_batches, valid_batches, test_batches


def create_dataset(lr_path, batch_size, crop_size, rotation=True, flip=True, repetition=True):
    """
    Creates the batches given the path of the low resolution dataset.

    :param lr_path: path of the low resolution dataset.
    :param batch_size: size of each batch.
    :param crop_size: dimension of the square images extracted from the original ones.
    :param rotation: if True images can be randomly rotated. default_value=True
    :param flip: if True images can be randomly and horizontally flipped. default_value=True
    :param repetition: if True the dataset is repeated indefinitely. It should be True for the training and valid
        batches and false for the test batches. default_value=True
    :return: the dataset divided in batches
    """
    # Get the path of the related folder containing the high resolution images
    hr_path = lr_path.split('_')[0] + '_HR'
    # List of paths regarding all the low and high resolution images
    lr_images_list = [os.path.join(lr_path, item) for item in os.listdir(lr_path)]
    hr_images_list = [os.path.join(hr_path, item) for item in os.listdir(hr_path)]
    # Creates a Dataset by zipping together the loaded datasets
    data_set = data.Dataset.zip((load_dataset(lr_images_list), load_dataset(hr_images_list)))
    # Preprocess each element of the obtained dataset
    data_set = data_set.map(lambda lr_image, hr_image: pre_processing(lr_image, hr_image, crop_size, rotation, flip),
                            num_parallel_calls=AUTOTUNE)
    # Combines consecutive elements of the dataset into batches
    data_set = data_set.batch(batch_size)
    if repetition:
        data_set = data_set.repeat()
    return data_set


def load_dataset(images_list):
    """
    Loads a dataset starting from the list of the paths of all the images.

    :param images_list: list of the images path.
    :return: the Dataset object generated
    """
    # Creates a Dataset whose elements are the images path
    data_set = data.Dataset.from_tensor_slices(images_list)
    # Reads the contents of the input filename
    data_set = data_set.map(io.read_file, num_parallel_calls=AUTOTUNE)
    # Decode a PNG image to a tensor
    data_set = data_set.map(lambda x: image.decode_png(x, channels=3), num_parallel_calls=AUTOTUNE)
    return data_set


def pre_processing(lr_image, hr_image, crop_size, rotation=True, flip=True, scale=4):
    """
    Processes a pair of low and high resolution images. It crops the same patch from the two images before rotating it
    and flipping it (if required).

    :param lr_image: low resolution image.
    :param hr_image: high resolution image.
    :param crop_size: dimension of the square images extracted from the original ones.
    :param rotation: if True images can be randomly rotated. default_value=True
    :param flip: if True images can be randomly and horizontally flipped. default_value=True
    :param scale: scale difference between the two images. default_value=4
    :return: the preprocessed patches of the images given.
    """
    # Get the dimension of the patch to be extracted from the HR image
    hr_crop_size = scale * crop_size
    # The original size of the low resolution images
    image_size = shape(lr_image)[:3]
    # Generate values from a uniform distribution within the range [0, max_val) to get the starting point from which
    # to apply the cropping.
    lr_width = random.uniform(shape=(), maxval=image_size[1] - crop_size + 1, dtype=int32)
    lr_height = random.uniform(shape=(), maxval=image_size[0] - crop_size + 1, dtype=int32)
    # Get the coordinates of the starting point for the HR image as well.
    hr_width = lr_width * scale
    hr_height = lr_height * scale
    # Crop images
    lr_image = lr_image[lr_height:lr_height + crop_size, lr_width:lr_width + crop_size]
    hr_image = hr_image[hr_height:hr_height + hr_crop_size, hr_width:hr_width + hr_crop_size]
    # Normalize images within the range [0,1]
    lr_image = lr_image / 255
    hr_image = hr_image / 255
    # Generate 0 and 1 values with the same probability
    toss = random.uniform(shape=(), maxval=2, dtype=int32)
    if flip and toss == 1:
        lr_image = image.flip_left_right(lr_image)
        hr_image = image.flip_left_right(hr_image)
    # Generate [0, 1, 2, 3] values with the same probability
    n_times = random.uniform(shape=(), maxval=4, dtype=int32)
    if rotation and n_times != 0:
        lr_image = image.rot90(lr_image, n_times)
        hr_image = image.rot90(hr_image, n_times)
    return lr_image, hr_image
