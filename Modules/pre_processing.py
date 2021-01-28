# Import packages
import os
from Modules.config import *
from tensorflow.data import Dataset
from tensorflow.io import read_file
from tensorflow.image import decode_png, flip_left_right, rot90
from tensorflow import shape, int32
from tensorflow.random import uniform
from tensorflow.python.data.experimental import AUTOTUNE


def prepare_batches(crop_size=96, batch_size=10, task='A', rotation=True, flip=True):
    train_dir = os.path.join(base_dir, 'Train_' + task)
    valid_dir = os.path.join(base_dir, 'Valid_' + task)
    test_dir = os.path.join(base_dir, 'Test_' + task)
    train_batches = create_dataset(train_dir, batch_size=batch_size, crop_size=crop_size, rotation=rotation, flip=flip)
    valid_batches = create_dataset(valid_dir, batch_size=batch_size, crop_size=crop_size, rotation=rotation, flip=flip)
    test_batches = create_dataset(test_dir, batch_size=batch_size, crop_size=crop_size, rotation=rotation, flip=flip)
    return train_batches, valid_batches, test_batches


def create_dataset(lr_path, batch_size=8, crop_size=96, rotation=True, flip=True):
    hr_path = lr_path.split('_')[0] + '_HR'
    lr_images_list = [os.path.join(lr_path, item) for item in os.listdir(lr_path)]
    hr_images_list = [os.path.join(hr_path, item) for item in os.listdir(hr_path)]
    data_set = Dataset.zip((images_dataset(lr_images_list), images_dataset(hr_images_list)))
    data_set = data_set.map(lambda lr_image, hr_image: pre_processing(lr_image, hr_image, crop_size, rotation, flip),
                            num_parallel_calls=AUTOTUNE)
    data_set = data_set.batch(batch_size)
    return data_set


def images_dataset(images_list):
    data_set = Dataset.from_tensor_slices(images_list)
    data_set = data_set.map(read_file, num_parallel_calls=AUTOTUNE)
    data_set = data_set.map(lambda x: decode_png(x, channels=3), num_parallel_calls=AUTOTUNE)
    return data_set


def pre_processing(lr_image, hr_image, crop_size=96, rotation=True, flip=True, scale=4):
    hr_crop_size = scale * crop_size
    image_size = shape(lr_image)[:3]
    lr_width = uniform(shape=(), maxval=image_size[1] - crop_size + 1, dtype=int32)
    lr_height = uniform(shape=(), maxval=image_size[0] - crop_size + 1, dtype=int32)
    hr_width = lr_width * scale
    hr_height = lr_height * scale
    lr_image = lr_image[lr_height:lr_height + crop_size, lr_width:lr_width + crop_size]
    hr_image = hr_image[hr_height:hr_height + hr_crop_size, hr_width:hr_width + hr_crop_size]
    lr_image = lr_image / 255
    hr_image = hr_image / 255
    toss = uniform(shape=(), maxval=2, dtype=int32)
    if flip and toss == 1:
        lr_image = flip_left_right(lr_image)
        hr_image = flip_left_right(hr_image)
    n_times = uniform(shape=(), maxval=4, dtype=int32)
    if rotation and n_times != 0:
        lr_image = rot90(lr_image, n_times)
        hr_image = rot90(hr_image, n_times)
    return lr_image, hr_image



