# Dedicated file to store global variables
# This file can help to reduce hard_coding and to change only once variables in different files and functions.
# In particular for this project the tasks share the following information.

base_dir = './Datasets'
dataset_url = 'https://data.vision.ee.ethz.ch/cvl/DIV2K/'

# Important parameters
# Although not compulsory, a good choice could be setting the value of batch_dim and test_dim so that:
# (number of training images - test_dim ) / batch_dim returns an integer
# and, number of validation images / batch_dim returns an integer
# and, number of test images / batch_dim returns an integer
batch_dim = 10
patch_size = 64
test_dim = 20

# Channel-wise Mean computed on the entire training DIV2K dataset. The mean is computed via the compute_dataset_mean()
# function within the module called utilities.
RGB_MEAN_A = [114.3569244, 111.5630249, 103.15689794]
RGB_MEAN_B = [114.33866302, 111.52755002, 103.09925148]
RGB_MEAN_HR = [114.35629928, 111.561547, 103.1545782]

