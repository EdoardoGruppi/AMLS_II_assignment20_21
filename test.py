# Import packages
from Modules.utilities import download_datasets, split_dataset
from Modules.pre_processing import prepare_batches
import tensorflow as tf
from A.a import A

tf.compat.v1.enable_eager_execution()
# set_memory_growth() allocates exclusively the GPU memory needed
physical_devices = tf.config.experimental.list_physical_devices('GPU')
print("Num GPUs Available: ", len(physical_devices))
if len(physical_devices) is not 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# ======================================================================================================================
# Download dataset
download_datasets()
split_dataset(test_size=20)
crop_size = 96

# ======================================================================================================================
# Data preprocessing
train_batches, valid_batches, test_batches = prepare_batches(crop_size=crop_size, batch_size=10, task='A',
                                                             rotation=True, flip=True)

# ====================================================================================================================
# Task A
input_shape = [crop_size, crop_size, 3]
# Build model object.
model_A = A(input_shape)
# Train model based on the training set (you should fine-tune your model based on validation set).
acc_A_train, acc_A_valid = model_A.train(train_batches, valid_batches, epochs=20, verbose=1)
# Test model based on the test set.
acc_A_test = model_A.test(test_batches, plot=True)
# Clean up memory/GPU etc...
#

# ======================================================================================================================
# Data preprocessing
# train_batches, valid_batches, test_batches = prepare_batches(crop_size=96, batch_size=10, task='B',
#                                                              rotation=True, flip=True)

# # ====================================================================================================================
# # Task B
# # Build model object.
# model_B = B(args...)
# # Train model based on the training set (you should fine-tune your model based on validation set).
# acc_B_train, acc_B_valid = model_B.train(args...)
# # Test model based on the test set.
# acc_B_test = model_B.test(args...)
# # Some code to free memory if necessary.
# Clean up memory/GPU etc...
#
# # ====================================================================================================================
# ## Print out your results with following format:
# print('Task  {:<12} {:<12} {:<12}\n'.format('Train Acc', 'Valid Acc', 'Test Acc'),
#       f'A:  {acc_A_train:<12.4f} {acc_A_valid:<12.4f} {acc_A_test:<12.4f}\n',
#       f'B:  {acc_B_train:<12.4f} {acc_B_valid:<12.4f} {acc_B_test:<12.4f}\n')
