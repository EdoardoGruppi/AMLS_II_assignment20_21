# Import packages
from Modules.utilities import download_datasets, split_dataset
from Modules.pre_processing import prepare_batches
from Modules.config import *
from Modules.components import *
from A.a import A
from B.b import B

tf.compat.v1.enable_eager_execution()
# set_memory_growth() allocates exclusively the GPU memory needed
physical_devices = tf.config.experimental.list_physical_devices('GPU')
print("Num GPUs Available: ", len(physical_devices))
if len(physical_devices) is not 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# ======================================================================================================================
# Download dataset
download_datasets()
split_dataset(test_size=test_dim)

# ======================================================================================================================
# Data preprocessing
train_batches, valid_batches, test_batches = prepare_batches(crop_size=patch_size, batch_size=batch_dim, task='A',
                                                             rotation=True, flip=True)

# ======================================================================================================================
# Task A
input_shape = [patch_size, patch_size, 3]
# Build model object.
# model_A = A(input_shape, loss='mae')
# # Train model based on the training set (you should fine-tune your model based on validation set).
# acc_A_train, acc_A_valid = model_A.train(train_batches, valid_batches, epochs=20, verbose=2)
# # Test model based on the test set.
# psnr_A_test, ssim_A_test = model_A.test(test_batches, plot='bicubic')
# # Clean up memory/GPU etc...
# del model_A
# print('\n Task   {:<12} {:<12} {:<12} {:<12}\n'.format('Train Psnr', 'Valid Psnr', 'Test Psnr', 'Test Ssim'),
#       f'B:     {acc_A_train:<12.4f} {acc_A_valid:<12.4f} {psnr_A_test:<12.4f} {ssim_A_test:<12.4f}\n')

# Model B Task A =======================================================================================================
# Build model object.
model_B = B(input_shape, loss='mse')
# Train model based on the training set (you should fine-tune your model based on validation set).
acc_B_train, acc_B_valid = model_B.train(train_batches, valid_batches, epochs=10, plot=True)
# # Test model based on the test set.
psnr_B_test, ssim_B_test = model_B.test(test_batches, plot='bicubic')
# Clean up memory/GPU etc...
print('\n Task   {:<12} {:<12} {:<12} {:<12}\n'.format('Train Psnr', 'Valid Psnr', 'Test Psnr', 'Test Ssim'),
      f'B:     {acc_B_train:<12.4f} {acc_B_valid:<12.4f} {psnr_B_test:<12.4f} {ssim_B_test:<12.4f}\n')

# ======================================================================================================================
# Data preprocessing
# train_batches, valid_batches, test_batches = prepare_batches(crop_size=96, batch_size=10, task='B',
#                                                              rotation=True, flip=True)

# # ====================================================================================================================
# Task B
# Build model object.
# model_B = B(input_shape, loss='mse')
# # Train model based on the training set (you should fine-tune your model based on validation set).
# acc_B_train, acc_B_valid = model_B.train(train_batches, valid_batches, epochs=10, plot=True)
# # # Test model based on the test set.
# psnr_B_test, ssim_B_test = model_B.test(test_batches, plot='bicubic')
# # Clean up memory/GPU etc...
# #
# print('\n Task   {:<12} {:<12} {:<12} {:<12}\n'.format('Train Psnr', 'Valid Psnr', 'Test Psnr', 'Test Ssim'),
#       f'B:     {acc_B_train:<12.4f} {acc_B_valid:<12.4f} {psnr_B_test:<12.4f} {ssim_B_test:<12.4f}\n')
#
# # ====================================================================================================================
# ## Print out your results with following format:
# print('\n Task   {:<12} {:<12} {:<12}\n'.format('Train Acc', 'Valid Acc', 'Test Acc'),
#       f'A:     {acc_A_train:<12.4f} {acc_A_valid:<12.4f} {acc_A_test:<12.4f}\n',
#       f'B:     {acc_B_train:<12.4f} {acc_B_valid:<12.4f} {acc_B_test:<12.4f}\n')
