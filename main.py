# Import packages
from Modules.utilities import download_datasets, split_dataset, download_test_datasets
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
# Download datasets
download_datasets()
download_test_datasets()
split_dataset(test_size=test_dim)
input_shape = [patch_size, patch_size, 3]

# ====================================================================================================================
# Task A - Ratio x2
results_A2 = []
scale = 2
# Data preprocessing
train_batches, valid_batches, test_batches = prepare_batches(crop_size=patch_size, batch_size=batch_dim,
                                                             task='A', rotation=True, flip=True, scale=scale)

# Build model object.
model_A = A(input_shape, loss='mae')
# Train model based on the training set (you should fine-tune your model based on validation set).
results_A2.append(model_A.train(train_batches, valid_batches, epochs=60, verbose=2))
# Test model based on the test set.
results_A2.append(model_A.test(test_batches, plot=True, scale=scale))
# Test model on the additional test datasets
model_A.additional_tests(plot=False, scale=scale)
results_A2 = np.array(results_A2).flatten()
# Clean up memory/GPU etc...
del bsd100_url, dataset_url, set14_url, set5_url

# ====================================================================================================================
# Task A - Ratio x3
results_A3 = []
scale = 3
# Data preprocessing
train_batches, valid_batches, test_batches = prepare_batches(crop_size=patch_size, batch_size=batch_dim,
                                                             task='A', rotation=True, flip=True, scale=scale)
# Change model object.
model_A.new_scale(scale=scale, loss='mae')
# Train model based on the training set (you should fine-tune your model based on validation set).
results_A3.append(model_A.train(train_batches, valid_batches, epochs=20, verbose=2))
# Test model based on the test set.
results_A3.append(model_A.test(test_batches, plot=True, scale=scale))
# Test model on the additional test datasets
model_A.additional_tests(plot=False, scale=scale)
results_A3 = np.array(results_A3).flatten()

# ====================================================================================================================
# Task A - Ratio x4
results_A4 = []
scale = 4
# Data preprocessing
train_batches, valid_batches, test_batches = prepare_batches(crop_size=patch_size, batch_size=batch_dim,
                                                             task='A', rotation=True, flip=True, scale=scale)
# Change model object.
model_A.new_scale(scale=scale, loss='mae')
# Train model based on the training set (you should fine-tune your model based on validation set).
results_A4.append(model_A.train(train_batches, valid_batches, epochs=20, verbose=2))
# Test model based on the test set.
results_A4.append(model_A.test(test_batches, plot=True, scale=scale))
# Test model on the additional test datasets
model_A.additional_tests(plot=False, scale=scale)
results_A4 = np.array(results_A4).flatten()
# Clean up memory/GPU etc...
del model_A

# ==================================================================================================================
# Task B - Ratio x2
results_B2 = []
scale = 2
# Data preprocessing
train_batches, valid_batches, test_batches = prepare_batches(crop_size=patch_size, batch_size=batch_dim,
                                                             task='B', rotation=True, flip=True, scale=scale)
# Build model object.
model_B = B(input_shape, loss='mae', scale=scale)
# Train model based on the training set (you should fine-tune your model based on validation set).
results_B2.append(model_B.train(train_batches, valid_batches, epochs=60))
# Test model based on the test set.
results_B2.append(model_B.test(test_batches, plot=True, scale=scale))
# Test model on the additional test datasets
model_B.additional_tests(plot=False, scale=scale)
results_B2 = np.array(results_B2).flatten()

# ====================================================================================================================
# Task B - Ratio x3
results_B3 = []
scale = 3
# Data preprocessing
train_batches, valid_batches, test_batches = prepare_batches(crop_size=patch_size, batch_size=batch_dim,
                                                             task='B', rotation=True, flip=True, scale=scale)
# Change model object.
model_B.new_scale(input_shape=input_shape, scale=scale, loss='mae')
# Train model based on the training set (you should fine-tune your model based on validation set).
results_B3.append(model_B.train(train_batches, valid_batches, epochs=20))
# Test model based on the test set.
results_B3.append(model_B.test(test_batches, plot=True, scale=scale))
# Test model on the additional test datasets
model_B.additional_tests(plot=False, scale=scale)
results_B3 = np.array(results_B3).flatten()

# ====================================================================================================================
# Task B - Ratio x4
results_B4 = []
scale = 4
# Data preprocessing
train_batches, valid_batches, test_batches = prepare_batches(crop_size=patch_size, batch_size=batch_dim,
                                                             task='B', rotation=True, flip=True, scale=scale)
# Change model object.
model_B.new_scale(input_shape=input_shape, scale=scale, loss='mae')
# Train model based on the training set (you should fine-tune your model based on validation set).
results_B4.append(model_B.train(train_batches, valid_batches, epochs=20))
# Test model based on the test set.
results_B4.append(model_B.test(test_batches, plot=True, scale=scale))
# Test model on the additional test datasets
model_B.additional_tests(plot=False, scale=scale)
results_B4 = np.array(results_B4).flatten()
# Clean up memory/GPU etc...
del model_B

# ======================================================================================================================
# Print out the results:
print('\n{:<12} {:<12} {:<12} {:<12} {:<12}\n'.format('Task', 'Train Psnr', 'Valid Psnr', 'Test Psnr', 'Test Ssim'),
      '{:<12} {:<12.4f} {:<12.4f} {:<12.4f} {:<12.4f}\n'.format('A_x2', results_A2[0], results_A2[1], results_A2[2],
                                                                results_A2[3]),
      '{:<12} {:<12.4f} {:<12.4f} {:<12.4f} {:<12.4f}\n'.format('A_x3', results_A3[0], results_A3[1], results_A3[2],
                                                                results_A3[3]),
      '{:<12} {:<12.4f} {:<12.4f} {:<12.4f} {:<12.4f}\n'.format('A_x4', results_A4[0], results_A4[1], results_A4[2],
                                                                results_A4[3]),
      '{:<12} {:<12.4f} {:<12.4f} {:<12.4f} {:<12.4f}\n'.format('B_x2', results_B2[0], results_B2[1], results_B2[2],
                                                                results_B2[3]),
      '{:<12} {:<12.4f} {:<12.4f} {:<12.4f} {:<12.4f}\n'.format('B_x3', results_B3[0], results_B3[1], results_B3[2],
                                                                results_B3[3]),
      '{:<12} {:<12.4f} {:<12.4f} {:<12.4f} {:<12.4f}\n'.format('B_x4', results_B4[0], results_B4[1], results_B4[2],
                                                                results_B4[3]))
