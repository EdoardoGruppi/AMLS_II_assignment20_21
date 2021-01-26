# Import packages
from Modules.utilities import download_datasets, split_dataset

# Download dataset
download_datasets()
split_dataset(test_size=100)

# # ====================================================================================================================
# # Data preprocessing
# data_train, data_val, data_test = data_preprocessing(args...)
# # ====================================================================================================================
# # Task A
# # Build model object.
# model_A = A(args...)
# # Train model based on the training set (you should fine-tune your model based on validation set).
# acc_A_train, acc_A_valid = model_A.train(args...)
# # Test model based on the test set.
# acc_A_test = model_A.test(args...)
# # Some code to free memory if necessary.
# Clean up memory/GPU etc...
#
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
