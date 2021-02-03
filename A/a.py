# Import packages
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, UpSampling2D
from tensorflow.keras import optimizers
from Modules.utilities import psnr_metric, plot_history, progressbar, plot_results, plot_results_bicubic
from tensorflow.keras.backend import get_value
from Modules.config import *


class A:
    def __init__(self, input_shape):
        """
        Creates the object of the model.

        :param input_shape: size of the first layer input
        """
        self.model = Sequential([
            Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same', input_shape=input_shape),
            Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same'),
            UpSampling2D(),
            Conv2D(filters=16, kernel_size=(3, 3), activation='relu', padding='same'),
            Conv2D(filters=16, kernel_size=(3, 3), activation='relu', padding='same'),
            UpSampling2D(),
            # The sigmoid activation function guarantees that the final output are within the range [0,1]
            Conv2D(filters=3, kernel_size=(3, 3), activation='sigmoid', padding='same')
        ])
        # Prints a summary of the network
        self.model.summary()
        # Configures the model for training
        self.model.compile(optimizer=optimizers.Adam(learning_rate=0.01), loss='mse', metrics=[psnr_metric])

    def train(self, training_batches, valid_batches, epochs=25, verbose=1, plot=True):
        """
        Trains the model for a fixed number of iterations on the entire dataset (epochs).

        :param training_batches: input data passed as batches of n examples.
        :param valid_batches: batches of examples on which to evaluate the loss and model metrics after each epoch.
            The model is not trained on them.
        :param epochs: number of epochs utilised to train the model. default_value=25
        :param verbose: verbosity level. default_value=1.
        :param plot: if True it plots the learning and performance curves. default_value=True
        :return: the last metric values measured on the training and validation sets.
        """
        # Trains the model for a fixed number of epochs
        history = self.model.fit(x=training_batches,
                                 steps_per_epoch=int((800 - test_dim) / batch_dim),
                                 validation_data=valid_batches,
                                 validation_steps=int(200 / batch_dim),
                                 epochs=epochs,
                                 verbose=verbose)
        if plot:
            # Plot the loss and the metric values achieved on the training and validation datasets
            plot_history(history.history['psnr_metric'], history.history['val_psnr_metric'], history.history['loss'],
                         history.history['val_loss'])
        # Return the last metric values achieved on the training and validation datasets
        return history.history['psnr_metric'][-1], history.history['val_psnr_metric'][-1]

    def test(self, test_batches, plot=None):
        """
        Generates output predictions for the examples passed and compares them with the true images returning
        the psnr metric gained.

        :param test_batches: input data passed as batches of n examples taken from the test dataset.
        :param plot: if 'normal' for every example considered a plot displaying the low-resolution, the prediction and
            the high-resolution images is shown. Otherwise, if 'bicubic' the bicubic up-sampling is also displayed.
            default_value=None
        :return: the test metric score
        """
        # List of all the results
        results = []
        results_bicubic = []
        # Number of batches for the test set
        n_batches = int(test_dim / batch_dim)
        print('\nTesting phase started...')
        # For every batch...
        for batch in progressbar(test_batches, 'Status', 30, iterable=False, iterations=n_batches):
            # ... the low nd high resolution images are extracted
            lr_test = batch[0]
            hr_test = batch[1]
            # The predictions are performed on the low resolution images
            predictions = self.model.predict_on_batch(x=lr_test)
            # The PSNR values are computed comparing the predictions made with the related high resolution images
            results.append(get_value(psnr_metric(hr_test, predictions)))
            # If plot is not None the results are displayed [lr_image, (bicubic), prediction, ground truth]
            if plot == 'normal':
                plot_results(lr_test, predictions, hr_test, title=True, ax=False)
            if plot == 'bicubic':
                bicubic_results = plot_results_bicubic(lr_test, predictions, hr_test, title=True, ax=False)
                results_bicubic.append(bicubic_results)
        if plot == 'bicubic':
            print(f'Metric achieved trough simple Bicubic interpolation: {np.array(results_bicubic).mean()}')
        # Compute the final result averaging all the values obtained
        results = np.array(results).mean()
        return results

