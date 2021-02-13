# Import packages
from tensorflow.keras import optimizers, Input, Model
from Modules.utilities import *
from tensorflow.keras.backend import get_value
from Modules.config import *
from Modules.components import *


class A:
    def __init__(self, input_shape, loss='mse'):
        """
        Creates the model.

        :param input_shape: size of the input of the first layer.
        :param loss: the loss selected. It can be: 'mae', 'mse', ssim_loss and new_loss. default_value='mse'
        """
        inputs = Input(shape=input_shape)
        x = DifferenceRGB(RGB_MEAN_A)(inputs)
        x1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = ResidualBlock(filters=32, kernel_size=(3, 3), scaling=None, activation='relu', padding='same')(x1)
        x = ResidualBlock(filters=32, kernel_size=(3, 3), scaling=None, activation='relu', padding='same')(x)
        x = ResidualBlock(filters=32, kernel_size=(3, 3), scaling=None, activation='relu', padding='same')(x)
        x = ResidualBlock(filters=32, kernel_size=(3, 3), scaling=None, activation='relu', padding='same')(x)
        x = ResidualBlock(filters=32, kernel_size=(3, 3), scaling=None, activation='relu', padding='same')(x)
        x = ResidualBlock(filters=32, kernel_size=(3, 3), scaling=None, activation='relu', padding='same')(x)
        x = Add()([x, x1])
        x = ResidualBlock(filters=32, kernel_size=(3, 3), scaling=None, activation='relu', padding='same')(x)
        x = ResidualBlock(filters=32, kernel_size=(3, 3), scaling=None, activation='relu', padding='same')(x)
        x = Add()([x, x1])
        x = ResidualBlock(filters=32, kernel_size=(3, 3), scaling=None, activation='relu', padding='same')(x)
        x = ResidualBlock(filters=32, kernel_size=(3, 3), scaling=None, activation='relu', padding='same')(x)
        x = Add()([x, x1])
        x = ResidualBlock(filters=32, kernel_size=(3, 3), scaling=None, activation='relu', padding='same')(x)
        x = ResidualBlock(filters=32, kernel_size=(3, 3), scaling=None, activation='relu', padding='same')(x)
        x = Add()([x, x1])
        x = SubPixelConv2D(channels=16, scale=2, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = SubPixelConv2D(channels=16, scale=2, kernel_size=(3, 3), activation='relu', padding='same')(x)
        # The sigmoid activation function guarantees that the final output are within the range [0,1]
        outputs = Conv2D(filters=3, kernel_size=(3, 3), activation='sigmoid', padding='same')(x)

        self.model = Model(inputs, outputs)
        # Prints a summary of the network
        self.model.summary()
        # Configures the model for training
        self.model.compile(optimizer=optimizers.Adam(learning_rate=0.001), loss=loss,
                           metrics=[psnr_metric, ssim_metric])

    def train(self, training_batches, valid_batches, epochs=25, verbose=1, plot=True):
        """
        Trains the model for a fixed number of iterations on the entire dataset (epochs).

        :param training_batches: input data passed as batches of n examples.
        :param valid_batches: batches of examples on which to evaluate the generalisation capability of the model and
            other important key elements. The model is not trained on them.
        :param epochs: number of epochs to train the model. default_value=25
        :param verbose: verbosity level. default_value=1.
        :param plot: if True it plots the learning and performance curves. default_value=True
        :return: the last metric values measured on the training and validation sets.
        """
        # Trains the model for a fixed number of epochs
        history = self.model.fit(x=training_batches,
                                 steps_per_epoch=int((800 - test_dim) / batch_dim),
                                 validation_data=valid_batches,
                                 validation_steps=int(100 / batch_dim),
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
        the psnr and ssim metrics achieved.

        :param test_batches: input data passed as batches of n examples taken from the test dataset.
        :param plot: if 'normal' for every example considered a plot displaying the low-resolution, the prediction and
            the high-resolution images is shown. Otherwise, if 'bicubic' the bicubic up-sampling is also displayed.
            default_value=None
        :return: the test metric scores
        """
        # List of all the results
        results_psnr = []
        results_ssim = []
        results_bicubic_psnr = []
        results_bicubic_ssim = []
        # Number of batches for the test set
        n_batches = int(test_dim / batch_dim)
        print('\nTesting phase started...')
        # For every batch...
        for batch in progressbar(test_batches, iterations=n_batches):
            # ... the low nd high resolution images are extracted
            lr_test = batch[0]
            hr_test = batch[1]
            # The predictions are performed on the low resolution images
            predictions = self.model.predict_on_batch(x=lr_test)
            # The PSNR and SSIM values are computed comparing the predictions with the related high resolution images
            results_psnr.append(get_value(psnr_metric(hr_test, predictions)))
            results_ssim.append(get_value(ssim_metric(hr_test, predictions)))
            # If plot is not None the results are displayed [lr_image, (bicubic), prediction, ground truth]
            if plot == 'normal':
                plot_results(lr_test, predictions, hr_test, title=True, ax=False)
            if plot == 'bicubic':
                bicubic_psnr, bicubic_ssim = plot_results_bicubic(lr_test, predictions, hr_test, title=True, ax=False)
                results_bicubic_psnr.append(bicubic_psnr)
                results_bicubic_ssim.append(bicubic_ssim)
        if plot == 'bicubic':
            print(f'\nPSNR achieved trough simple Bicubic interpolation: {np.array(results_bicubic_psnr).mean():.4f}',
                  f'\nSSIM achieved trough simple Bicubic interpolation: {np.array(results_bicubic_ssim).mean():.4f}')
        # Compute the final result averaging all the values obtained
        results_psnr = np.array(results_psnr).mean()
        results_ssim = np.array(results_ssim).mean()
        return results_psnr, results_ssim
