# Import packages
from tensorflow.keras import optimizers, Input, Model
from Modules.utilities import *
from tensorflow.keras.backend import get_value
from Modules.config import *
from Modules.pre_processing import prepare_custom_test_batches
import os
from Modules.components import *
from tensorflow.keras.applications import VGG16
from tensorflow.keras.backend import mean, square


class A:
    def __init__(self, input_shape, loss='mse'):
        """
        Creates the model.

        :param input_shape: size of the input of the first layer.
        :param loss: the loss selected. It can be: 'mae', 'mse', ssim_loss, vgg and new_loss. default_value='mse'
        """
        # If the loss selected is the content loss, aka perceptual loss or vgg loss.
        if loss == 'vgg':
            # Load the pre-trained model VGG16 without including the top
            model = VGG16(include_top=False)
            model.trainable = False
            self.vgg_model = Model([model.input], model.get_layer('block5_conv2').output, name='vggL')
            # Assign the vgg_loss as the loss adopted during the training phase
            loss = self.vgg_loss

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
        x = SubPixelConv2D(channels=16, scale=2, kernel_size=(3, 3), activation='relu', padding='same')(x)
        # The sigmoid activation function guarantees that the final output are within the range [0,1]
        outputs = Conv2D(filters=3, kernel_size=(3, 3), activation='sigmoid', padding='same')(x)

        self.model = Model(inputs, outputs)
        # Prints a summary of the network
        self.model.summary()
        # Configures the model for training
        self.model.compile(optimizer=optimizers.Adam(learning_rate=0.0001), loss=loss,
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

    def test(self, test_batches, plot=False, scale=4):
        """
        Generates output predictions for the examples passed and compares them with the true images returning
        the psnr and ssim metrics achieved.

        :param test_batches: input data passed as batches of n examples taken from the test dataset.
        :param plot: if True for every example considered a plot displaying the low-resolution, the bicubic
            interpolation, the prediction and the high-resolution images is shown. default_value=False
        :param scale: the up-scaling ratio desired. default_value=4
        :return: the test metric scores
        """
        # List of all the results
        results_psnr, results_ssim = [], []
        results_bicubic_psnr, results_bicubic_ssim = [], []
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
            # The PSNR and SSIM values are computed comparing the bicubic images with the related HR images
            bicubic_psnr, bicubic_ssim = compute_results_bicubic(lr_test, predictions, hr_test, scale=scale)
            results_bicubic_psnr.append(bicubic_psnr)
            results_bicubic_ssim.append(bicubic_ssim)
            # If plot is True the results are displayed [lr_image, bicubic, prediction, ground truth]
            if plot:
                _, _ = plot_results_bicubic(lr_test, predictions, hr_test, title=True, ax=False, scale=scale)
        # Compute the final result averaging all the values obtained
        results_psnr = np.hstack(results_psnr).mean()
        results_ssim = np.hstack(results_ssim).mean()
        results_bicubic_psnr = np.hstack(results_bicubic_psnr).mean()
        results_bicubic_ssim = np.hstack(results_bicubic_ssim).mean()
        print('\nTest executed on the following test datasets:',
              '\n{:<10} {:<25} {:<25}'.format('Dataset', 'Bicubic', 'Model'),
              '\n{:<10} {:<25} {:<25}'.format('DIV2k', f'{results_bicubic_psnr:.4f} / {results_bicubic_ssim:.4f}',
                                              f'{results_psnr:.4f} / {results_ssim:.4f}'))
        return results_psnr, results_ssim

    def additional_tests(self, plot=False, scale=4):
        """
        Generates output predictions for the examples from the additional test_datasets and compares them with the true
        images returning the psnr and ssim metrics achieved by the model and through a bicubic interpolation.

        :param plot: if True for every example considered a plot displaying the low-resolution, the bicubic
            interpolation, the prediction and the high-resolution images is shown. default_value=False
        :param scale: the up-scaling ratio desired. default_value=4
        :return:
        """
        # List of the folders containing the test datasets
        test_datasets_folders = ['Set5', 'Set14', 'BSD100']
        # For every folder name
        for folder in test_datasets_folders:
            # Find the folder path
            folder_path = os.path.join(base_dir, folder)
            # Prepare the batches related to the current test dataset
            batches = prepare_custom_test_batches(folder_path, patch_size, scale=scale)
            # Iterate across the batches generated to get the results
            results_psnr, results_ssim = [], []
            results_bicubic_psnr, results_bicubic_ssim = [], []
            for batch in batches:
                # ... the low nd high resolution images are extracted from the current batch
                lr_test = batch[0]
                hr_test = batch[1]
                # The predictions are performed on the low resolution images
                predictions = self.model.predict_on_batch(x=lr_test)
                # The PSNR and SSIM values are computed comparing the predictions with the related HR images
                results_psnr.append(get_value(psnr_metric(hr_test, predictions)))
                results_ssim.append(get_value(ssim_metric(hr_test, predictions)))
                # The PSNR and SSIM values are computed comparing the bicubic images with the related HR images
                bicubic_psnr, bicubic_ssim = compute_results_bicubic(lr_test, predictions, hr_test, scale=scale)
                results_bicubic_psnr.append(bicubic_psnr)
                results_bicubic_ssim.append(bicubic_ssim)
                # If plot is True the results are displayed as well. [lr_image, bicubic, prediction, ground truth]
                if plot:
                    _, _ = plot_results_bicubic(lr_test, predictions, hr_test, title=True, ax=False, scale=scale)
            # Compute the final result averaging all the values obtained
            results_psnr = np.hstack(results_psnr).mean()
            results_ssim = np.hstack(results_ssim).mean()
            results_bicubic_psnr = np.hstack(results_bicubic_psnr).mean()
            results_bicubic_ssim = np.hstack(results_bicubic_ssim).mean()
            print('{:<10} {:<25} {:<25}'.format(folder, f'{results_bicubic_psnr:.4f} / {results_bicubic_ssim:.4f}',
                                                f'{results_psnr:.4f} / {results_ssim:.4f}'))

    def new_scale(self, scale=2, loss='mae'):
        """
        Changes only the last layer of the model to support different up-scaling ratios.

        :param scale: the up-scaling ratio desired. default_value=2
        :param loss: the loss selected. It can be: 'mae', 'mse', ssim_loss and new_loss. default_value='mse'
        :return:
        """
        # If the loss required is the perceptual loss
        if loss == 'vgg':
            loss = self.vgg_loss
        # Remove the last two layers of the model
        x = self.model.layers[-3].output
        # Replace them with the new layers
        x = SubPixelConv2D(channels=16, scale=scale, kernel_size=(3, 3), activation='relu', padding='same')(x)
        # The sigmoid activation function guarantees that the final output are within the range [0,1]
        outputs = Conv2D(filters=3, kernel_size=(3, 3), activation='sigmoid', padding='same')(x)
        # Over subscribe the old model with the new model
        self.model = Model(inputs=self.model.input, outputs=outputs)
        self.model.summary()
        # Configures the model for training
        self.model.compile(optimizer=optimizers.Adam(learning_rate=0.0001), loss=loss,
                           metrics=[psnr_metric, ssim_metric])

    def vgg_loss(self, image_true, image_pred):
        """
        Evaluates the image quality based on its perceptual quality comparing the high level features of the
        generated image and the ground truth image. The features are extracted from the outputs of one of the
        middle-final layers of the vgg network.

        :param image_true: ground truth image.
        :param image_pred: predicted image.
        :return:
        """
        # Extract high features from the predicted image
        features_pred = self.vgg_model(image_pred)
        # Extract high features from the true image
        features_true = self.vgg_model(image_true)
        # Return the mean square error computed on the images representations extracted plus the original MSE
        loss = mean(square(features_true - features_pred)) + mean(square(image_true - image_pred))
        return loss
