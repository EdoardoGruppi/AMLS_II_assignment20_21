# Import packages
from tensorflow.keras.layers import Flatten, MaxPooling2D
from tensorflow.keras import optimizers, Input, Model
from Modules.utilities import *
from tensorflow.keras.backend import get_value
from Modules.pre_processing import prepare_custom_test_batches
from Modules.config import *
from Modules.components import *
from tqdm import tqdm
from tensorflow.keras.applications import VGG16
from tensorflow.keras.backend import mean, square
import numpy as np


def create_generator(kernel_size=(3, 3), activation='relu', padding='same'):
    """
    Creates the generator model whose aim is to generate Super Resolution images starting from Low Resolution pictures.

    :param kernel_size: the height and width of the filters of the 2D convolution layers. default_value=(3, 3)
    :param activation: activation function used. default_value='relu'
    :param padding: which padding to apply. It can be 'same' or 'valid'. default_value='same'
    :return: the generator model.
    """
    inputs = Input(shape=(None, None, 3))
    # Since the residual blocks have skip connection inside it is necessary that their inputs are equal to their
    # outputs in terms of depth, width and height.
    x = DifferenceRGB(RGB_MEAN_A)(inputs)
    x1 = Conv2D(filters=32, kernel_size=kernel_size, activation=activation, padding=padding)(x)
    x = ResidualBlock(filters=32, kernel_size=kernel_size, scaling=None, activation=activation, padding=padding)(x1)
    x = ResidualBlock(filters=32, kernel_size=kernel_size, scaling=None, activation=activation, padding=padding)(x)
    x = ResidualBlock(filters=32, kernel_size=kernel_size, scaling=None, activation=activation, padding=padding)(x)
    x = ResidualBlock(filters=32, kernel_size=kernel_size, scaling=None, activation=activation, padding=padding)(x)
    x = ResidualBlock(filters=32, kernel_size=kernel_size, scaling=None, activation=activation, padding=padding)(x)
    x = ResidualBlock(filters=32, kernel_size=kernel_size, scaling=None, activation=activation, padding=padding)(x)
    x = Add()([x, x1])
    x = ResidualBlock(filters=32, kernel_size=kernel_size, scaling=None, activation=activation, padding=padding)(x)
    x = ResidualBlock(filters=32, kernel_size=kernel_size, scaling=None, activation=activation, padding=padding)(x)
    x = Add()([x, x1])
    x = SubPixelConv2D(channels=16, scale=2, kernel_size=kernel_size, activation=activation, padding=padding)(x)
    # The sigmoid activation function guarantees that the final output are within the range [0,1]
    outputs = Conv2D(filters=3, kernel_size=(3, 3), activation='sigmoid', padding='same')(x)
    return Model(inputs=inputs, outputs=outputs, name='Generator')


def create_discriminator(input_shape, kernel_size=(3, 3), activation='relu', padding='same'):
    """
    Creates the discriminator model whose aim is to distinguish the High Resolution (i.e. ground truth) images and
    the Super Resolution pictures created by the generator.

    :param input_shape: shape of the HR and SR images.
    :param kernel_size: the height and width of the filters of the 2D convolution layers. default_value=(3, 3)
    :param activation: activation function used. default_value='relu'
    :param padding: which padding to apply. It can be 'same' or 'valid'. default_value='same'
    :return: the discriminator model.
    """
    inputs = Input(input_shape)
    x = Conv2D(filters=32, kernel_size=kernel_size, activation=activation, padding=padding)(inputs)
    x = Conv2D(filters=32, kernel_size=kernel_size, activation=activation, padding=padding)(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=2)(x)
    x = Conv2D(filters=32, kernel_size=kernel_size, activation=activation, padding=padding)(x)
    x = Conv2D(filters=32, kernel_size=kernel_size, activation=activation, padding=padding)(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=2)(x)
    x = Conv2D(filters=32, kernel_size=kernel_size, activation=activation, padding=padding)(x)
    x = Conv2D(filters=32, kernel_size=kernel_size, activation=activation, padding=padding)(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=2)(x)
    x = Flatten()(x)
    x = Dense(1, activation='sigmoid')(x)
    return Model(inputs=inputs, outputs=x, name='Discriminator')


class B:
    def __init__(self, input_shape, loss='mse', scale=4, loss_weights=None):
        """
        Creates the Generative Adversarial Network model.

        :param input_shape: size of the input of the first layer.
        :param scale: the up-scaling ratio desired. default_value=4
        :param loss: the loss selected. It can be: 'mae', 'mse', ssim_loss, vgg and new_loss. default_value='mse'
        :param loss_weights: weights assigned to the loss functions of the GAN model. default_value=None
        """
        # If the loss selected is the content loss, aka perceptual loss or vgg loss.
        if loss == 'vgg':
            # Load the pre-trained model VGG16 without including the top
            model = VGG16(include_top=False)
            model.trainable = False
            self.vgg_model = Model([model.input], model.get_layer('block5_conv2').output, name='Vgg_model')
            # Assign the vgg_loss as the loss adopted during the training phase
            loss = self.vgg_loss
            # Reduce the weight of the binary crossentropy loss of the discriminator
            loss_weights = [1., 1e-3]

        # Input of the generative model
        inputs = Input(shape=input_shape)
        # Input shape of the discriminative model
        discriminator_input_shape = [input_shape[0] * scale, input_shape[1] * scale, 3]
        # Instantiate the generator and the discriminator
        self.generator = create_generator(kernel_size=(3, 3), activation='relu', padding='same')
        self.discriminator = create_discriminator(input_shape=discriminator_input_shape, kernel_size=(3, 3),
                                                  activation='relu', padding='same')
        # Compile both of them and print their summaries.
        self.generator.compile(loss=loss, optimizer=optimizers.Adam(learning_rate=0.0001),
                               metrics=[psnr_metric, ssim_metric])
        self.discriminator.compile(loss="binary_crossentropy", optimizer=optimizers.Adam(learning_rate=0.0001))
        self.generator.summary()
        self.discriminator.summary()
        # Join the two parts in order to obtain the GAN model
        x = self.generator(inputs)
        outputs = self.discriminator(x)
        self.model = Model(inputs=inputs, outputs=[x, outputs], name='Generative_Adversarial_Network')
        # Compile the GAN model
        self.model.compile(loss=[loss, "binary_crossentropy"], optimizer=optimizers.Adam(learning_rate=0.0001),
                           loss_weights=loss_weights)
        self.model.summary()

    def train(self, training_batches, valid_batches, epochs=25, plot=True):
        """
        Trains the model for a fixed number of iterations on the entire dataset (epochs).

        :param training_batches: input data passed as batches of n examples.
        :param valid_batches: batches of examples on which to evaluate the generalisation capability of the model and
            other important key elements. The model is not trained on them.
        :param epochs: number of epochs to train the model. default_value=25
        :param plot: if True it plots the learning and performance curves. default_value=True
        :return: the last metric values measured on the training and validation sets.
        """
        # Number of batches to work on for each epoch
        steps_per_epoch = int((800 - test_dim) / batch_dim)
        validation_steps = int(100 / batch_dim)
        valid_loss, valid_psnr, valid_ssim = [], [], []
        train_loss, train_psnr, train_ssim = [], [], []
        dis_loss_SR, dis_loss_HR = [], []
        gen_loss, model_loss = [], []
        # Cycle on the epochs
        for epoch in range(1, epochs + 1):
            print(f'\nProcessing Epoch {epoch} out of {epochs} ... ')
            for counter, batch in tqdm(enumerate(training_batches), desc=f"Epoch {epoch}/{epochs}: ",
                                       total=steps_per_epoch):
                if counter == steps_per_epoch:
                    break
                # Train the discriminator with the SR images predicted by the generator
                self.discriminator.trainable = True
                # Load a new batch of images
                batch_LR = batch[0]
                batch_HR = batch[1]
                # Get the results from the generator, i.e. SR images
                batch_SR = self.generator.predict(batch_LR)
                # Labels are near 0 since the images are not real but created by the generator. Label smoothing: soft
                # values are adopted in order to control network confidence and improving Gan training.
                labels = np.random.uniform(0.0, 0.3, size=batch_dim).astype(np.float32)
                # Compute the discriminator loss on the SR images and updates the weights to minimize it.
                # The loss here corresponds to the following loss computed on all the examples in the batch and summed.
                # loss_single_sample = -(label * log(D(G(z))) + (1-label) * log(1 - D(G(z))))
                # Both the terms are kept since label smoothing. Discriminator wants D(G(z)) low.
                loss = self.discriminator.train_on_batch(batch_SR, labels)
                dis_loss_SR.append(loss)
                # Train the discriminator with the real HR images. Labels are set near to 1 given that the images are
                # true. Label smoothing: soft values are adopted in order to control network confidence and improve GAN
                # training.
                labels = np.random.uniform(0.7, 1, size=batch_dim).astype(np.float32)
                # Compute the discriminator loss on the SR images and updates the weights to minimize it.
                # The loss here corresponds to the following loss computed on all the examples in the batch and summed.
                # loss_single_sample = -(label * log(D(x)) + (1-label) * log(1 - D(x)))
                # Both the terms are kept since label smoothing. Discriminator wants D(x) high.
                loss = self.discriminator.train_on_batch(batch_HR, labels)
                dis_loss_HR.append(loss)
                # Train the generator
                self.discriminator.trainable = False
                labels = np.ones(size=batch_dim, dtype=np.float32)
                # Compute the generator loss (discriminator is fixed) on the SR images and updates the weights to
                # minimize it. The adversarial loss, that is only a part of the generator loss,  here corresponds to
                # the following loss computed on all the examples in the batch and summed.
                # loss_single_sample = -(label * log(D(G(z))) + (1-label) * log(1 - D(x))) = -log(D(G(z)))
                # Generator wants D(G(z)) high to fool the discriminator
                loss = self.model.train_on_batch(batch_LR, [batch_HR, labels])
                gen_loss.append(loss[0])
                model_loss.append(loss[1])
                # Compute metrics and loss after the update of the network parameters
                loss, psnr, ssim = self.generator.test_on_batch(x=batch_LR, y=batch_HR)
                train_loss.append(loss)
                train_psnr.append(psnr)
                train_ssim.append(ssim)
            # Print results after every epoch
            print(f' Discriminator loss_SR: {np.mean(dis_loss_SR[-steps_per_epoch:]):.4f} '
                  f'- loss_HR: {np.mean(dis_loss_HR[-steps_per_epoch:]):.4f} |',
                  f'Generator loss: {np.mean(gen_loss[-steps_per_epoch:]):.4f} | '
                  f'Model loss: {np.mean(model_loss[-steps_per_epoch:]):.4f}')
            print(f' Train Model loss: {np.mean(train_loss[-steps_per_epoch:]):.4f} '
                  f'- Train psnr: {np.mean(train_psnr[-steps_per_epoch:]):.4f} '
                  f'- Train ssim: {np.mean(train_ssim[-steps_per_epoch:]):.4f}')
            # Get and print the validation results (i.e. metrics and loss)
            loss, psnr, ssim = self.evaluate(batches=valid_batches, n_batches=validation_steps)
            print(f' Valid Model loss: {loss:.4f} - Valid psnr: {psnr:.4f} - Valid ssim: {ssim:.4f}')
            valid_loss.append(loss)
            valid_psnr.append(psnr)
            valid_ssim.append(ssim)
        # Plot Loss
        if plot:
            # Plot the loss and the metric values achieved on the training and validation datasets
            plot_learning(train_psnr, train_loss, valid_psnr, valid_loss)
        # Return the last metric values achieved on the training and validation datasets
        return np.mean(train_psnr[-steps_per_epoch:]), np.mean(valid_psnr)

    def evaluate(self, batches, n_batches=50):
        """
        Evaluates the generator model on the batches passed. It returns the loss and the psnr and ssim metrics. This
        function is used to evaluate the model on the validation dataset during the learning phase.

        :param batches: batches to evaluate (for instance: validation batches).
        :param n_batches: number of batches to evaluate. This parameter is necessary when working on tf.data.Dataset
        repeated objects. default_value=50
        :return: the loss and the psnr and ssim metrics computed on the batches passed.
        """
        # List of all the results
        results_psnr = []
        results_ssim = []
        loss_list = []
        # For every batch...
        for counter, batch in enumerate(batches):
            if counter == n_batches:
                break
            # ... the low nd high resolution images are extracted
            batch_LR = batch[0]
            batch_HR = batch[1]
            # The predictions are performed on the low resolution images
            loss, psnr, ssim = self.generator.test_on_batch(x=batch_LR, y=batch_HR)
            # The PSNR and SSIM values are computed comparing the predictions with the related high resolution images
            results_psnr.append(psnr)
            results_ssim.append(ssim)
            loss_list.append(loss)
        # Compute the mean of the loss and metric values measured
        loss = np.mean(loss_list)
        results_psnr = np.mean(results_psnr)
        results_ssim = np.mean(results_ssim)
        return loss, results_psnr, results_ssim

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
            predictions = self.generator.predict_on_batch(x=lr_test)
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
                predictions = self.generator.predict_on_batch(x=lr_test)
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

    def new_scale(self, input_shape, scale=2, loss='mae'):
        """
        Changes only the last layer of the model to support different up-scaling ratios.

        :param input_shape: size of the input of the first layer.
        :param scale: the up-scaling ratio desired. default_value=2
        :param loss: the loss selected. It can be: 'mae', 'mse', ssim_loss and new_loss. default_value='mse'
        :return:
        """
        # New Generator
        # If the loss required is the perceptual loss
        if loss == 'vgg':
            loss = self.vgg_loss
        # Remove the last two layers of the generator
        x = self.generator.layers[-3].output
        # Replace them with the new layers
        x = SubPixelConv2D(channels=16, scale=scale, kernel_size=(3, 3), activation='relu', padding='same')(x)
        # The sigmoid activation function guarantees that the final output are within the range [0,1]
        outputs = Conv2D(filters=3, kernel_size=(3, 3), activation='sigmoid', padding='same')(x)
        # Over subscribe the old generator with the new generator
        self.generator = Model(inputs=self.generator.input, outputs=outputs)
        # Configures the model for training
        self.generator.compile(optimizer=optimizers.Adam(learning_rate=0.0001), loss=loss,
                               metrics=[psnr_metric, ssim_metric])
        self.generator.summary()
        # New Discriminator
        # Get the size of the input of the new discriminator
        discriminator_input_shape = [input_shape[0] * scale, input_shape[1] * scale, 3]
        # Get the size of the input of the current discriminator
        current_input_shape = self.discriminator.input[0][0].shape[0]
        # Compute the down-scaling that is needed to transform the new input shape to the old input shape
        difference = current_input_shape / discriminator_input_shape[0]
        # Add before the old discriminator model a custom bicubic down sampling layer
        inputs = Input(discriminator_input_shape)
        # Downscale the input to match the original input shape of the discriminator
        x = BicubicUpSampling2D(scale=difference, image_size=discriminator_input_shape[0])(inputs)
        # Connect the layer at the old discriminator
        outputs = self.discriminator(x)
        self.discriminator = Model(inputs=inputs, outputs=outputs)
        self.discriminator.compile(loss="binary_crossentropy", optimizer=optimizers.Adam(learning_rate=0.0001))
        self.discriminator.summary()
        # New Model
        # Join the two parts in order to obtain the new GAN model
        inputs = Input(input_shape)
        x = self.generator(inputs)
        outputs = self.discriminator(x)
        self.model = Model(inputs=inputs, outputs=[x, outputs])
        # Compile the GAN model
        self.model.compile(loss=[loss, "binary_crossentropy"], optimizer=optimizers.Adam(learning_rate=0.0001))
        self.model.summary()

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
