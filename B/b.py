# Import packages
from tensorflow.keras.layers import MaxPooling2D, Flatten, Dense
from tensorflow.keras import optimizers, Input, Model
from Modules.utilities import *
from tensorflow.keras.backend import get_value
from Modules.config import *
from Modules.components import *
from tqdm import tqdm


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
    filters = 16
    x1 = Conv2D(filters=filters, kernel_size=kernel_size, activation=activation, padding=padding)(inputs)
    x = ResidualBlock(filters=filters, kernel_size=kernel_size, activation=activation, padding=padding)(x1)
    x = ResidualBlock(filters=filters, kernel_size=kernel_size, activation=activation, padding=padding)(x)
    # x = ResidualBlock(filters=filters, kernel_size=kernel_size, activation=activation, padding=padding)(x)
    x = Conv2D(filters=filters, kernel_size=kernel_size, activation=activation, padding=padding)(x)
    x = Add()([x1, x])
    x = SubPixelConv2D(channels=8, scale=2, kernel_size=kernel_size, activation=activation, padding=padding)(x)
    x = SubPixelConv2D(channels=8, scale=2, kernel_size=kernel_size, activation=activation, padding=padding)(x)
    x = Conv2D(filters=3, kernel_size=kernel_size, padding=padding, activation='sigmoid')(x)
    return Model(inputs=inputs, outputs=x)


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
    x = Conv2D(filters=16, kernel_size=kernel_size, activation=activation, padding=padding)(inputs)
    x = Conv2D(filters=16, kernel_size=kernel_size, activation=activation, padding=padding)(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=2)(x)
    x = Conv2D(filters=32, kernel_size=kernel_size, activation=activation, padding=padding)(x)
    x = Conv2D(filters=32, kernel_size=kernel_size, activation=activation, padding=padding)(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=2)(x)
    x = Conv2D(filters=16, kernel_size=kernel_size, activation=activation, padding=padding)(x)
    x = Conv2D(filters=16, kernel_size=kernel_size, activation=activation, padding=padding)(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=2)(x)
    x = Flatten()(x)
    x = Dense(8, activation='relu')(x)
    x = Dense(1, activation='sigmoid')(x)
    return Model(inputs=inputs, outputs=x)


class B:
    def __init__(self, input_shape, loss='mse'):
        """
        Creates the Generative Adversarial Network model.

        :param input_shape: size of the input of the first layer.
        :param loss: the loss selected. It can be: 'mae', 'mse', ssim_loss and new_loss. default_value='mse'
        """
        # Input of the generative model
        inputs = Input(shape=input_shape)
        # Input shape of the discriminative model
        discriminator_input_shape = [input_shape[0] * 4, input_shape[1] * 4, 3]
        # Instantiate the generator and the discriminator
        self.generator = create_generator(kernel_size=(3, 3), activation='relu', padding='same')
        self.discriminator = create_discriminator(input_shape=discriminator_input_shape, kernel_size=(3, 3),
                                                  activation='relu', padding='same')
        # Compile both of them and print their summaries.
        self.generator.compile(loss=loss, optimizer=optimizers.Adam(learning_rate=0.001),
                               metrics=[psnr_metric, ssim_metric])
        self.generator.summary()
        self.discriminator.compile(loss="binary_crossentropy", optimizer=optimizers.Adam(learning_rate=0.001))
        self.discriminator.summary()
        # Join the two parts in order to obtain the GAN model
        x = self.generator(inputs)
        outputs = self.discriminator(x)
        self.model = Model(inputs=inputs, outputs=[x, outputs])
        # Compile the GAN model
        self.model.compile(loss=[loss, "binary_crossentropy"], optimizer=optimizers.Adam(learning_rate=0.001))

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
        gen_loss_mse, model_loss = [], []
        # Cycle on the epochs
        for epoch in range(1, epochs + 1):
            print(f'\nProcessing Epoch {epoch} out of {epochs} ... ')
            for _ in tqdm(range(steps_per_epoch), desc=f"Epoch {epoch}/{epochs}: "):
                # Train the discriminator with the SR images predicted by the generator
                self.discriminator.trainable = True
                # Load a new batch of images
                batch_LR, batch_HR = next(iter(training_batches))
                # Get the results from the generator, i.e. SR images
                batch_SR = self.generator.predict(batch_LR)
                # Labels are 0 since the images are not real but created by the generator
                labels = np.zeros(batch_dim, dtype=np.float32)
                # Compute the discriminator loss on the SR images
                loss = self.discriminator.train_on_batch(batch_SR, labels)
                dis_loss_SR.append(loss)
                # Train the discriminator with the real HR images. Labels are set to 1 given that the images are true
                labels = np.ones(batch_dim, dtype=np.float32)
                loss = self.discriminator.train_on_batch(batch_HR, labels)
                dis_loss_HR.append(loss)
                # # Train the generator
                self.discriminator.trainable = False
                loss = self.model.train_on_batch(batch_LR, [batch_HR, labels])
                gen_loss_mse.append(loss[0])
                model_loss.append(loss[1])
                # Compute metrics and loss after the update of the network parameters
                loss, psnr, ssim = self.generator.test_on_batch(x=batch_LR, y=batch_HR)
                train_loss.append(loss)
                train_psnr.append(psnr)
                train_ssim.append(ssim)
            # Print results after every epoch
            print(f' Discriminator loss_SR: {np.mean(dis_loss_SR[-steps_per_epoch:]):.4f} '
                  f'- loss_HR: {np.mean(dis_loss_HR[-steps_per_epoch:]):.4f} |',
                  f'Generator loss_MSE: {np.mean(gen_loss_mse[-steps_per_epoch:]):.4f} | '
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
            # Save the model after every 50 epochs
            if epoch % 50 == 0:
                self.generator.save(base_dir + f'Generator_Gan_{epoch}.h5')
                self.discriminator.save(base_dir + f'Discriminator_Gan_{epoch}.h5')
        # Plot Loss
        if plot:
            # Plot the loss and the metric values achieved on the training and validation datasets
            plot_learning(train_psnr, train_loss, valid_psnr, valid_loss)
        # Return the last metric values achieved on the training and validation datasets
        return np.mean(train_psnr[-steps_per_epoch:]), np.mean(train_ssim[-steps_per_epoch:])

    def evaluate(self, batches, n_batches=50):
        """
        Evaluates the generator model on the batches passed. It returns the loss and the psnr and ssim metrics.

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
        for step in range(n_batches):
            # ... the low nd high resolution images are extracted
            batch_LR, batch_HR = next(iter(batches))
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
        for batch in progressbar(test_batches, 'Status', 30, iterable=False, iterations=n_batches):
            # ... the low nd high resolution images are extracted
            lr_test = batch[0]
            hr_test = batch[1]
            # The predictions are performed on the low resolution images
            predictions = self.generator.predict_on_batch(x=lr_test)
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
