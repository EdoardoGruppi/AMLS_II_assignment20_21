# Import packages
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, MaxPooling2D, BatchNormalization, Conv2D, UpSampling2D, Conv2DTranspose
from tensorflow.keras import optimizers
from Modules.utilities import psnr_metric, plot_history


class A:
    def __init__(self, input_shape):
        self.model = Sequential([
            Conv2D(filters=8, kernel_size=(3, 3), activation='relu', padding='same', input_shape=input_shape),
            Conv2D(filters=8, kernel_size=(3, 3), activation='relu', padding='same'),
            UpSampling2D(),
            Conv2D(filters=16, kernel_size=(3, 3), activation='relu', padding='same'),
            Conv2D(filters=16, kernel_size=(3, 3), activation='relu', padding='same'),
            UpSampling2D(),
            Conv2D(filters=8, kernel_size=(3, 3), activation='relu', padding='same'),
            Conv2D(filters=3, kernel_size=(3, 3), activation='relu', padding='same')
        ])
        self.model.summary()
        self.model.compile(optimizer=optimizers.Adam(learning_rate=0.01), loss='mse',
                           metrics=[psnr_metric])

    def train(self, training_batches, valid_batches, epochs=25, verbose=1, plot=True):
        history = self.model.fit(x=training_batches,
                                 steps_per_epoch=len(training_batches),
                                 validation_data=valid_batches,
                                 validation_steps=len(valid_batches),
                                 epochs=epochs,
                                 verbose=verbose
                                 )
        if plot:
            # Plot loss and accuracy achieved on training and validation dataset
            plot_history(history.history['psnr_metric'], history.history['val_psnr_metric'], history.history['loss'],
                         history.history['val_loss'])
        return history.history['psnr_metric'][-1], history.history['val_psnr_metric'][-1]

