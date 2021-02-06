# Import packages
from tensorflow.keras.layers import Layer, Conv2D, Lambda, Add, BatchNormalization
from tensorflow import nn, image, reduce_mean
from tensorflow.keras.losses import MSE
import tensorflow as tf


class SubPixelConv2D(Layer):
    def __init__(self, channels=3, scale=2, kernel_size=(3, 3), activation='relu', padding='same'):
        """
        SubPixelConv2D layer created according to the information retrieved from the paper
        https://arxiv.org/pdf/1609.05158.pdf.

        :param channels: number of channels of the output images. default_value=3
        :param scale: upscale factor. default_value=2
        :param kernel_size: the height and width of the 2D convolution filter. default_value=(3, 3)
        :param activation: activation function used. default_value='relu'
        :param padding: which padding to apply. It can be 'same' or 'valid'. default_value='same'
        """
        super(SubPixelConv2D, self).__init__()
        self.kernel_size = kernel_size
        self.scale = scale
        self.padding = padding
        self.channels = channels
        self.activation = activation
        self.conv = Conv2D(self.channels * (self.scale ** 2), kernel_size=self.kernel_size, activation=self.activation,
                           padding=self.padding)

    @tf.function
    def call(self, inputs):
        x = self.conv(inputs)
        outputs = nn.depth_to_space(x, self.scale)
        return outputs


class ResidualBlock(Layer):
    def __init__(self, filters=16, scaling=None, kernel_size=(3, 3), activation='relu', padding='same'):
        """
        Residual block created according to the information retrieved from the paper
        https://arxiv.org/pdf/1707.02921.pdf. Differently from the versions of the residual block presented in the
        papers https://arxiv.org/pdf/1512.03385v1.pdf and https://bit.ly/3jmuLxo, in this case there is no
        BatchNormalization layer involved as well as no activation function after the skip connection.

        :param filters: the number of output filters in the convolution. default_value=16
        :param scaling: factor defining the constant scaling layer. default_value=None
        :param kernel_size: the height and width of the 2D convolution filter. default_value=(3, 3)
        :param activation: activation function used. default_value='relu'
        :param padding: which padding to apply. It can be 'same' or 'valid'. default_value='same'
        """
        super(ResidualBlock, self).__init__()
        self.kernel_size = kernel_size
        self.scaling = scaling
        self.padding = padding
        self.filters = filters
        self.activation = activation
        self.conv1 = Conv2D(filters=self.filters, kernel_size=self.kernel_size, activation=self.activation,
                            padding=self.padding)
        self.conv2 = Conv2D(filters=self.filters, kernel_size=self.kernel_size, activation=None, padding=self.padding)

    @tf.function
    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        # The scaling allows to stabilise the training when increasing the width of the architecture
        if self.scaling:
            x = Lambda(lambda layer: layer * self.scaling)(x)
        outputs = Add()([inputs, x])
        return outputs


class SRResNetBlock(Layer):
    def __init__(self, filters=16, momentum=0.5, kernel_size=(3, 3), activation='relu', padding='same'):
        """
        Residual block created according to the information retrieved from the paper https://bit.ly/3jmuLxo.

        :param filters: the number of output filters in the convolution. default_value=16
        :param momentum: Momentum for the moving average of the BatchNormalization layer. default_value=0.5
        :param kernel_size: the height and width of the 2D convolution filter. default_value=(3, 3)
        :param activation: activation function used. default_value='relu'
        :param padding: which padding to apply. It can be 'same' or 'valid'. default_value='same'
        """
        super(SRResNetBlock, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.momentum = momentum
        self.filters = filters
        self.activation = activation
        self.conv1 = Conv2D(filters=self.filters, kernel_size=self.kernel_size, activation=self.activation,
                            padding=self.padding)
        self.conv2 = Conv2D(filters=self.filters, kernel_size=self.kernel_size, activation=None, padding=self.padding)
        self.batch1 = BatchNormalization(momentum=momentum)
        self.batch2 = BatchNormalization(momentum=momentum)

    @tf.function
    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.batch1(x)
        x = self.conv2(x)
        x = self.batch2(x)
        outputs = Add()([inputs, x])
        return outputs


def psnr_metric(true_img, pred_img):
    """
    Computes the Peak Signal-to-Noise Ratio (PSNR) on the images passed. The input can be a list of images
    as well. Nevertheless, in this last case the two lists must have the same length.

    :param true_img: real image/images.
    :param pred_img: predicted image/images.
    :return: the psnr metric computed on the inputs.
    """
    # The metric is computed exploiting the tf.image.psnr function provided by tensorflow
    return image.psnr(true_img, pred_img, max_val=1)


def ssim_metric(true_img, pred_img):
    """
    Computes the Structural Similarity Index Measure (SSIM) on the images passed. The input can be a list of images
    as well. Nevertheless, in this last case the two lists must have the same length.

    :param true_img: real image/images.
    :param pred_img: predicted image/images.
    :return: the ssim metric computed on the inputs.
    """
    # The metric is computed exploiting the tf.image.ssim function provided by tensorflow
    return image.ssim(true_img, pred_img, max_val=1)


def ssim_loss(true_img, pred_img):
    """
    Computes the loss using the Structural Similarity Index Measure (SSIM) computed on the images passed. The input
    can be a list of images as well. Nevertheless, in this last case the two lists must have the same length.

    :param true_img: real image/images.
    :param pred_img: predicted image/images.
    :return: the ssim loss computed on the inputs.
    """
    return 1 - reduce_mean(image.ssim(true_img, pred_img, max_val=1.0))


def new_loss(true_img, pred_img):
    weight = 3
    loss_mse = reduce_mean(MSE(true_img, pred_img))
    loss_ssim = ssim_loss(true_img, pred_img)
    return loss_ssim + (loss_mse * weight)