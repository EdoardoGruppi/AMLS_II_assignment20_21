# Introduction

The following is a short summary extracted from the papers cited below.

Image super resolution's aim is to rebuild a high-resolution image from its lower resolution counterpart. This technique is often adopted to facilitate other computer vision tasks. Since several high resolution images can be constructed starting from a single low resolution image this is considered an ill posed problem. In the recent years the image super resolution technique has been implemented through various deep learning methods, such as Convolutional Neural Network (CNNs) and Generative Adversarial Networks (GANs), leading to noteworthy results.

## Frameworks

### 1. Pre-upsampling

At the beginning, the low resolution images are up-sampled obtaining coarse high resolution images. Then the details are reconstructed through deep neural networks such as CNNs.

- Pros: reduction of the learning difficulty.
- Cons: the predefined up-sampling can introduce side effects and the high dimensionality requires more resources.

### 2. Post-upsampling

The upsampling is realised at the end of the network. Most of the computation is therefore performed in a low dimensional space reducing the amount of resources required.

- Pros: reduction of the model's complexity.
- Cons: the upsampling performed in one step increases the learning difficulty for large scaling factors.

### 3. Progressive upsampling

In this type of framework, the low resolution images are upsampled in multiple steps and, each time their resolution is increased, the pictures are refined by a series of convolutional layers.

- Pros: reduces the learning difficulty upsampling the images in more than one step.
- Cons: model design complexity and the training stability.

### 4. Iterative up-and-down sampling

In this framework the images are iteratively upsampled and downsampled allowing the model to better capture the deep relationships between the pairs of LR and HR images.

- Pros: better reconstruction results.
- Cons: model's design not always clear.

## Custom Layers

Since keras does not provide all the possible layers that can be used to perform the tasks assigned, a number of custom layers are implemented in the code of the project.

### 1. BicubicUpSampling2D

In Keras, the upsampling of the images can be done directly using two distinct pre-defined layers. The UpSampling2D layer allows to increase the image resolution through a nearest or a bilinear interpolation. On the other hand, the Conv2DTranspose layer up-samples the images through a transposed convolution that is very well represented in the image below extracted from the paper [1]. The BicubicUpSampling2D custom layer, created purposefully for this project, aims to increase the resolution of the input image through a bicubic interpolation.

![2021-02-09-13-01-47](https://user-images.githubusercontent.com/48513387/115118312-e32fa180-9fa2-11eb-8d51-4bd384e93ffc.png)

### 2. SubPixelConv2D

The SubPixelConv2D custom layer is an implementation of the layer proposed by Shi et al. [3]. Again, a very comprehensive illustration is provided by Wang et al. [1].

![2021-02-09-13-09-06](https://user-images.githubusercontent.com/48513387/115118323-ecb90980-9fa2-11eb-8cae-92d16b5be0cc.png)

### 3. ResidualBlock

Residual block created according to the information retrieved from the paper [2]. Differently from the residual blocks presented in the papers [4, 5], in this case there is no BatchNormalization layer involved as well as no activation functions after the skip connection. The picture below, from the paper [2], represents the three different versions of the residual block.

![2021-02-09-13-15-22](https://user-images.githubusercontent.com/48513387/115118325-ef1b6380-9fa2-11eb-8849-5c45a2d71523.png)

### 4. SRResNetBlock

Residual block created according to the information retrieved from the paper [5]. It corresponds to the graph b) of the picture displayed above.

### 5. DifferenceRGB

This custom preprocessing layer subtracts the mean RGB value computed on a dataset to the images passed as inputs.

## Loss Functions

The objective of loss functions is to measure the difference between the SR and HR images in order to drive the optimization of the model. Two or more loss functions can be weigthed and summed obtaining a new custom loss that focuses on more aspects.

### 1. Mean Absolute Error (MAE)

The MAE, alternatively known as L1 loss, belongs to the class of pixel-wise loss functions. It corresponds to the average of the absolute distances between the pixels of the target image (HR) and the predicted one (SR).

![2021-02-09-15-45-28](https://user-images.githubusercontent.com/48513387/115118329-f478ae00-9fa2-11eb-86c8-d54ebd0ebe3a.png)

### 2. Mean Square Error (MSE)

The MSE, also known as L2 loss or Quadratic loss, is a pixel-wise loss functions. It corresponds to the average of the squared distances between the pixels of the target image (HR) and the predicted one (SR). The L2 loss penalizes more the larger errors while it is more tolerant to small errors.

![2021-02-09-15-44-42](https://user-images.githubusercontent.com/48513387/115118332-f7739e80-9fa2-11eb-9af6-f3ada2388fb6.png)

### 3. Content Loss

The Content Loss, also called Perceptual loss and sometimes Vgg loss, evaluates the perceptual quality of images measuring the difference between the high level representations extracted after the n-th layer of a specific pre trained model, such as the VGG19.

### 4. Adversarial Loss

This loss is used to train GANs where two distinct networks, namely the generator and the discriminator, dueling each other. In general, the models trained with adversarial loss showcase a greater perceptual quality compared to those trained on pixel loss. Nevertheless, the PSNR achieved may be lower than the one gained with models trained thorugh a pixel-wise loss.

## Metrics

### 1. Peak Signal Noise Ratio (PSNR)

As it can be seen by the formula represented below the PSNR takes into consideration only the difference between corresponding pixels. It is therefore not the best metric to understand the quality of the reconstruction, resulting often into poor performance. Nevertheless, the PSNR is adopted by the vast majority of the literature becoming thus an important element of comparison.

![2021-02-09-11-48-32](https://user-images.githubusercontent.com/48513387/115118334-f9d5f880-9fa2-11eb-8834-3244109f9457.png)

where L defines the maximum pixel value, and the denominator of the logaritmic argument represents the mean square error (MSE) computed between the corresponding pixels of the two images (SR and HR).

### 2. Structural SIMilarity (SSIM)

The structural similarity index evaluates the quality of the reconstruction considering the structural similarity between pictures resulting into a better perceptual assessment.

![2021-02-09-12-09-08](https://user-images.githubusercontent.com/48513387/115118335-fcd0e900-9fa2-11eb-94e7-68128f333e9a.png)

The elements within the formula correspond respectively to the comparisons computed on the luminance, the contrast and the structure of the two pictures (SR and HR). As well as the MSE, the SSIM can be adopted also as a loss function.

# CNN experiments

All the values reported are obtained starting from a train dataset of 770, a validation dataset of 200 images and a test dataset of 30 images. **Note: the table below does not contain all the experiments conducted. It also does not include important information on the training as well as on the loss values, although they are kept into account while defining the network structure.**

| Model | Crop | Batch | Epoch | Loss |    Train Acc.    |   Valid. Acc.    |    Test Acc.     |     Bicubic      | Parameters |
| :---: | :--: | :---: | :---: | :--: | :--------------: | :--------------: | :--------------: | :--------------: | :--------: |
|  001  |  64  |  10   |  25   |  L1  | 27.1700 / 0.7480 | 27.7627 / 0.7500 | 26.4882 / 0.7046 | 26.9229 / 0.6997 |   25,315   |
|  002  |  64  |  10   |  30   |  L1  | 26.3345 / 0.7346 | 26.7128 / 0.7540 | 27.4635 / 0.7283 | 29.3971 / 0.7341 |   47,107   |
|  003  |  64  |  10   |  30   |  L1  | 27.1294 / 0.7504 | 26.7838 / 0.7548 |  26.0547 0.7038  | 27.8766 / 0.6966 |  135,299   |
|  004  |  64  |  10   |  40   |  L1  | 27.7628 / 0.7561 | 28.0063 / 0.7553 | 28.3941 / 0.7769 | 30.5163 / 0.7829 |   66,099   |
|  005  |  64  |  10   |  40   |  L1  | 27.5981 / 0.7605 | 28.1771 / 0.7589 | 27.8299 / 0.7247 | 29.3485 / 0.7176 |   66,099   |
|  006  |  64  |  10   |  40   |  L1  | 28.4397 / 0.7635 | 27.9805 / 0.7576 | 27.1800 / 0.7224 | 27.6928 / 0.7168 |   66,099   |
|  007  |  64  |  10   |  50   |  L1  | 27.5742 / 0.7523 | 27.7993 / 0.7610 | 26.5054 / 0.6996 | 26.9981 / 0.6914 |   35,483   |
|  008  |  64  |  10   |  50   |  L1  | 28.4945 / 0.7660 | 28.5997 / 0.7744 | 29.3338 / 0.7794 | 29.7809 / 0.7710 |  140,083   |
|  009  |  64  |  10   |  50   |  L1  | 28.5527 / 0.7692 | 28.1731 / 0.7532 | 26.8589 / 0.7156 | 26.3849 / 0.6957 |  177,075   |
|  010  |  64  |  10   |  50   |  L1  | 27.9535 / 0.7619 | 28.3993 / 0.7678 | 29.1760 / 0.7604 | 28.9419 / 0.7467 |  426,531   |
|  011  |  64  |  10   |  50   |  L1  | 28.4682 / 0.7691 | 28.2119 / 0.7772 | 27.7840 / 0.7222 | 28.0824 / 0.7069 |  223,715   |
|  012  |  64  |  10   |  50   |  L1  | 28.3123 / 0.7725 | 28.1187 / 0.7715 | 29.7127 / 0.7731 | 30.3862 / 0.7491 |  426,531   |
|  012  |  64  |  10   |  60   |  L2  | 28.6738 / 0.7736 | 28.7952 / 0.7776 | 28.9561 / 0.7756 | 28.3306 / 0.7450 |  426,531   |
|  012  |  64  |  10   |  60   |  L1  | 28.0835 / 0.7674 | 27.5101 / 0.7399 | 30.9927 / 0.8185 | 33.5879 / 0.8075 |  426,531   |
|  013  |  64  |  10   |  60   |  L1  | 28.3950 / 0.7740 | 28.6822 / 0.7737 | 27.3872 / 0.7231 | 26.8761 / 0.6965 |  426,531   |
|  014  |  64  |  10   |  60   |  L1  | 27.8945 / 0.7672 | 28.6099 / 0.7875 | 29.0427 / 0.7763 | 29.9721 / 0.7605 |  426,531   |
|  015  |  64  |  10   |  60   |  L1  | 27.8944 / 0.7652 | 28.7895 / 0.7682 | 27.1903 / 0.7615 | 26.4787 / 0.7289 |  574,243   |
|  016  |  64  |  10   |  60   |  L1  | 28.8518 / 0.7730 | 28.4657 / 0.7720 | 27.3735 / 0.7513 | 26.8045 / 0.7269 |  251,059   |
|  016  |  48  |  10   |  60   |  L1  | 28.8981 / 0.7719 | 28.2699 / 0.7634 | 27.8732 / 0.7225 | 28.5648 / 0.6962 |  251,059   |
|  016  |  56  |  10   |  60   |  L1  | 28.5662 / 0.7611 | 28.0348 / 0.7446 | 27.4440 / 0.7411 | 27.2333 / 0.7239 |  251,059   |
|  017  |  64  |  10   |  60   |  L1  | 28.5288 / 0.7638 | 28.7303 / 0.7716 | 27.3641 / 0.7307 | 27.0024 / 0.7075 |  297,267   |
|  017  |  56  |  10   |  60   |  L1  | 29.0887 / 0.7763 | 29.0340 / 0.7666 | 29.1884 / 0.7563 | 30.3011 / 0.7457 |  297,267   |
|  018  |  64  |  10   |  60   |  L1  | 28.4208 / 0.7718 | 28.6069 / 0.7685 | 27.4901 / 0.7326 | 26.8332 / 0.7045 |  426,531   |
|  019  |  64  |  10   |  60   |  L1  | 28.0669 / 0.7653 | 27.7093 / 0.7450 | 26.2622 / 0.7163 | 25.8070 / 0.6860 |  251,059   |
|  020  |  56  |  10   |  60   |  L2  | 27.8986 / 0.7435 | 27.5504 / 0.7275 | 26.7945 / 0.6931 | 27.7352 / 0.6928 |  186,723   |

## Architectures

001

In the following model investigated the upsampling of the images is performed at the beginning through a bicubic interpolation thanks to the BicubicUpSampling2D layer created. Then the coarse HR representations are elaborated by a series of convolutional layers whose job is to refine the details. As expressed in the section below, the last layer is needed in order to have outputs within the range [0,1]. Although the results are promising there are some drawbacks, in terms of resources utilised, due to the upsampling positioned in the first layer of the model. Moreover, the number of epochs can be increased since after 25 epochs there is still room for further improvement. To reduce the negative aspects highlighted, the upsampling can be postponed at the end of the model. Nevertheless, under the new configuration the BicubicUpSampling2D slows down the execution time of each epoch without improving the model performance.

```python
inputs = Input(shape=input_shape)
x = BicubicUpSampling2D(scale=4, image_size=patch_size)(inputs)
x = Conv2D(filters=16, kernel_size=(3, 3), activation='relu', padding='same')(x)
x = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(x)
x = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(x)
# The sigmoid activation function guarantees that the final output are within the range [0,1]
outputs = Conv2D(filters=3, kernel_size=(3, 3), activation='sigmoid', padding='same')(x)
```

002

In the experiments conducted, the posinioning of the BicubicUpSampling2D layer at the end of the model lowered the performance achieved while requiring more time for the execution. Although the time spent is then lowered substituing the BicubicUpSampling2D with the UpSampling2D layers provided by keras, the results are not satisfactory. Consequently, the upsampling is performed by using two consecutive Conv2DTranspose.

```python
inputs = Input(shape=input_shape)
x = Conv2D(filters=16, kernel_size=(3, 3), activation='relu', padding='same')(inputs)
x = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(x)
x = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(x)
x = Conv2DTranspose(filters=32, kernel_size=3, strides=2, activation='relu', padding='same')(x)
x = Conv2DTranspose(filters=16, kernel_size=3, strides=2, activation='relu', padding='same')(x)
# The sigmoid activation function guarantees that the final output are within the range [0,1]
outputs = Conv2D(filters=3, kernel_size=(3, 3), activation='sigmoid', padding='same')(x)
```

003

```python
inputs = Input(shape=input_shape)
x = Conv2D(filters=16, kernel_size=(3, 3), activation='relu', padding='same')(inputs)
x = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(x)
x = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(x)
x = SubPixelConv2D(channels=32, scale=2, kernel_size=(3, 3), activation='relu', padding='same')(x)
x = SubPixelConv2D(channels=32, scale=2, kernel_size=(3, 3), activation='relu', padding='same')(x)
# The sigmoid activation function guarantees that the final output are within the range [0,1]
outputs = Conv2D(filters=3, kernel_size=(3, 3), activation='sigmoid', padding='same')(x)
```

004

```python
inputs = Input(shape=input_shape)
x = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(inputs)
x = ResidualBlock(filters=32, kernel_size=(3, 3), scaling=None, activation='relu', padding='same')(x)
x = ResidualBlock(filters=32, kernel_size=(3, 3), scaling=None, activation='relu', padding='same')(x)
x = SubPixelConv2D(channels=16, scale=2, kernel_size=(3, 3), activation='relu', padding='same')(x)
x = SubPixelConv2D(channels=16, scale=2, kernel_size=(3, 3), activation='relu', padding='same')(x)
# The sigmoid activation function guarantees that the final output are within the range [0,1]
outputs = Conv2D(filters=3, kernel_size=(3, 3), activation='sigmoid', padding='same')(x)
```

005

```python
inputs = Input(shape=input_shape)
x = DifferenceRGB(RGB_MEAN_A)(inputs)
x1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(x)
x = ResidualBlock(filters=32, kernel_size=(3, 3), scaling=None, activation='relu', padding='same')(x1)
x = ResidualBlock(filters=32, kernel_size=(3, 3), scaling=None, activation='relu', padding='same')(x)
x = Add()([x1, x])
x = SubPixelConv2D(channels=16, scale=2, kernel_size=(3, 3), activation='relu', padding='same')(x)
x = SubPixelConv2D(channels=16, scale=2, kernel_size=(3, 3), activation='relu', padding='same')(x)
# The sigmoid activation function guarantees that the final output are within the range [0,1]
outputs = Conv2D(filters=3, kernel_size=(3, 3), activation='sigmoid', padding='same')(x)
```

006

```python
inputs = Input(shape=input_shape)
x = DifferenceRGB(RGB_MEAN_A)(inputs)
x1 = Conv2D(filters=16, kernel_size=(3, 3), activation='relu', padding='same')(x)
x = ResidualBlock(filters=16, kernel_size=(3, 3), scaling=None, activation='relu', padding='same')(x1)
x = ResidualBlock(filters=16, kernel_size=(3, 3), scaling=None, activation='relu', padding='same')(x)
x = ResidualBlock(filters=16, kernel_size=(3, 3), scaling=None, activation='relu', padding='same')(x)
x = ResidualBlock(filters=16, kernel_size=(3, 3), scaling=None, activation='relu', padding='same')(x)
x = ResidualBlock(filters=16, kernel_size=(3, 3), scaling=None, activation='relu', padding='same')(x)
x = ResidualBlock(filters=16, kernel_size=(3, 3), scaling=None, activation='relu', padding='same')(x)
x = Add()([x1, x])
x = SubPixelConv2D(channels=8, scale=2, kernel_size=(3, 3), activation='relu', padding='same')(x)
x = SubPixelConv2D(channels=8, scale=2, kernel_size=(3, 3), activation='relu', padding='same')(x)
# The sigmoid activation function guarantees that the final output are within the range [0,1]
outputs = Conv2D(filters=3, kernel_size=(3, 3), activation='sigmoid', padding='same')(x)
```

007

```python
inputs = Input(shape=input_shape)
x = DifferenceRGB(RGB_MEAN_A)(inputs)
x1 = Conv2D(filters=16, kernel_size=(3, 3), activation='relu', padding='same')(x)
x = ResidualBlock(filters=16, kernel_size=(3, 3), scaling=None, activation='relu', padding='same')(x1)
x = ResidualBlock(filters=16, kernel_size=(3, 3), scaling=None, activation='relu', padding='same')(x)
x = ResidualBlock(filters=16, kernel_size=(3, 3), scaling=None, activation='relu', padding='same')(x)
x = ResidualBlock(filters=16, kernel_size=(3, 3), scaling=None, activation='relu', padding='same')(x)
x = ResidualBlock(filters=16, kernel_size=(3, 3), scaling=None, activation='relu', padding='same')(x)
x = ResidualBlock(filters=16, kernel_size=(3, 3), scaling=None, activation='relu', padding='same')(x)
x = Add()([x1, x])
x = SubPixelConv2D(channels=8, scale=2, kernel_size=(3, 3), activation='relu', padding='same')(x)
x = SubPixelConv2D(channels=8, scale=2, kernel_size=(3, 3), activation='relu', padding='same')(x)
# The sigmoid activation function guarantees that the final output are within the range [0,1]
outputs = Conv2D(filters=3, kernel_size=(3, 3), activation='sigmoid', padding='same')(x)
```

008

```python
inputs = Input(shape=input_shape)
x1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(x)
x = ResidualBlock(filters=32, kernel_size=(3, 3), scaling=None, activation='relu', padding='same')(x1)
x = ResidualBlock(filters=32, kernel_size=(3, 3), scaling=None, activation='relu', padding='same')(x)
x = ResidualBlock(filters=32, kernel_size=(3, 3), scaling=None, activation='relu', padding='same')(x)
x = ResidualBlock(filters=32, kernel_size=(3, 3), scaling=None, activation='relu', padding='same')(x)
x = ResidualBlock(filters=32, kernel_size=(3, 3), scaling=None, activation='relu', padding='same')(x)
x = ResidualBlock(filters=32, kernel_size=(3, 3), scaling=None, activation='relu', padding='same')(x)
x = Add()([x1, x])
x = SubPixelConv2D(channels=16, scale=2, kernel_size=(3, 3), activation='relu', padding='same')(x)
x = SubPixelConv2D(channels=16, scale=2, kernel_size=(3, 3), activation='relu', padding='same')(x)
# The sigmoid activation function guarantees that the final output are within the range [0,1]
outputs = Conv2D(filters=3, kernel_size=(3, 3), activation='sigmoid', padding='same')(x)
```

009

```python
inputs = Input(shape=input_shape)
x = DifferenceRGB(RGB_MEAN_A)(inputs)
x1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(x)
x = ResidualBlock(filters=32, kernel_size=(3, 3), scaling=None, activation='relu', padding='same')(x1)
x = ResidualBlock(filters=32, kernel_size=(3, 3), scaling=None, activation='relu', padding='same')(x)
x = ResidualBlock(filters=32, kernel_size=(3, 3), scaling=None, activation='relu', padding='same')(x)
x = ResidualBlock(filters=32, kernel_size=(3, 3), scaling=None, activation='relu', padding='same')(x)
x = ResidualBlock(filters=32, kernel_size=(3, 3), scaling=None, activation='relu', padding='same')(x)
x = ResidualBlock(filters=32, kernel_size=(3, 3), scaling=None, activation='relu', padding='same')(x)
x = ResidualBlock(filters=32, kernel_size=(3, 3), scaling=None, activation='relu', padding='same')(x)
x = ResidualBlock(filters=32, kernel_size=(3, 3), scaling=None, activation='relu', padding='same')(x)
x = Add()([x, x1])
x = SubPixelConv2D(channels=16, scale=2, kernel_size=(3, 3), activation='relu', padding='same')(x)
x = SubPixelConv2D(channels=16, scale=2, kernel_size=(3, 3), activation='relu', padding='same')(x)
# The sigmoid activation function guarantees that the final output are within the range [0,1]
outputs = Conv2D(filters=3, kernel_size=(3, 3), activation='sigmoid', padding='same')(x)
```

010

```python
inputs = Input(shape=input_shape)
x = DifferenceRGB(RGB_MEAN_A)(inputs)
x = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(x)
x1 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(x)
x = ResidualBlock(filters=64, kernel_size=(3, 3), scaling=None, activation='relu', padding='same')(x1)
x = ResidualBlock(filters=64, kernel_size=(3, 3), scaling=None, activation='relu', padding='same')(x)
x = ResidualBlock(filters=64, kernel_size=(3, 3), scaling=None, activation='relu', padding='same')(x)
x = ResidualBlock(filters=64, kernel_size=(3, 3), scaling=None, activation='relu', padding='same')(x)
x = Add()([x, x1])
x = SubPixelConv2D(channels=32, scale=2, kernel_size=(3, 3), activation='relu', padding='same')(x)
x = SubPixelConv2D(channels=32, scale=2, kernel_size=(3, 3), activation='relu', padding='same')(x)
# The sigmoid activation function guarantees that the final output are within the range [0,1]
outputs = Conv2D(filters=3, kernel_size=(3, 3), activation='sigmoid', padding='same')(x)
```

011

```python
inputs = Input(shape=input_shape)
x = DifferenceRGB(RGB_MEAN_A)(inputs)
x1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(x)
x = ResidualBlock(filters=32, kernel_size=(3, 3), scaling=None, activation='relu', padding='same')(x1)
x = ResidualBlock(filters=32, kernel_size=(3, 3), scaling=None, activation='relu', padding='same')(x)
x = ResidualBlock(filters=32, kernel_size=(3, 3), scaling=None, activation='relu', padding='same')(x)
x = ResidualBlock(filters=32, kernel_size=(3, 3), scaling=None, activation='relu', padding='same')(x)
x = ResidualBlock(filters=32, kernel_size=(3, 3), scaling=None, activation='relu', padding='same')(x)
x = ResidualBlock(filters=32, kernel_size=(3, 3), scaling=None, activation='relu', padding='same')(x)
x = ResidualBlock(filters=32, kernel_size=(3, 3), scaling=None, activation='relu', padding='same')(x)
x = ResidualBlock(filters=32, kernel_size=(3, 3), scaling=None, activation='relu', padding='same')(x)
x = Add()([x, x1])
x = SubPixelConv2D(channels=32, scale=2, kernel_size=(3, 3), activation='relu', padding='same')(x)
x = SubPixelConv2D(channels=32, scale=2, kernel_size=(3, 3), activation='relu', padding='same')(x)
# The sigmoid activation function guarantees that the final output are within the range [0,1]
outputs = Conv2D(filters=3, kernel_size=(3, 3), activation='sigmoid', padding='same')(x)
```

012

```python
inputs = Input(shape=input_shape)
x = DifferenceRGB(RGB_MEAN_A)(inputs)
x = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(x)
x1 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(x)
x = ResidualBlock(filters=64, kernel_size=(3, 3), scaling=None, activation='relu', padding='same')(x1)
x = ResidualBlock(filters=64, kernel_size=(3, 3), scaling=None, activation='relu', padding='same')(x)
x = ResidualBlock(filters=64, kernel_size=(3, 3), scaling=None, activation='relu', padding='same')(x)
x = ResidualBlock(filters=64, kernel_size=(3, 3), scaling=None, activation='relu', padding='same')(x)
x = Add()([x, x1])
x = SubPixelConv2D(channels=32, scale=2, kernel_size=(3, 3), activation='relu', padding='same')(x)
x = SubPixelConv2D(channels=32, scale=2, kernel_size=(3, 3), activation='relu', padding='same')(x)
# The sigmoid activation function guarantees that the final output are within the range [0,1]
outputs = Conv2D(filters=3, kernel_size=(3, 3), activation='sigmoid', padding='same')(x)
```

013

```python
inputs = Input(shape=input_shape)
x = DifferenceRGB(RGB_MEAN_A)(inputs)
x = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(x)
x1 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(x)
x = ResidualBlock(filters=64, kernel_size=(3, 3), scaling=None, activation='relu', padding='same')(x1)
x = ResidualBlock(filters=64, kernel_size=(3, 3), scaling=None, activation='relu', padding='same')(x)
x = ResidualBlock(filters=64, kernel_size=(3, 3), scaling=None, activation='relu', padding='same')(x)
x = Add()([x, x1])
x = ResidualBlock(filters=64, kernel_size=(3, 3), scaling=None, activation='relu', padding='same')(x)
x = Add()([x, x1])
x = SubPixelConv2D(channels=32, scale=2, kernel_size=(3, 3), activation='relu', padding='same')(x)
x = SubPixelConv2D(channels=32, scale=2, kernel_size=(3, 3), activation='relu', padding='same')(x)
# The sigmoid activation function guarantees that the final output are within the range [0,1]
outputs = Conv2D(filters=3, kernel_size=(3, 3), activation='sigmoid', padding='same')(x)
```

014

```python
inputs = Input(shape=input_shape)
x = DifferenceRGB(RGB_MEAN_A)(inputs)
x = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(x)
x1 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(x)
x = ResidualBlock(filters=64, kernel_size=(3, 3), scaling=None, activation='relu', padding='same')(x1)
x = ResidualBlock(filters=64, kernel_size=(3, 3), scaling=None, activation='relu', padding='same')(x)
x = Add()([x, x1])
x = ResidualBlock(filters=64, kernel_size=(3, 3), scaling=None, activation='relu', padding='same')(x)
x = Add()([x, x1])
x = ResidualBlock(filters=64, kernel_size=(3, 3), scaling=None, activation='relu', padding='same')(x)
x = Add()([x, x1])
x = SubPixelConv2D(channels=32, scale=2, kernel_size=(3, 3), activation='relu', padding='same')(x)
x = SubPixelConv2D(channels=32, scale=2, kernel_size=(3, 3), activation='relu', padding='same')(x)
# The sigmoid activation function guarantees that the final output are within the range [0,1]
outputs = Conv2D(filters=3, kernel_size=(3, 3), activation='sigmoid', padding='same')(x)

```

015

```python
inputs = Input(shape=input_shape)
x = DifferenceRGB(RGB_MEAN_A)(inputs)
x = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(x)
x1 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(x)
x = ResidualBlock(filters=64, kernel_size=(3, 3), scaling=None, activation='relu', padding='same')(x1)
x = ResidualBlock(filters=64, kernel_size=(3, 3), scaling=None, activation='relu', padding='same')(x)
x = ResidualBlock(filters=64, kernel_size=(3, 3), scaling=None, activation='relu', padding='same')(x)
x = ResidualBlock(filters=64, kernel_size=(3, 3), scaling=None, activation='relu', padding='same')(x)
x = ResidualBlock(filters=64, kernel_size=(3, 3), scaling=None, activation='relu', padding='same')(x)
x = ResidualBlock(filters=64, kernel_size=(3, 3), scaling=None, activation='relu', padding='same')(x)
x = Add()([x, x1])
x = SubPixelConv2D(channels=32, scale=2, kernel_size=(3, 3), activation='relu', padding='same')(x)
x = SubPixelConv2D(channels=32, scale=2, kernel_size=(3, 3), activation='relu', padding='same')(x)
# The sigmoid activation function guarantees that the final output are within the range [0,1]
outputs = Conv2D(filters=3, kernel_size=(3, 3), activation='sigmoid', padding='same')(x)
```

016

```python
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
```

017

```python
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
x = SubPixelConv2D(channels=16, scale=4, kernel_size=(3, 3), activation='relu', padding='same')(x)
# The sigmoid activation function guarantees that the final output are within the range [0,1]
outputs = Conv2D(filters=3, kernel_size=(3, 3), activation='sigmoid', padding='same')(x)
```

018

```python
inputs = Input(shape=input_shape)
x = DifferenceRGB(RGB_MEAN_A)(inputs)
x = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(x)
x1 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(x)
x = ResidualBlock(filters=64, kernel_size=(3, 3), scaling=None, activation='relu', padding='same')(x1)
x = ResidualBlock(filters=64, kernel_size=(3, 3), scaling=None, activation='relu', padding='same')(x)
x = Add()([x, x1])
x = ResidualBlock(filters=64, kernel_size=(3, 3), scaling=None, activation='relu', padding='same')(x)
x = Add()([x, x1])
x = ResidualBlock(filters=64, kernel_size=(3, 3), scaling=None, activation='relu', padding='same')(x)
x = Add()([x, x1])
x = SubPixelConv2D(channels=32, scale=2, kernel_size=(3, 3), activation='relu', padding='same')(x)
x = SubPixelConv2D(channels=32, scale=2, kernel_size=(3, 3), activation='relu', padding='same')(x)
# The sigmoid activation function guarantees that the final output are within the range [0,1]
outputs = Conv2D(filters=3, kernel_size=(3, 3), activation='sigmoid', padding='same')(x)
```

019

```python
inputs = Input(shape=input_shape)
x = DifferenceRGB(RGB_MEAN_A)(inputs)
x1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(x)
x = ResidualBlock(filters=32, kernel_size=(3, 3), scaling=None, activation='relu', padding='same')(x1)
x = ResidualBlock(filters=32, kernel_size=(3, 3), scaling=None, activation='relu', padding='same')(x)
x = Add()([x, x1])
x = ResidualBlock(filters=32, kernel_size=(3, 3), scaling=None, activation='relu', padding='same')(x)
x = Add()([x, x1])
x = ResidualBlock(filters=32, kernel_size=(3, 3), scaling=None, activation='relu', padding='same')(x)
x = Add()([x, x1])
x = ResidualBlock(filters=32, kernel_size=(3, 3), scaling=None, activation='relu', padding='same')(x)
x = Add()([x, x1])
x = ResidualBlock(filters=32, kernel_size=(3, 3), scaling=None, activation='relu', padding='same')(x)
x = Add()([x, x1])
x = ResidualBlock(filters=32, kernel_size=(3, 3), scaling=None, activation='relu', padding='same')(x)
x = Add()([x, x1])
x = ResidualBlock(filters=32, kernel_size=(3, 3), scaling=None, activation='relu', padding='same')(x)
x = Add()([x, x1])
x = ResidualBlock(filters=32, kernel_size=(3, 3), scaling=None, activation='relu', padding='same')(x)
x = Add()([x, x1])
x = ResidualBlock(filters=32, kernel_size=(3, 3), scaling=None, activation='relu', padding='same')(x)
x = Add()([x, x1])
x = ResidualBlock(filters=32, kernel_size=(3, 3), scaling=None, activation='relu', padding='same')(x)
x = Add()([x, x1])
x = ResidualBlock(filters=32, kernel_size=(3, 3), scaling=None, activation='relu', padding='same')(x)
x = Add()([x, x1])
x = SubPixelConv2D(channels=16, scale=2, kernel_size=(3, 3), activation='relu', padding='same')(x)
x = SubPixelConv2D(channels=16, scale=2, kernel_size=(3, 3), activation='relu', padding='same')(x)
# The sigmoid activation function guarantees that the final output are within the range [0,1]
outputs = Conv2D(filters=3, kernel_size=(3, 3), activation='sigmoid', padding='same')(x)
```

020

```python
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
x = SubPixelConv2D(channels=32, scale=2, kernel_size=(3, 3), activation='relu', padding='same')(x)
# The sigmoid activation function guarantees that the final output are within the range [0,1]
outputs = Conv2D(filters=3, kernel_size=(3, 3), activation='sigmoid', padding='same')(x)
```

## References

- Wang, Zhihao, Jian Chen, and Steven CH Hoi. "Deep learning for image super-resolution: A survey." IEEE transactions on pattern analysis and machine intelligence (2020).
- Lim, Bee, et al. "Enhanced deep residual networks for single image super-resolution." Proceedings of the IEEE conference on computer vision and pattern recognition workshops. 2017.
- Shi, Wenzhe, et al. "Real-time single image and video super-resolution using an efficient sub-pixel convolutional neural network." Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.
- He, Kaiming, et al. "Deep residual learning for image recognition." Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.
- Ledig, Christian, et al. "Photo-realistic single image super-resolution using a generative adversarial network." Proceedings of the IEEE conference on computer vision and pattern recognition. 2017.
