# Instructions

## Setup

1. Install Tensorflow and all the other packages appointed in the [README.md](https://github.com/EdoardoGruppi/AMLS_II_assignment20_21/blob/main/README.md) file.

2. Download the project directory from [GitHub](https://github.com/EdoardoGruppi/AMLS_II_assignment20_21).
3. Tensorflow enables to work directly on GPU without requiring explicity additional code. The only hardware requirement is having a Nvidia GPU card with Cuda enabled. To see if Tensorflow has detected a GPU on your device run the following few lines (see main.py).

   ```python
   import tensorflow as tf
   print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
   ```

   If not, there are lots of guides on the web to install everything you need. For instance, you can take a look at
   [this](https://deeplizard.com/learn/video/IubEtS2JAiY).

4. Finally, it is crucial to run the code below since Tensorflow tends to allocate directly all the GPU memory even if is not entirely needed. With these lines instead, it will allocate gradually the memory required by the program (see main.py).

   ```python
   if len(physical_devices) is not 0:
      tf.config.experimental.set_memory_growth(physical_devices[0], True)
   ```

## Run the code

Once all the necessary packages have been installed you can run the code by typing this line on the terminal or by the specific command within the IDE.

**Note:** To follow step by step the main execution take a look at the dedicated Section below.

```
python main.py
```

## Datasets

### DIV2K

The [DIVerse 2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) dataset is a set of high resolution images firstly introduced for the NTIRE challenges. It comprises 800 training images and 100 validation images.

As the test dataset is not released, in the project exclusively 770 of the starting training images are used during the learning phase. The remaining part is dedicated to the testing phase, whereas the validation set has remained unchanged.

### Additional test datasets

The models designed are evaluated on further well-known benchmark datasets as well. To download the datasets please visit this [Github](https://github.com/jbhuang0604/SelfExSR) page or click the links below.

| Test Dataset | Number of images | Link                                                                                  |
| ------------ | ---------------- | ------------------------------------------------------------------------------------- |
| Set5         | 5                | [Set5_url](https://uofi.box.com/shared/static/kfahv87nfe8ax910l85dksyl2q212voc.zip)   |
| Set14        | 14               | [Set14_url](https://uofi.box.com/shared/static/igsnfieh4lz68l926l8xbklwsnnk8we9.zip)  |
| BSD100       | 100              | [BSD100_url](https://uofi.box.com/shared/static/qgctsplb8txrksm9to9x01zfa4m61ngq.zip) |

## Models

**CNN model for Task A**

Root Network

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
```

Final part for the x2, x3, x4 ratios

```python
x = SubPixelConv2D(channels=32, scale=ratio, kernel_size=(3, 3), activation='relu', padding='same')(x)
# The sigmoid activation function guarantees that the final output are within the range [0,1]
outputs = Conv2D(filters=3, kernel_size=(3, 3), activation='sigmoid', padding='same')(x)
```

**GAN model for Task B**

Root network of the generative model

```python
inputs = Input(shape=input_shape)
x = DifferenceRGB(RGB_MEAN_B)(inputs)
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
```

Final part of the generative model for the x2, x3, x4 ratios

```python
x = SubPixelConv2D(channels=32, scale=ratio, kernel_size=(3, 3), activation='relu', padding='same')(x)
# The sigmoid activation function guarantees that the final output are within the range [0,1]
outputs = Conv2D(filters=3, kernel_size=(3, 3), activation='sigmoid', padding='same')(x)
```

Root network of the discriminative model

```python
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
outputs = Dense(1, activation='sigmoid')(x)
```

Initial part added to the discriminative model for only the x3,x4 ratios

```python
inputs = Input(new_input_shape)
difference = current_input_shape / new_input_shape
x = BicubicUpSampling2D(scale=difference, image_size=discriminator_input_shape[0])(inputs)
```

## Main execution

Before the execution of the code, the Datasets folder must be left empty. Once the functions displayed below are called...

```python
# Download datasets
download_datasets()
download_test_datasets()
split_dataset(test_size=test_dim)
input_shape = [patch_size, patch_size, 3]
```

...the folder presents the following structure.

![image](https://user-images.githubusercontent.com/48513387/108914265-62e86200-762b-11eb-8f25-5e10b25d2582.png)

## Issues

- If the device's GPU memory is not enough to run the code, it is possible to execute the training of each model inside a dedicated subprocess. Tensorflow then will release the part of GPU memory used by each subprocess as soon as it ends.
- At the present (04/17/2021), all the functions implemented to download the datasets work fine. However, whenever some problems are encountered with the original links, please download the datasets from the following directory: [Datasets](https://drive.google.com/drive/u/1/folders/1ulU6A3lDZhzolb16oOe7YaplhmS1WqpD).

## Reference for the additional test datasets

@inproceedings{Huang-CVPR-2015,
title={Single Image Super-Resolution From Transformed Self-Exemplars},
Author = {Huang, Jia-Bin and Singh, Abhishek and Ahuja, Narendra},
booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
pages={5197--5206},
Year = {2015}
}
