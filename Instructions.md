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

### Example dataset

Dataset example celeba [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)

| Change | Change | Change |
| ------ | ------ | ------ |
| -1     | 2500   | 2500   |
| 1      | 2500   | 2500   |

### Dataset division

The rule of thumb followed throughout the division of both the datasets consists in assigning 80\% of the images to the training and validation sets. The remaining part is reserved to the test set. This rule is usually related to the Pareto principle: 20\% of causes produce 80\% of effects.

Control

## Models

Example of a model

**CNN model for Task A1**

| Layer (type)       | Output shape    | Parameters |
| ------------------ | --------------- | ---------- |
| Convolutional_2D   | ( , 96, 96, 16) | 448        |
| Convolutional_2D   | ( , 96, 96, 16) | 2320       |
| MaxPooling_2D      | ( , 48, 48, 16) | 0          |
| Convolutional_2D   | ( , 48, 48, 32) | 4640       |
| Convolutional_2D   | ( , 48, 48, 32) | 9248       |
| MaxPooling_2D      | ( , 24, 24, 32) | 0          |
| Convolutional_2D   | ( , 24, 24, 64) | 18496      |
| Convolutional_2D   | ( , 24, 24, 64) | 36928      |
| BatchNormalization | ( , 24, 24, 64) | 256 (128)  |
| MaxPooling_2D      | ( , 12, 12, 64) | 0          |
| Flatten            | ( , 9216)       | 0          |
| Dropout            | ( , 9216)       | 0          |
| Dense              | ( , 2)          | 18434      |
| Total params       |                 | 90770      |

## Main execution

Before the code execution, the Datasets folder must have the following structure.

![image](https://user-images.githubusercontent.com/48513387/100546886-065feb80-3264-11eb-97a5-fc698833878b.png)

## Issues

- If the device's GPU memory is not enough to run the code, it is possible to execute the training of each model inside a dedicated subprocess. Tensorflow then will release the part of GPU memory used by each subprocess as soon as it ends.
