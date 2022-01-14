# 加载和预处理图像

- [加载和预处理图像](#加载和预处理图像)
  - [简介](#简介)
  - [配置](#配置)
    - [下载 flowers 数据集](#下载-flowers-数据集)
  - [参考](#参考)

2022-01-14, 09:53
***

## 简介

下面介绍三种加载和预处理图像数据集的方式：

- 第一种，使用 Keras 预处理工具（如 `tf.keras.utils.image_dataset_from_directory`） 和 layers（如 `tf.keras.layers.Rescaling`）从目录读取图像；
- 第二种，使用 `tf.data` 从头开始编写输入管道；
- 第三种，从 TensorFlow Datasets 下载数据集。

## 配置

```python
import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
import tensorflow_datasets as tfds
```

```python
print(tf.__version__)
```

```sh
2.6.0
```

### 下载 flowers 数据集

下面使用一个包含几千字花照片的数据集。该数据集包含五个子目录，每个类别一个。

```sh
flowers_photos/
  daisy/
  dandelion/
  roses/
  sunflowers/
  tulips/
```

下载图片：

```python
import pathlib
dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
data_dir = tf.keras.utils.get_file(origin=dataset_url,
                                   fname='flower_photos',
                                   untar=True)
data_dir = pathlib.Path(data_dir)
```

下载完成后（218 MB），可以发现共有 3,670 张图片：

```python
image_count = len(list(data_dir.glob('*/*.jpg')))
print(image_count)
```

```sh
3670
```

每个目录都包含该类型花的图像，例如：

```python
roses = list(data_dir.glob('roses/*'))
PIL.Image.open(str(roses[0]))
```



## 参考

- https://www.tensorflow.org/tutorials/load_data/images
