# 时间序列预测

## 简介

下面介绍如何使用 TensorFlow 进行时间序列预测。构建不同风格的模型，包括 CNN 和 RNN。

包括两个部分：

- 预测单个时间步
  - 单个特征
  - 所有特征
- 预测多个时间步
  - **single shot**: 一次预测所有值
  - 自回归（autoregressive）：一次预测一个，并将输出反馈给模型

## 配置

```python
import os
import datetime

import IPython
import IPython.display
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf

mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False
```

## weather 数据集

本教程使用马普生物地球化学研究所记录的[天气时间序列数据集](https://www.bgc-jena.mpg.de/wetter/)。

该数据集包含 14 个不同的特征，如气温、大气压和湿度等。该数据从 2003 年开始，每 10 分钟收集一次。为了提高效率，我们只用 2009 年到 2016 年的数据。

```python
zip_path = tf.keras.utils.get_file(
    origin='https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip',
    fname='jena_climate_2009_2016.csv.zip',
    extract=True)
csv_path, _ = os.path.splitext(zip_path)
```

## 参考

- https://www.tensorflow.org/tutorials/structured_data/time_series
