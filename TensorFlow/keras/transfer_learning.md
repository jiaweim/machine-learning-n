# 迁移学习和微调

## 设置

```python
import numpy as np
import tensorflow as tf
from tensorflow import keras
```

## 简介

迁移学习（Transfer learning）在一个问题上学习特征，并利用学习的特征解决新的类似问题。例如，学会识别浣熊的模型的特征很可能有助于启动一个识别狸猫的模型。

迁移学习一般用在数据集太少而无法从头开始全面训练的任务。

在深度学习中，迁移学习的流程：

1. 


## 参考

- https://www.tensorflow.org/guide/keras/transfer_learning/
