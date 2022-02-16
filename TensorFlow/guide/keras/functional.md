# Functional API

2022-02-16, 10:13
***

## 配置

```python
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
```

## 简介

Keras 函数式（functional）API 是一种比 [tf.keras.Sequential](sequential_model.md) 更灵活的创建模型的方法。函数式 API 可以处理非线性拓扑结构、共享层，甚至可以处理包含多个输入或多个输出的模型。

其主要思想是，深度学习模型是由 layer 组成的有向无环图（directed acyclic graph, DAG），函数式 API 是构建这种图的一种方法。

考虑如下模型：

```py
(input: 784-dimensional vectors)
       ↧
[Dense (64 units, relu activation)]
       ↧
[Dense (64 units, relu activation)]
       ↧
[Dense (10 units, softmax activation)]
       ↧
(output: logits of a probability distribution over 10 classes)
```

该模型包含三层，使用函数 API 创建该模型，首先创建输入节点：

```python
inputs = keras.Input(shape=(784,))
```

输入数据是长度为 784 的向量。此处只指定样本 shape，忽略 batch size。

假如输入是 shape 为 `(32, 32, 3)` 的图片，此时输入定义为：

```python
img_inputs = keras.Input(shape=(32, 32, 3))
```

创建的 `inputs` 包含输入数据的 shape 和 `dtype`：

```python
>>> inputs.shape
TensorShape([None, 784])
>>> inputs.dtype
tf.float32
```

然后创建下一层，并将 `inputs` 对象作为输入：

```python
dense = layers.Dense(64, activation="relu")
x = dense(inputs)
```

层调用（layer call）就像从 "inputs" 到创建这个 dense 层画了个箭头。将 `inputs` 传入 `dense` 层，获得输出 `x`。

继续创建图中余下两层：

```python
x = layers.Dense(64, activation="relu")(x)
outputs = layers.Dense(10)(x)
```

此时可以将 `inputs` 和 `outputs` 作为参数创建 `Model`：

```python
model = keras.Model(inputs=inputs, outputs=outputs, name="mnist_model")
```

查看模型：

```python
>>> model.summary()

Model: "mnist_model"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 input_1 (InputLayer)        [(None, 784)]             0         
                                                                 
 dense (Dense)               (None, 64)                50240     
                                                                 
 dense_1 (Dense)             (None, 64)                4160      
                                                                 
 dense_2 (Dense)             (None, 10)                650       
                                                                 
=================================================================
Total params: 55,050
Trainable params: 55,050
Non-trainable params: 0
_________________________________________________________________
```

也可以将模型结构输出为图片：

```python
keras.utils.plot_model(model, "my_first_model.png")
```


## 参考

- https://www.tensorflow.org/guide/keras/functional
