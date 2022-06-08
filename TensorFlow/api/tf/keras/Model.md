# tf.keras.Model

2022-03-09, 21:59
***

## 简介

```python
tf.keras.Model(
    *args, **kwargs
)
```

`Model` 类表示模型，将多个 layer 组合在一起，并包含训练和推断功能。

|参数|说明|
|---|---|
|inputs|The input(s) of the model: a keras.Input object or list of keras.Input objects.|
|outputs|The output(s) of the model. See Functional API example below.|
|name|String, the name of the model.|

实例化 `Model` 的方法方法有两种：

1. 使用函数 API

从 `Input` 开始，通过 layer 调用来指定模型的传递，最后从输入和输出创建模型：

```python
import tensorflow as tf

inputs = tf.keras.Input(shape=(3,))
x = tf.keras.layers.Dense(4, activation=tf.nn.relu)(inputs)
outputs = tf.keras.layers.Dense(5, activation=tf.nn.softmax)(x)
model = tf.keras.Model(inputs=inputs, outputs=outputs)
```

> ⚡：输入只支持张量的 dict, list 或 tuple。目前不支持嵌套。



## 参考

- https://www.tensorflow.org/api_docs/python/tf/keras/Model
