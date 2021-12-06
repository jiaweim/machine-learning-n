# Losses

- [Losses](#losses)
  - [简介](#简介)
  - [compile 和 fit 中使用](#compile-和-fit-中使用)
  - [概率损失](#概率损失)
    - [BinaryCrossentropy class](#binarycrossentropy-class)
    - [sparse_categorical_crossentropy function](#sparse_categorical_crossentropy-function)
  - [参考](#参考)

2021-11-26, 10:24
***

## 简介

损失函数用于计算模型在训练期间应该最小化的量值。

Keras 所有损失函数都可以通过类或者函数指定。

## compile 和 fit 中使用

损失函数是编辑 Keras 模型所需的两个参数之一：

```py
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential()
model.add(layers.Dense(64, kernel_initializer='uniform', input_shape=(10,)))
model.add(layers.Activation('softmax'))

loss_fn = keras.losses.SparseCategoricalCrossentropy()
model.compile(loss=loss_fn, optimizer='adam')
```

所有内置的损失函数可以通过它们的字符串标识符的使用：

```py
# pass optimizer by name: default parameters will be used
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')
```

损失函数通常通过实例化类（如 `keras.losses.SparseCategoricalCrossentropy`）来创建，所有损失函数也提供了函数形式（如 `keras.losses.sparse_categorical_crossentropy`）。

使用类可以在实例化时传递参数，例如：

```py
loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
```

## 概率损失

### BinaryCrossentropy class

```py
tf.keras.losses.BinaryCrossentropy(
    from_logits=False,
    label_smoothing=0,
    axis=-1,
    reduction="auto",
    name="binary_crossentropy",
)
```

计算真实标签和预测标签之间的交叉熵损失。

应用于二元分类问题，输入格式要求：

- `y_true`（真实标签）： 0 或 1；
- `y_pred`（预测标签）：模型预测值，

### sparse_categorical_crossentropy function

```py
tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=False, reduction="auto", name="sparse_categorical_crossentropy"
)
```

计算标签和预测值之间的交叉熵损失。

当有 2 个或以上的分类标签时使用。标签以整数形式提供。如果标签以 `one-hot` 表示提供，则应该使用 

## 参考

- https://keras.io/api/losses/
