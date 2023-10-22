# Regularizer

- [Regularizer](#regularizer)
  - [简介](#简介)
  - [示例](#示例)
  - [可用惩罚](#可用惩罚)
  - [直接调用 regularizer](#直接调用-regularizer)
  - [](#)

## 简介

`Regularizer` 是正则化基类。

正则化器（Regularizer）可以在优化期间对 layer 的参数进行惩罚。这些惩罚将汇总到神经网络优化的损失函数中。

正则化惩罚按层执行。具体的 API 取决于 layer，不过大多数 layer （如 `Dense`, `Conv1D`, `Conv2D`, `Conv3D`）具有统一的 API。

这些 layer 公开了 3 个关键字参数：

- `kernel_regularizer`，对 layer weights 施加惩罚的正则化器
- `bias_regularizer`，对 layer bias 施加惩罚的正则化器
- `activity_regularizer`，对 layer 输出施加惩罚的正则化器

所有 layer (包括自定义 layer) `activity_regularizer` 为可设置属性，不管是否在构造函数参数中。

`activity_regularizer` 返回的值已经除过 batch 大小，因此权重正则化和输出正则化的相对权重不受 batch 大小影响。

在对输入调用 layer 后，可以使用 `layer.losses` 访问 layer 的正则化惩罚值。

## 示例

```python
layer = tf.keras.layers.Dense(
    5, input_dim=5,
    kernel_initializer='ones',
    kernel_regularizer=tf.keras.regularizers.L1(0.01),
    activity_regularizer=tf.keras.regularizers.L2(0.01))
tensor = tf.ones(shape=(5, 5)) * 2.0
out = layer(tensor)
```

```python
# 权重正则化项为 0.25
# 输出正则化项为 5 （除过 batch size）
tf.math.reduce_sum(layer.losses)
```

```txt
<tf.Tensor: shape=(), dtype=float32, numpy=5.25>
```

## 可用惩罚

```python
tf.keras.regularizers.L1(0.3)  # L1 Regularization Penalty
tf.keras.regularizers.L2(0.1)  # L2 Regularization Penalty
tf.keras.regularizers.L1L2(l1=0.01, l2=0.01)  # L1 + L2 penalties
```

## 直接调用 regularizer

直接在张量上调用 regularizer 来计算正则化损失，和调用单参数函数一样：

```python
regularizer = tf.keras.regularizers.L2(2.)
tensor = tf.ones(shape=(5, 5))
regularizer(tensor)
```

```txt
<tf.Tensor: shape=(), dtype=float32, numpy=50.0>
```

## 