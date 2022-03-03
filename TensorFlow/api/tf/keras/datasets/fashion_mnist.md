# Module: tf.keras.datasets.fashion_mnist

2022-03-03, 16:23
****

## 简介

该模块包含函数 `tf.keras.datasets.fashion_mnist.load_data` 用于载入 Fashion-MNIST 数据集。

```python
tf.keras.datasets.fashion_mnist.load_data()
```

该数据包含 60,000 张 10 种服装类别的 28x28 灰度图片，以及 10,000 张图片作为测试集。该数据集可以替代 MNIST。

类别信息：

|Label|Description|
|---|---|
|0|T-shirt/top|
|1|Trouser|
|2|Pullover|
|3|Dress|
|4|Coat|
|5|Sandal|
|6|Shirt|
|7|Sneaker|
|8|Bag|
|9|Ankle boot|

返回 NumPy 数组元素：(x_train, y_train), (x_test, y_test)。

- **x_train**: uint8 numpy 数组的灰度图像数据，shape `(60000, 28, 28)`，训练数据；
- **y_train**: uint8 numpy 数组的标签（整数 0-9），shape `(60000,)`，训练数据；
- **x_test**: uint8 numpy 数组的灰度图像数据，shape `(10000, 28, 28)`，测试数据；
- **y_test**: uint8 numpy 数组的标签（整数 0-9），shape `(10000,)`，测试数据。

## 示例

```python
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
assert x_train.shape == (60000, 28, 28)
assert x_test.shape == (10000, 28, 28)
assert y_train.shape == (60000,)
assert y_test.shape == (10000,)
```

## 参考

- https://www.tensorflow.org/api_docs/python/tf/keras/datasets/fashion_mnist/load_data
