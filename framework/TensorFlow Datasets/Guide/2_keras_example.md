# 用 Keras 在 MNIST 数据集上训练神经网络

- [用 Keras 在 MNIST 数据集上训练神经网络](#用-keras-在-mnist-数据集上训练神经网络)
  - [简介](#简介)
  - [1. 创建输入管道](#1-创建输入管道)
    - [加载数据集](#加载数据集)
    - [构建训练管道](#构建训练管道)
    - [构建评估管道](#构建评估管道)
  - [2. 创建并训练模型](#2-创建并训练模型)
  - [参考](#参考)

2022-01-18, 17:43
@author Jiawei Mao
****

## 简介

下面演示如何在 Keras 模型中使用 TensorFlow Datasets (TFDS) 。

```python
import tensorflow as tf
import tensorflow_datasets as tfds
```

## 1. 创建输入管道

首先构建一个高效的输入管道。

### 加载数据集

使用如下参数加载 MNIST 数据集：

- `shuffle_files=True`，MNIST 数据集保存在单个文件中，但是对大数据集，最好在训练前 shuffle；
- `as_supervised=True`，返回 tuple `(img, label)` 而非 dict `{'image': img, 'label': label}`.

```python
(ds_train, ds_test), ds_info = tfds.load(
    'mnist',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)
```

### 构建训练管道

应用如下转换：

- [tf.data.Dataset.map](../../api/tf/data/Dataset.md#map)：TFDS 提供的图像类型为 `tf.uint8`，而模型需要 `tf.float32` 类型，因此，需要对图像进行归一化处理；
- [tf.data.Dataset.cache](../../api/tf/data/Dataset.md#cache)：在 shuffle 前缓存数据集可以提高性能。

> ⚡ 应该在缓存之后应用随机转换。

- [tf.data.Dataset.shuffle](../../api/tf/data/Dataset.md#shuffle)：为了实现真正的随机，应该将 shuffer buffer 设置为数据集大小。

> ⚡ 对内存无法容纳的大型数据集，如果系统允许，可以设置 `buffer_size=1000`。

- [tf.data.Dataset.batch](../../api/tf/data/Dataset.md#batch)：每个 epoch对数据集 shuffle 之后，再分批，保证每次训练的 batch 都不同。
- [tf.data.Dataset.prefetch](../../api/tf/data/Dataset.md#prefetch)：推荐通过 prefetch 来结束管道流，可以提高性能。

```python
def normalize_img(image, label):
  """Normalizes images: `uint8` -> `float32`."""
  return tf.cast(image, tf.float32) / 255., label

ds_train = ds_train.map(
    normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
ds_train = ds_train.cache()
ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
ds_train = ds_train.batch(128)
ds_train = ds_train.prefetch(tf.data.AUTOTUNE)
```

### 构建评估管道

测试管道和训练管道类似，主要差别：

- 不需要调用 `tf.data.Dataset.shuffle`；
- 可以在 batch 之后再缓存，因为对验证集不同 epoch 的 batch可以相同。

```python
ds_test = ds_test.map(
    normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
ds_test = ds_test.batch(128)
ds_test = ds_test.cache()
ds_test = ds_test.prefetch(tf.data.AUTOTUNE)
```

## 2. 创建并训练模型

将 TFDS 输入管道插入 Keras 模型中，编译并训练模型：

```python
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(10)
])
model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
)

model.fit(
    ds_train,
    epochs=6,
    validation_data=ds_test,
)
```

```sh
Epoch 1/6
469/469 [==============================] - 4s 4ms/step - loss: 0.3545 - sparse_categorical_accuracy: 0.9017 - val_loss: 0.1848 - val_sparse_categorical_accuracy: 0.9476
Epoch 2/6
469/469 [==============================] - 2s 3ms/step - loss: 0.1619 - sparse_categorical_accuracy: 0.9538 - val_loss: 0.1404 - val_sparse_categorical_accuracy: 0.9573
Epoch 3/6
469/469 [==============================] - 2s 3ms/step - loss: 0.1181 - sparse_categorical_accuracy: 0.9665 - val_loss: 0.1157 - val_sparse_categorical_accuracy: 0.9656
Epoch 4/6
469/469 [==============================] - 2s 3ms/step - loss: 0.0928 - sparse_categorical_accuracy: 0.9730 - val_loss: 0.1004 - val_sparse_categorical_accuracy: 0.9708
Epoch 5/6
469/469 [==============================] - 2s 3ms/step - loss: 0.0757 - sparse_categorical_accuracy: 0.9777 - val_loss: 0.0984 - val_sparse_categorical_accuracy: 0.9695
Epoch 6/6
469/469 [==============================] - 2s 3ms/step - loss: 0.0626 - sparse_categorical_accuracy: 0.9815 - val_loss: 0.0851 - val_sparse_categorical_accuracy: 0.9739
<keras.callbacks.History at 0x2c22c0aecd0>
```

## 参考

- https://www.tensorflow.org/datasets/keras_example
