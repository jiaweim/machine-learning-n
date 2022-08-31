# 保存和加载模型

- [保存和加载模型](#保存和加载模型)
  - [1. 简介](#1-简介)
  - [2. 选择](#2-选择)
  - [3. 设置](#3-设置)
    - [3.1 安装和导入](#31-安装和导入)
    - [3.2 获取示例数据](#32-获取示例数据)
    - [3.3 定义模型](#33-定义模型)
  - [4. 训练时保存 checkpoints](#4-训练时保存-checkpoints)
    - [4.1 Checkpoint callback 使用](#41-checkpoint-callback-使用)
    - [4.2 Checkpoint callback 选项](#42-checkpoint-callback-选项)
  - [5. 文件说明](#5-文件说明)
  - [6. 手动保存权重](#6-手动保存权重)
  - [7. 保存整个模型](#7-保存整个模型)
    - [7.1 SavedModel 格式](#71-savedmodel-格式)
    - [7.2 HDF5 格式](#72-hdf5-格式)
    - [7.3 保存自定义对象](#73-保存自定义对象)
  - [8. 参考](#8-参考)

Last updated: 2022-08-30, 15:38
@author Jiawei Mao
****

## 1. 简介

可以在训练过程中和训练后保存模型状态，这样模型可以从停止的地方恢复，以避免长时间的训练。能够保存模型也意味着可以共享模型，在发布模型和技术研究工作时，大多会共享：

- 创建模型的代码
- 模型的权重和参数

共享这些数据有助于他人了解模型的工作原理，并在新数据进行尝试。

## 2. 选择

根据所使用的 API，保存 TF 模型的方法也不同。本教程使用 tf.keras API 在 TF 中构建和训练模型。其它方法可以参考 [Using the SavedModel format](https://www.tensorflow.org/guide/saved_model) 和 [Save and load Keras models](https://www.tensorflow.org/guide/keras/save_and_serialize).

## 3. 设置

### 3.1 安装和导入

安装并导入 TensorFlow 和依赖项：

```powershell
pip install pyyaml h5py  # Required to save models in HDF5 format
```

```python
import os

import tensorflow as tf
from tensorflow import keras

print(tf.version.VERSION)
```

```txt
2.9.1
```

### 3.2 获取示例数据

下面使用 MNIST 数据集演示如何保存和加载权重。为了快速演示，只使用前 1000 个样本：

```python
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

train_labels = train_labels[:1000]
test_labels = test_labels[:1000]

train_images = train_images[:1000].reshape(-1, 28 * 28) / 255.0
test_images = test_images[:1000].reshape(-1, 28 * 28) / 255.0
```

### 3.3 定义模型

首先构建一个简单的 Sequential 模型：

```python
# Define a simple sequential model
def create_model():
    model = tf.keras.Sequential(
        [
            keras.layers.Dense(512, activation="relu", input_shape=(784,)),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(10),
        ]
    )

    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )

    return model


# Create a basic model instance
model = create_model()

# Display the model's architecture
model.summary()
```

```txt
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 dense (Dense)               (None, 512)               401920    
                                                                 
 dropout (Dropout)           (None, 512)               0         
                                                                 
 dense_1 (Dense)             (None, 10)                5130      
                                                                 
=================================================================
Total params: 407,050
Trainable params: 407,050
Non-trainable params: 0
_________________________________________________________________
```

## 4. 训练时保存 checkpoints

`tf.keras.callbacks.ModelCheckpoint` callback 可用于在训练期间和训练后保存模型。

### 4.1 Checkpoint callback 使用

创建一个只在训练期间保存权重的 `tf.keras.callbacks.ModelCheckpoint` callback:

```python
checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# 创建保存模型 weights 的 callback
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path, save_weights_only=True, verbose=1
)

# 使用新的 callback 训练模型
model.fit(
    train_images,
    train_labels,
    epochs=10,
    validation_data=(test_images, test_labels),
    callbacks=[cp_callback]
)  # Pass callback to training
```

```txt
Epoch 1/10
27/32 [========================>.....] - ETA: 0s - loss: 1.2181 - sparse_categorical_accuracy: 0.6412
Epoch 1: saving model to training_1\cp.ckpt
32/32 [==============================] - 2s 15ms/step - loss: 1.1210 - sparse_categorical_accuracy: 0.6720 - val_loss: 0.6737 - val_sparse_categorical_accuracy: 0.8010
Epoch 2/10
29/32 [==========================>...] - ETA: 0s - loss: 0.4270 - sparse_categorical_accuracy: 0.8847
Epoch 2: saving model to training_1\cp.ckpt
32/32 [==============================] - 0s 10ms/step - loss: 0.4148 - sparse_categorical_accuracy: 0.8880 - val_loss: 0.5796 - val_sparse_categorical_accuracy: 0.8060
......
Epoch 9/10
22/32 [===================>..........] - ETA: 0s - loss: 0.0504 - sparse_categorical_accuracy: 0.9957
Epoch 9: saving model to training_1\cp.ckpt
32/32 [==============================] - 0s 9ms/step - loss: 0.0528 - sparse_categorical_accuracy: 0.9960 - val_loss: 0.3934 - val_sparse_categorical_accuracy: 0.8750
Epoch 10/10
32/32 [==============================] - ETA: 0s - loss: 0.0436 - sparse_categorical_accuracy: 0.9980
Epoch 10: saving model to training_1\cp.ckpt
32/32 [==============================] - 0s 9ms/step - loss: 0.0436 - sparse_categorical_accuracy: 0.9980 - val_loss: 0.4183 - val_sparse_categorical_accuracy: 0.8620
<keras.callbacks.History at 0x17e17edf7c0>
```

这样就创建好了一个 TF checkpoint 文件集合，在每个 epoch 后更新这些文件：

```python
os.listdir(checkpoint_dir)
```

```txt
['checkpoint', 'cp.ckpt.data-00000-of-00001', 'cp.ckpt.index']
```

只要两个模型架构相同，就可以在它们之间共享权重。因此，当从权重恢复模型时，需要先创建一个与原始模型架构相同的模型，然后设置其权重。

现在重建一个新的、没训练过的模型，并在测试集上进行评估。没训练过的模型性能接近随机水平（~10% 的准确率）：

```python
# Create a basic model instance
model = create_model()

# Evaluate the model
loss, acc = model.evaluate(test_images, test_labels, verbose=2)
print("Untrained model, accuracy: {:5.2f}%".format(100 * acc))
```

```txt
32/32 - 0s - loss: 2.3703 - sparse_categorical_accuracy: 0.0950 - 173ms/epoch - 5ms/step
Untrained model, accuracy:  9.50%
```

从 checkpoint 载入权重，并重新评估模型：

```python
# Loads the weights
model.load_weights(checkpoint_path)

# Re-evaluate the model
loss, acc = model.evaluate(test_images, test_labels, verbose=2)
print("Restored model, accuracy: {:5.2f}%".format(100 * acc))
```

```txt
32/32 - 0s - loss: 0.4197 - sparse_categorical_accuracy: 0.8650 - 106ms/epoch - 3ms/step
Restored model, accuracy: 86.50%
```

### 4.2 Checkpoint callback 选项

Checkpoint callback 可以设置 checkpoint 名称和 checkpoint 频率。

例如，训练一个模型，每 5 个 epochs 保存一个 checkpoints：

```python
# Include the epoch in the file name (uses `str.format`)
checkpoint_path = "training_2/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

batch_size = 32

# Create a callback that saves the model's weights every 5 epochs
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    verbose=1,
    save_weights_only=True,
    save_freq=5 * batch_size,
)

# Create a new model instance
model = create_model()

# Save the weights using the `checkpoint_path` format
model.save_weights(checkpoint_path.format(epoch=0))

# Train the model with the new callback
model.fit(
    train_images,
    train_labels,
    epochs=50,
    batch_size=batch_size,
    callbacks=[cp_callback],
    validation_data=(test_images, test_labels),
    verbose=0,
)
```

```txt
Epoch 5: saving model to training_2\cp-0005.ckpt

Epoch 10: saving model to training_2\cp-0010.ckpt

Epoch 15: saving model to training_2\cp-0015.ckpt

Epoch 20: saving model to training_2\cp-0020.ckpt

Epoch 25: saving model to training_2\cp-0025.ckpt

Epoch 30: saving model to training_2\cp-0030.ckpt

Epoch 35: saving model to training_2\cp-0035.ckpt

Epoch 40: saving model to training_2\cp-0040.ckpt

Epoch 45: saving model to training_2\cp-0045.ckpt

Epoch 50: saving model to training_2\cp-0050.ckpt
<keras.callbacks.History at 0x22c88300160>
```

查看生成的 checkpoints 并选择最新的：

```python
os.listdir(checkpoint_dir)
```

```txt
['checkpoint',
 'cp-0000.ckpt.data-00000-of-00001',
 'cp-0000.ckpt.index',
 'cp-0005.ckpt.data-00000-of-00001',
 'cp-0005.ckpt.index',
 'cp-0010.ckpt.data-00000-of-00001',
 'cp-0010.ckpt.index',
 'cp-0015.ckpt.data-00000-of-00001',
 'cp-0015.ckpt.index',
 'cp-0020.ckpt.data-00000-of-00001',
 'cp-0020.ckpt.index',
 'cp-0025.ckpt.data-00000-of-00001',
 'cp-0025.ckpt.index',
 'cp-0030.ckpt.data-00000-of-00001',
 'cp-0030.ckpt.index',
 'cp-0035.ckpt.data-00000-of-00001',
 'cp-0035.ckpt.index',
 'cp-0040.ckpt.data-00000-of-00001',
 'cp-0040.ckpt.index',
 'cp-0045.ckpt.data-00000-of-00001',
 'cp-0045.ckpt.index',
 'cp-0050.ckpt.data-00000-of-00001',
 'cp-0050.ckpt.index']
```

```python
latest = tf.train.latest_checkpoint(checkpoint_dir)
latest
```

```txt
'training_2/cp-0050.ckpt'
```

现在，重置模型，载入最新 checkpoint：

```python
# Create a new model instance
model = create_model()

# Load the previously saved weights
model.load_weights(latest)

# Re-evaluate the model
loss, acc = model.evaluate(test_images, test_labels, verbose=2)
print("Restored model, accuracy: {:5.2f}%".format(100 * acc))
```

```txt
32/32 - 0s - loss: 0.4864 - sparse_categorical_accuracy: 0.8740 - 169ms/epoch - 5ms/step
Restored model, accuracy: 87.40%
```

## 5. 文件说明

上面的代码将权重存储到 checkpoint 文件集合，该二进制格式文件只包含训练后的权重。checkpoints 文件包括：

- 包含模型权重的一个或多个 shards
- 索引文件，指定权重在哪个 shard

如果是在单个机器上训练的，则只有一个 shard，后缀为 `.data-00000-of-00001`。

## 6. 手动保存权重

使用 `tf.keras.Model.save_weights` 手动保存权重。`Model.save_weights` 方法默认使用 TF checkpoint 格式保存权重，`.ckpt` 扩展名。要保存为 HDF5 格式（`.h5` 扩展），可参考 [Save and load Keras models](https://www.tensorflow.org/guide/keras/save_and_serialize)。

```python
# Save the weights
model.save_weights("./checkpoints/my_checkpoint")

# Create a new model instance
model = create_model()

# Restore the weights
model.load_weights("./checkpoints/my_checkpoint")

# Evaluate the model
loss, acc = model.evaluate(test_images, test_labels, verbose=2)
print("Restored model, accuracy: {:5.2f}%".format(100 * acc))
```

```txt
32/32 - 0s - loss: 0.4864 - sparse_categorical_accuracy: 0.8740 - 181ms/epoch - 6ms/step
Restored model, accuracy: 87.40%
```

## 7. 保存整个模型

调用 `tf.keras.Model.save` 可将模型的架构、权重和训练配置保存到单个文件/目录。这样导出的模型，不需要创建模型的原始 Python 代码就可以使用该模型。由于 optimizer 的状态也恢复了，所以可以从停止的地方恢复训练。

完整的模型可以保存为两种文件格式 `SavedModel` 和 `HDF5`。TF `SavedModel` 格式为 TF2.x 的默认格式。

### 7.1 SavedModel 格式

SavedModel 是另一种序列化模型的方式，以该格式保存的模型可以使用 `tf.keras.models.load_model` 恢复，并且与 TF 服务兼容。[SavedModel guide](https://www.tensorflow.org/guide/saved_model) 详细介绍了如何使用 SavedModel。下面说明如何保存和恢复模型。

```python
# Create and train a new model instance.
model = create_model()
model.fit(train_images, train_labels, epochs=5)

# Save the entire model as a SavedModel.
!mkdir -p saved_model
model.save('saved_model/my_model')
```

```txt
Epoch 1/5
32/32 [==============================] - 0s 6ms/step - loss: 1.1627 - sparse_categorical_accuracy: 0.6610
Epoch 2/5
32/32 [==============================] - 0s 5ms/step - loss: 0.4104 - sparse_categorical_accuracy: 0.8870
Epoch 3/5
32/32 [==============================] - 0s 5ms/step - loss: 0.2832 - sparse_categorical_accuracy: 0.9330
Epoch 4/5
32/32 [==============================] - 0s 6ms/step - loss: 0.2084 - sparse_categorical_accuracy: 0.9540
Epoch 5/5
32/32 [==============================] - 0s 5ms/step - loss: 0.1397 - sparse_categorical_accuracy: 0.9720
INFO:tensorflow:Assets written to: saved_model/my_model\assets
```

SavedModel 格式是一个目录，包含 protobuf binary 和 TF checkpoint。检查 SavedModel 目录：

```powershell
# my_model directory
ls saved_model

# Contains an assets folder, saved_model.pb, and variables folder.
ls saved_model/my_model
```

```txt
my_model
assets  keras_metadata.pb  saved_model.pb  variables
```

从 saved model 重新加载新的 Keras 模型：

```python
new_model = tf.keras.models.load_model("saved_model/my_model")

# Check its architecture
new_model.summary()
```

```txt
Model: "sequential_5"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 dense_10 (Dense)            (None, 512)               401920    
                                                                 
 dropout_5 (Dropout)         (None, 512)               0         
                                                                 
 dense_11 (Dense)            (None, 10)                5130      
                                                                 
=================================================================
Total params: 407,050
Trainable params: 407,050
Non-trainable params: 0
_________________________________________________________________
```

恢复的模型与原始模型具有相同的参数。对加载的模型进行评估和预测：

```python
# Evaluate the restored model
loss, acc = new_model.evaluate(test_images, test_labels, verbose=2)
print("Restored model, accuracy: {:5.2f}%".format(100 * acc))

print(new_model.predict(test_images).shape)
```

```txt
32/32 - 0s - loss: 0.4293 - sparse_categorical_accuracy: 0.8640 - 179ms/epoch - 6ms/step
Restored model, accuracy: 86.40%
32/32 [==============================] - 0s 2ms/step
(1000, 10)
```

### 7.2 HDF5 格式

Keras 支持保存为 HDF5 格式。

```python
# Create and train a new model instance.
model = create_model()
model.fit(train_images, train_labels, epochs=5)

# Save the entire model to a HDF5 file.
# The '.h5' extension indicates that the model should be saved to HDF5.
model.save("my_model.h5")
```

```txt
Epoch 1/5
32/32 [==============================] - 0s 6ms/step - loss: 1.1112 - sparse_categorical_accuracy: 0.6940
Epoch 2/5
32/32 [==============================] - 0s 5ms/step - loss: 0.4098 - sparse_categorical_accuracy: 0.8880
Epoch 3/5
32/32 [==============================] - 0s 5ms/step - loss: 0.2890 - sparse_categorical_accuracy: 0.9250
Epoch 4/5
32/32 [==============================] - 0s 5ms/step - loss: 0.1997 - sparse_categorical_accuracy: 0.9450
Epoch 5/5
32/32 [==============================] - 0s 5ms/step - loss: 0.1543 - sparse_categorical_accuracy: 0.9650
```

从 HDF5 文件重新创建模型：

```python
# Recreate the exact same model, including its weights and the optimizer
new_model = tf.keras.models.load_model("my_model.h5")

# Show the model architecture
new_model.summary()
```

```txt
Model: "sequential_6"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 dense_12 (Dense)            (None, 512)               401920    
                                                                 
 dropout_6 (Dropout)         (None, 512)               0         
                                                                 
 dense_13 (Dense)            (None, 10)                5130      
                                                                 
=================================================================
Total params: 407,050
Trainable params: 407,050
Non-trainable params: 0
_________________________________________________________________
```

检查其精度：

```python
loss, acc = new_model.evaluate(test_images, test_labels, verbose=2)
print("Restored model, accuracy: {:5.2f}%".format(100 * acc))
```

```txt
32/32 - 0s - loss: 0.4416 - sparse_categorical_accuracy: 0.8520 - 205ms/epoch - 6ms/step
Restored model, accuracy: 85.20%
```

Keras 通过检查模型结构来保存模型。该技术保存所有内容：

- weight 值
- model 架构
- model 训练配置（`.compile()` 设置内容）
- optimizer 及其状态

Keras 无法保存 v1.x optimizers (来自 `tf.compat.v1.train`)，因为它们与 checkpoint 不兼容。对 v1.x optimizers，需要在加载模型后重新编译，因为 optimizer 的状态已丢失。

### 7.3 保存自定义对象

如果使用 SavedModel 格式，可以跳过本节。HDF5 和 SavedModel 的关键区别在于，HDF5 使用对象配置来保存模型架构，而 SavedModel 保存执行 graph。因此 SavedModel 能够保存自定义对象，如 subclass 模型、自定义 layer。

要将自定义对象保存到 HDF5，需要执行如下操作：

1. 在自定义对象中定义 `get_config` 方法，以及可选的 `from_config` classmethod。
   - `get_config(self)` 返回重建对象所需参数的 JSON 序列化的 dict。
   - `from_config(cls, config)` 使用 `get_config` 返回的 config 创建新对象。该函数默认使用该 config 作为初始化 kwargs `return cls(**config)`。
2. 加载模型时，将自定义对象传递给 `custom_objects` 参数。该参数必须是将类名（string）映射到 Python 类的 dict，例如 `tf.keras.models.load_model(path, custom_objects={'CustomLayer': CustomLayer})`。

有关自定义对象和 `get_config` 的示例，可以参考 [Making new Layers and Models via subclassing](https://www.tensorflow.org/guide/keras/custom_layers_and_models)。

## 8. 参考

- https://www.tensorflow.org/tutorials/keras/save_and_load
