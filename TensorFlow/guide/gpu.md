# GPU 的使用

- [GPU 的使用](#gpu-的使用)
  - [简介](#简介)
  - [配置](#配置)
  - [记录设备位置](#记录设备位置)
  - [手动设置设备](#手动设置设备)
  - [显示 GPU 显存](#显示-gpu-显存)
  - [多 GPU 平台使用单个 GPU](#多-gpu-平台使用单个-gpu)
  - [使用多个 GPU](#使用多个-gpu)
    - [tf.distribute.Strategy](#tfdistributestrategy)
    - [手动](#手动)
  - [参考](#参考)

Last updated: 2023-02-14, 15:51
****

## 简介

TensorFlow 代码和 tf.keras 模型无需修改代码就能在单个 GPU 上运行。

> **NOTE**
> 使用 `tf.config.list_physical_devices('GPU')` 确认是否有可用 GPU。

本教程介绍 GPU 的细粒度控制。TensorFlow 支持在各种设备上运行计算，包括 CPU 和 GPU。设备用字符串标识符表示，如：

- `"/device:CPU:0"`：指 CPU
- `"/GPU:0"`：TF 检测到的机器上的第一个 GPU
- `"/job:localhost/replica:0/task:0/device:GPU:1"`：TF 检测到的机器上的第二个 GPU 的完全名称

一个 TF 操作如果同时有 CPU 和 GPU 实现，则优先使用 GPU。例如，`tf.matmul` 支持 CPU 和 GPU，在同时包含 `CPU:0` 和 `GPU:0` 设备的机器上，TF 选择 `GPU:0` 运行 `tf.matmul`，除非显式指定其它设备。

如果一个 TF 操作只有 CPU 实现，则该操作将返回 CPU 上运行。例如，`tf.cast` 只有 CPU 实现，因此在同时包含 `CPU:0` 和 `GPU:0` 设备的机器上，在 `CPU:0` 上运行该操作，即使显式指定 `GPU:0` 也是如此。

## 配置

```python
import tensorflow as tf

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
```

```txt
Num GPUs Available:  1
```

## 记录设备位置

将 `tf.debugging.set_log_device_placement(True)` 放在程序第一行，可以显示操作和张量所在设备。启用该功能会打印所有张量分配和操作。

```python
tf.debugging.set_log_device_placement(True)

# Create some tensors
a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
c = tf.matmul(a, b)

print(c)
```

```txt
Executing op _EagerConst in device /job:localhost/replica:0/task:0/device:GPU:0
Executing op _EagerConst in device /job:localhost/replica:0/task:0/device:GPU:0
Executing op MatMul in device /job:localhost/replica:0/task:0/device:GPU:0
tf.Tensor(
[[22. 28.]
 [49. 64.]], shape=(2, 2), dtype=float32)
```

## 手动设置设备

如果需要将某个操作放在特定设备上运行，可以使用 `with tf.device` 创建设备 context，该 context 中的所有操作都在指定设备运行。

```python
tf.debugging.set_log_device_placement(True)

# Place tensors on the CPU
with tf.device('/CPU:0'):
    a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

# Run on the GPU
c = tf.matmul(a, b)
print(c)
```

```txt
Executing op MatMul in device /job:localhost/replica:0/task:0/device:GPU:0
tf.Tensor(
[[22. 28.]
 [49. 64.]], shape=(2, 2), dtype=float32)
```

可以看到，`a` 和 `b` 放到了 `CPU:0`。由于没有为 `MatMul` 操作指定设备，TF 在运行时将根据操作和可用设备（本例中为 `GPU:0`）选择一个设备，并在需要时自动在设备之间复制张量。

## 显示 GPU 显存

TF 默认使用所有 GPU（受 [CUDA_VISIBLE_DEVICES](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#env-vars) 约束）的所有内存。这样可以通过减少内存碎片来更有效地使用 GPU 内存资源。使用 `tf.config.set_visible_devices` 可以将 TF 限制到一组特定的 GPU。

```python
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    # Restrict TensorFlow to only use the first GPU
    try:
        tf.config.set_visible_devices(gpus[0], 'GPU')
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
    except RuntimeError as e:
        # Visible devices must be set before GPUs have been initialized
        print(e)
```

```txt
1 Physical GPUs, 1 Logical GPU
```

在某些情况，希望进程只分配可用内存的一部分，或者按需增长。TF 提供了两个方法来实现这一点。

通过 `tf.config.experimental.set_memory_growth` 启用内存增长功能，即尝试只分配运行时所需的 GPU  内存：分配的初始内存很少，当程序运行需要更多 GPU 内存时，再扩展 TF 进程的 GPU 内存。不会释放内存，以避免内存碎片。使用以下代码来为特定 GPU 启用内存增长功能：

```python
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)
```

启用该选项的另一种方法是将环境变量 `TF_FORCE_GPU_ALLOW_GROWTH` 设置为 `true`。该配置依赖于平台。

第二种方法是使用 `tf.config.set_logical_device_configuration` 配置虚拟 GPU 设备，设置在 GPU 上分配内存的上限。

```python
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
    try:
        tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=1024)])
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)
```

当与其它应用程序（如 GUI）共享 GPU 时，这是本地开发的常见用法。

## 多 GPU 平台使用单个 GPU

当系统上有多个 GPU，TF 默认选择 ID 最低的 GPU。如果要在不同的 GPU 上运行，则需要显式指定：

```python
tf.debugging.set_log_device_placement(True)

try:
    # Specify an invalid GPU device
    with tf.device('/device:GPU:2'):
        a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        c = tf.matmul(a, b)
except RuntimeError as e:
    print(e)
```

如果指定的设备不存在，将引发 `RuntimeError: .../device:GPU:2 unknown device`。

如果希望 TF 在指定的设备不存在时自动选择可用设备，可调用 `tf.config.set_soft_device_placement(True)`。

```python
tf.config.set_soft_device_placement(True)
tf.debugging.set_log_device_placement(True)

# Creates some tensors
a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
c = tf.matmul(a, b)

print(c)
```

## 使用多个 GPU

如果只有一个 GPU，可以使用虚拟设备模拟多个 GPU，从而可以在不需要额外资源的情况下测试多 GPU 设备。

```python
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    # Create 2 virtual GPUs with 1GB memory each
    try:
        tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=1024),
             tf.config.LogicalDeviceConfiguration(memory_limit=1024)])
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPU,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)
```

当有多个逻辑 GPU 可用时，可以使用 `tf.distribute.Strategy` 或手动来利用多 GPU。

### tf.distribute.Strategy

使用 `tf.distribute.Strategy` 是利用多 GPU 的最佳方法。例如：

```python
tf.debugging.set_log_device_placement(True)
gpus = tf.config.list_logical_devices('GPU')
strategy = tf.distribute.MirroredStrategy(gpus)
with strategy.scope():
    inputs = tf.keras.layers.Input(shape=(1,))
    predictions = tf.keras.layers.Dense(1)(inputs)
    model = tf.keras.models.Model(inputs=inputs, outputs=predictions)
    model.compile(loss='mse',
                  optimizer=tf.keras.optimizers.SGD(learning_rate=0.2))
```

该程序将在每个 GPU 上运行模型的一个 copy，在它们之间拆分输入数据，也称为数据并行。

### 手动

`tf.distribute.Strategy` 通过跨设备复制计算。也可以通过在每个 GPU 上构建模型来手动复制。例如：

```python
tf.debugging.set_log_device_placement(True)

gpus = tf.config.list_logical_devices('GPU')
if gpus:
    # Replicate your computation on multiple GPUs
    c = []
    for gpu in gpus:
        with tf.device(gpu.name):
            a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
            b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
            c.append(tf.matmul(a, b))

    with tf.device('/CPU:0'):
        matmul_sum = tf.add_n(c)

    print(matmul_sum)
```

## 参考

- https://www.tensorflow.org/guide/gpu
