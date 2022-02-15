# Sequential model

- [Sequential model](#sequential-model)
  - [配置](#配置)
  - [何时使用序列模型](#何时使用序列模型)
  - [创建 Sequential 模型](#创建-sequential-模型)
  - [指定输入 shape](#指定输入-shape)
  - [通用调试工作流](#通用调试工作流)
  - [使用模型](#使用模型)
  - [使用序列模型提取特征](#使用序列模型提取特征)
  - [基于序列模型的迁移学习](#基于序列模型的迁移学习)
  - [参考](#参考)

2021-12-21, 17:27
***

## 配置

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
```

## 何时使用序列模型

序列（`Sequential`）模型适合于简单的层堆栈，每一层只有一个输入张量和一个输出张量。

从语法上来说，下面的序列模型：

```python
# 定义 3 层 Sequential 模型
model = keras.Sequential(
    [
        layers.Dense(2, activation='relu', name='layer1'),
        layers.Dense(3, activation='relu', name='layer2'),
        layers.Dense(4, name='layer3')
    ]
)
# 对示例样本调用模型
x = tf.ones((3, 3))
y = model(x)
```

与下面的函数定义等价：

```python
# Create 3 layers
layer1 = layers.Dense(2, activation="relu", name="layer1")
layer2 = layers.Dense(3, activation="relu", name="layer2")
layer3 = layers.Dense(4, name="layer3")

# Call layers on a test input
x = tf.ones((3, 3))
y = layer3(layer2(layer1(x)))
```

以下情形不适合用 `Sequential` 模型：

- 有多个输入或多个输出；
- 任意一个网络层有多个输入或多个输出；
- 需要实现 layer 共享；
- 需要非线性拓扑结构。

## 创建 Sequential 模型

可以直接将 layer 列表传递给 `Sequential` 的构造函数创建 `Sequential` 模型：

```python
model = keras.Sequential(
    [
        layers.Dense(2, activation="relu"),
        layers.Dense(3, activation="relu"),
        layers.Dense(4),
    ]
)
```

可以通过 `layers` 属性访问网络层：

```py
>>> model.layers
[<keras.layers.core.dense.Dense at 0x1e0d4f2a580>,
 <keras.layers.core.dense.Dense at 0x1e0d4f2ad90>,
 <keras.layers.core.dense.Dense at 0x1e0d4f2aa30>]
```

也可以通过调用 `add()` 创建序列模型：

```python
model = keras.Sequential()
model.add(layers.Dense(2, activation="relu"))
model.add(layers.Dense(3, activation="relu"))
model.add(layers.Dense(4))
```

可以使用 `pop()` 方法移除 layer：

```py
>>> model.pop()
>>> print(len(model.layers))  # 2
2
```

可以使用 `name` 参数为创建的模型指定名称：

```py
model = keras.Sequential(name="my_sequential")
model.add(layers.Dense(2, activation="relu", name="layer1"))
model.add(layers.Dense(3, activation="relu", name="layer2"))
model.add(layers.Dense(4, name="layer3"))
```

## 指定输入 shape

Keras 中的所有 layers 都需要知道其输入 shape，以便能够创建对应的 weights。所以按照如下方式创建 layer，初始状态没有 weights：

```py
>>> layer = layers.Dense(3)
>>> layer.weights  # Empty
[]
```

它在第一次调用某个 input 时根据输入 shape 创建 weights：

```py
>>> # Call layer on a test input
>>> x = tf.ones((1, 4))
>>> y = layer(x)
>>> layer.weights  # Now it has weights, of shape (4, 3) and (3,)
[<tf.Variable 'dense_6/kernel:0' shape=(4, 3) dtype=float32, numpy=
 array([[-0.57910645,  0.01889718,  0.81052744],
        [-0.6837379 , -0.6663338 , -0.09140235],
        [ 0.91084313, -0.40995848,  0.41564345],
        [-0.3284163 , -0.52211523,  0.5777253 ]], dtype=float32)>,
 <tf.Variable 'dense_6/bias:0' shape=(3,) dtype=float32, numpy=array([0., 0., 0.], dtype=float32)>]
```

该行为也适用于 Sequential 模型，在实例化没有输入的 Sequential 模型时，它还没有构建，此时也没有 weights，此时调用 `model.weights` 会出错。在接受输入数据后，模型才创建 weights：

```py
>>> model = keras.Sequential([
        layers.Dense(2, activation='relu'),
        layers.Dense(3, activation='relu'),
        layers.Dense(4)
    ]) # 此时没有 weights
>>> model.weights # 会抛出错误
ValueError                                Traceback (most recent call last)
...
ValueError: Weights for model sequential_4 have not yet been created. Weights are created when the Model is first called on inputs or `build()` is called with an `input_shape`.
>>> model.summary() # 也会抛出错误
ValueError                                Traceback (most recent call last)
...
ValueError: This model has not yet been built. Build the model first by calling `build()` or by calling the model on a batch of data.
>>> x = tf.ones((1, 4))
>>> y = model(x)
>>> len(model.weights)
6
>>> model.summary()
Model: "sequential_4"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 dense_7 (Dense)             (1, 2)                    10        
                                                                 
 dense_8 (Dense)             (1, 3)                    9         
                                                                 
 dense_9 (Dense)             (1, 4)                    16        
                                                                 
=================================================================
Total params: 35
Trainable params: 35
Non-trainable params: 0
```

但是，在以递增的方式构建 Sequential 模型时能够显示模型的 summary 会很有用。此时可以通过向模型传递 `Input` 对象来启动模型，这样它从一开始就知道输入 shape：

```py
>>> model = keras.Sequential()
>>> model.add(keras.Input(shape=(4,)))
>>> model.add(layers.Dense(2, activation='relu'))
>>> model.summary()
Model: "sequential_5"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 dense_10 (Dense)            (None, 2)                 10        
                                                                 
=================================================================
Total params: 10
Trainable params: 10
Non-trainable params: 0
>>> model.layers
[<keras.layers.core.dense.Dense at 0x1e0b35a6f40>]
```

从上面的输出可以发现 `Input` 没有作为 `model.layers` 在 Layer 中显示，因为它不是 layer。

还有一种更简单的指定输入 shape 的方法，对第一层传入 `input_shape` 参数即可：

```py
>>> model = keras.Sequential()
>>> model.add(layers.Dense(2, activation='relu', input_shape=(4,)))
>>> model.summary()
Model: "sequential_7"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 dense_12 (Dense)            (None, 2)                 10        
                                                                 
=================================================================
Total params: 10
Trainable params: 10
Non-trainable params: 0
```

这样预定义输入 shape 的模型总有 weights，也有定义好的输出 shape。

一般来说，如果知道输入 shape，对 Sequential 模型建议提前指定好 shape。

## 通用调试工作流

在构建新的 Sequential 模型时，使用 `add()` 递增堆叠 layers，并使用 `summary()` 查看模型摘要很有用。例如，可以查看 `Conv2D` 和 `MaxPooling2D` 层如何向下采样图像特征的：

```py
>>> model = keras.Sequential()
>>> model.add(keras.Input(shape=(250, 250, 3)))  # 250x250 RGB 图片
>>> model.add(layers.Conv2D(32, 5, strides=2, activation='relu'))
>>> model.add(layers.Conv2D(32, 3, activation='relu'))
>>> model.add(layers.MaxPooling2D(3))

# 此时输出的 shape 是多少？不好计算，但是可以直接输出
>>> model.summary()
Model: "sequential_8"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d (Conv2D)             (None, 123, 123, 32)      2432      
                                                                 
 conv2d_1 (Conv2D)           (None, 121, 121, 32)      9248      
                                                                 
 max_pooling2d (MaxPooling2D  (None, 40, 40, 32)       0         
 )                                                               
                                                                 
=================================================================
Total params: 11,680
Trainable params: 11,680
Non-trainable params: 0

# 是 (40, 40, 32)，继续向下采样
>>> model.add(layers.Conv2D(32, 3, activation="relu"))
>>> model.add(layers.Conv2D(32, 3, activation="relu"))
>>> model.add(layers.MaxPooling2D(3))
>>> model.add(layers.Conv2D(32, 3, activation="relu"))
>>> model.add(layers.Conv2D(32, 3, activation="relu"))
>>> model.add(layers.MaxPooling2D(2))

# 然后再次查看
>>> model.summary()
Model: "sequential_8"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d (Conv2D)             (None, 123, 123, 32)      2432      
                                                                 
 conv2d_1 (Conv2D)           (None, 121, 121, 32)      9248      
                                                                 
 max_pooling2d (MaxPooling2D  (None, 40, 40, 32)       0         
 )                                                               
                                                                 
 conv2d_2 (Conv2D)           (None, 38, 38, 32)        9248      
                                                                 
 conv2d_3 (Conv2D)           (None, 36, 36, 32)        9248      
                                                                 
 max_pooling2d_1 (MaxPooling  (None, 12, 12, 32)       0         
 2D)                                                             
                                                                 
 conv2d_4 (Conv2D)           (None, 10, 10, 32)        9248      
                                                                 
 conv2d_5 (Conv2D)           (None, 8, 8, 32)          9248      
                                                                 
 max_pooling2d_2 (MaxPooling  (None, 4, 4, 32)         0         
 2D)                                                             
                                                                 
=================================================================
Total params: 48,672
Trainable params: 48,672
Non-trainable params: 0
# 此时输出位 4x4 的 feature map，然后添加 global max pooling
>>> model.add(layers.GlobalMaxPool2D())
# 以及最终的分类层
>>> model.add(layers.Dense(10))
```

最终模型如下：

```py
>>> model.summary()
Model: "sequential_8"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d (Conv2D)             (None, 123, 123, 32)      2432      
                                                                 
 conv2d_1 (Conv2D)           (None, 121, 121, 32)      9248      
                                                                 
 max_pooling2d (MaxPooling2D  (None, 40, 40, 32)       0         
 )                                                               
                                                                 
 conv2d_2 (Conv2D)           (None, 38, 38, 32)        9248      
                                                                 
 conv2d_3 (Conv2D)           (None, 36, 36, 32)        9248      
                                                                 
 max_pooling2d_1 (MaxPooling  (None, 12, 12, 32)       0         
 2D)                                                             
                                                                 
 conv2d_4 (Conv2D)           (None, 10, 10, 32)        9248      
                                                                 
 conv2d_5 (Conv2D)           (None, 8, 8, 32)          9248      
                                                                 
 max_pooling2d_2 (MaxPooling  (None, 4, 4, 32)         0         
 2D)                                                             
                                                                 
 global_max_pooling2d (Globa  (None, 32)               0         
 lMaxPooling2D)                                                  
                                                                 
 dense_13 (Dense)            (None, 10)                330       
                                                                 
=================================================================
Total params: 49,002
Trainable params: 49,002
Non-trainable params: 0
```

## 使用模型

准备好模型后，可以使用模型：

- 训练、评估模型，并使用模型 inference；
- 保存并恢复模型；
- 使用多个 GPU 加速训练模型。

## 使用序列模型提取特征

Sequential 构建好后，其行为和函数API模型类似，即每一层都有 `input` 和 `output` 属性。这些属性可用来做一些简单的事情，比如提取 Sequential 模型的所有中间层的输出：

```py
initial_model = keras.Sequential(
    [
        keras.Input(shape=(250, 250, 3)),
        layers.Conv2D(32, 5, strides=2, activation="relu"),
        layers.Conv2D(32, 3, activation="relu"),
        layers.Conv2D(32, 3, activation="relu"),
    ]
)
feature_extractor = keras.Model(
    inputs=initial_model.inputs,
    outputs=[layer.output for layer in initial_model.layers],
)

# Call feature extractor on test input.
x = tf.ones((1, 250, 250, 3))
features = feature_extractor(x)
```

也可以提取某一层的特征：

```py
initial_model = keras.Sequential(
    [
        keras.Input(shape=(250, 250, 3)),
        layers.Conv2D(32, 5, strides=2, activation="relu"),
        layers.Conv2D(32, 3, activation="relu", name="my_intermediate_layer"),
        layers.Conv2D(32, 3, activation="relu"),
    ]
)
feature_extractor = keras.Model(
    inputs=initial_model.inputs,
    outputs=initial_model.get_layer(name="my_intermediate_layer").output,
)
# Call feature extractor on test input.
x = tf.ones((1, 250, 250, 3))
features = feature_extractor(x)
```

## 基于序列模型的迁移学习

迁移学习冻结模型的底层，只循环顶层。下面是两个常见的设计序列模型的迁移学习。

首先，假设有一个序列模型，需要冻结除最后一层外的所有层，此时只需要在模型的 `model.layers` 上进行迭代（除最后一层），对每层设置 `layer.trainable = False`。如下：

```py
model = keras.Sequential([
    keras.Input(shape=(784)),
    layers.Dense(32, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(10),
])

# Presumably you would want to first load pre-trained weights.
model.load_weights(...)

# Freeze all layers except the last one.
for layer in model.layers[:-1]:
  layer.trainable = False

# Recompile and train (this will only update the weights of the last layer).
model.compile(...)
model.fit(...)
```

另一种是使用序列模型堆叠预先训练过的模型和一些刚初始化的分类层。如下：

```py
# Load a convolutional base with pre-trained weights
base_model = keras.applications.Xception(
    weights='imagenet',
    include_top=False,
    pooling='avg')

# Freeze the base model
base_model.trainable = False

# Use a Sequential model to add a trainable classifier on top
model = keras.Sequential([
    base_model,
    layers.Dense(1000),
])

# Compile & train
model.compile(...)
model.fit(...)
```

使用迁移学习会经常使用这两种模式。

## 参考

- https://www.tensorflow.org/guide/keras/sequential_model
