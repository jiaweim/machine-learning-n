# 模块、层和模型

- [模块、层和模型](#模块层和模型)
  - [简介](#简介)
  - [定义模型和 layers](#定义模型和-layers)
    - [创建变量延迟](#创建变量延迟)
  - [保存 weights](#保存-weights)
  - [保存函数](#保存函数)
    - [创建 SavedModel](#创建-savedmodel)
  - [Keras models and layers](#keras-models-and-layers)
    - [Keras layers](#keras-layers)
    - [build 步骤](#build-步骤)
    - [Keras models](#keras-models)
  - [保存 keras 模型](#保存-keras-模型)
  - [参考](#参考)

2021-12-26, 21:44
@author Jiawei Mao
***

## 简介

使用 TensorFlow 进行机器学习，需要定义、保存和恢复模型。

模型，抽象的说，是：

- 一个使用张量计算的函数（向前传播）；
- 在训练过程中，会更新部分变量会。

下面我们介绍如何定义 TensorFlow 模型，包括如何收集变量和模型，如何保存和恢复模型。

配置：

```python
import tensorflow as tf
from datetime import datetime

%load_ext tensorboard
```

## 定义模型和 layers

大部分模型由 layer 组成。layer 是具有已知数学结构的函数，可以重复使用并具有可训练的变量。在 TensorFlow，大多数 layer 和 model 的高级实现，如 Keras 或 Sonnet，都构建在相同的基础类 `tf.Module` 上。

下面是一个对标量张量进行操作的简单 `tf.Module`：

```python
class SimpleModule(tf.Module):
  def __init__(self, name=None):
    super().__init__(name=name)
    self.a_variable = tf.Variable(5.0, name="train_me")
    self.non_trainable_variable = tf.Variable(5.0, trainable=False, name="do_not_train_me")
  def __call__(self, x):
    return self.a_variable * x + self.non_trainable_variable

simple_module = SimpleModule(name="simple")

simple_module(tf.constant(5.0))
```

```sh
<tf.Tensor: shape=(), dtype=float32, numpy=30.0>
```

模块，及其扩展 layer 都是深度学习属于的“对象”：它们具有内置状态，以及使用该状态的方法。

上面的 `__call__` 方法没有任何特别之处，除了是一个 Python 可调用对象，你可以使用任何你喜欢的函数调用你的模型。

你可以根据需要打开或关闭变量的可训练性，包括 fine-tuning 期间冻结 layer 和变量。

> `tf.Module` 是 `tf.keras.layers.Layer` 和 `tf.keras.Model` 的基类，所以这里描述的内容也适用于 Keras。出于兼容性考虑，Keras 的 layer 不从模型收集变量，所以模型要么只使用模块，要么只使用 keras layer。不过，下面展示的检查变量的方法对两种方式都适用。

通过扩展 `tf.Module`，会自动收集该对象的任何 `tf.Variable` 或 `tf.Module` 实例。这样就能保存和加载变量，并可以创建 `tf.Module` 集合。

```python
# All trainable variables
print("trainable variables:", simple_module.trainable_variables)
# Every variable
print("all variables:", simple_module.variables)
```

```sh
trainable variables: (<tf.Variable 'train_me:0' shape=() dtype=float32, numpy=5.0>,)
all variables: (<tf.Variable 'train_me:0' shape=() dtype=float32, numpy=5.0>, <tf.Variable 'do_not_train_me:0' shape=() dtype=float32, numpy=5.0>)
2021-10-26 01:29:45.284549: W tensorflow/python/util/util.cc:348] Sets are not currently considered sequences, but this may change in the future, so consider avoiding using them.
```

下面是一个由 module 组成的两层线性层模型能。

首先是一个 dense linear layer:

```python
class Dense(tf.Module):
  def __init__(self, in_features, out_features, name=None):
    super().__init__(name=name)
    self.w = tf.Variable(
      tf.random.normal([in_features, out_features]), name='w')
    self.b = tf.Variable(tf.zeros([out_features]), name='b')
  def __call__(self, x):
    y = tf.matmul(x, self.w) + self.b
    return tf.nn.relu(y)
```

然后是完整的模型，创建了两个 layer 实例：

```python
class SequentialModule(tf.Module):
  def __init__(self, name=None):
    super().__init__(name=name)

    self.dense_1 = Dense(in_features=3, out_features=3)
    self.dense_2 = Dense(in_features=3, out_features=2)

  def __call__(self, x):
    x = self.dense_1(x)
    return self.dense_2(x)

# You have made a model!
my_model = SequentialModule(name="the_model")

# Call it, with random results
print("Model results:", my_model(tf.constant([[2.0, 2.0, 2.0]])))
```

```sh
Model results: tf.Tensor([[7.706234  3.0919805]], shape=(1, 2), dtype=float32)
```

`tf.Module` 实例会自动地、递归地收集分配它的任何 `tf.Variable` 和 `tf.Module` 实例。这样就能用一个模型实例管理 `tf.Module` 集合，保存及加载整个模型。

```python
print("Submodules:", my_model.submodules)
```

```sh
Submodules: (<__main__.Dense object at 0x7f7ab2391290>, <__main__.Dense object at 0x7f7b6869ea10>)
```

```python
for var in my_model.variables:
  print(var, "\n")
```

```sh
<tf.Variable 'b:0' shape=(3,) dtype=float32, numpy=array([0., 0., 0.], dtype=float32)> 

<tf.Variable 'w:0' shape=(3, 3) dtype=float32, numpy=
array([[ 0.05711935,  0.22440144,  0.6370985 ],
       [ 0.3136791 , -1.7006774 ,  0.7256515 ],
       [ 0.16120772, -0.8412193 ,  0.5250952 ]], dtype=float32)> 

<tf.Variable 'b:0' shape=(2,) dtype=float32, numpy=array([0., 0.], dtype=float32)> 

<tf.Variable 'w:0' shape=(3, 2) dtype=float32, numpy=
array([[-0.5353216 ,  1.2815404 ],
       [ 0.62764466,  0.47087234],
       [ 2.19187   ,  0.45777202]], dtype=float32)>
```

### 创建变量延迟

在上面定义 layer 时需要指定输入和输出 size。这样才找到 `w` 变量从而分配内存。

通过将变量的创建推迟到模块第一次被调用时，就不需要在前面指定输入 size。如下：

```python
class FlexibleDenseModule(tf.Module):
  # Note: No need for `in_features`
  def __init__(self, out_features, name=None):
    super().__init__(name=name)
    self.is_built = False
    self.out_features = out_features

  def __call__(self, x):
    # Create variables on first call.
    if not self.is_built:
      self.w = tf.Variable(
        tf.random.normal([x.shape[-1], self.out_features]), name='w')
      self.b = tf.Variable(tf.zeros([self.out_features]), name='b')
      self.is_built = True

    y = tf.matmul(x, self.w) + self.b
    return tf.nn.relu(y)
```

```python
# Used in a module
class MySequentialModule(tf.Module):
  def __init__(self, name=None):
    super().__init__(name=name)

    self.dense_1 = FlexibleDenseModule(out_features=3)
    self.dense_2 = FlexibleDenseModule(out_features=2)

  def __call__(self, x):
    x = self.dense_1(x)
    return self.dense_2(x)

my_model = MySequentialModule(name="the_model")
print("Model results:", my_model(tf.constant([[2.0, 2.0, 2.0]])))
```

```sh
Model results: tf.Tensor([[4.0598335 0.       ]], shape=(1, 2), dtype=float32)
```

所以 TensorFlow 的 layer 通常只需要指定输出的 shape，如 `tf.keras.layer.Dense`中，而不需要同时指定输入和输出 size。

## 保存 weights

`tf.Module` 可以保存为 [checkpoint](save_model/save_checkpoint.md) 或 [SavedModel](save_model/save_savedmodel.md)。

checkpoint 只包含 weights，即 module 及其子 module 包含的变量的值：

```python
chkp_path = "my_checkpoint"
checkpoint = tf.train.Checkpoint(model=my_model)
checkpoint.write(chkp_path)
```

```sh
'my_checkpoint'
```

checkpoint 由两类文件组成：数据本身和索引文件。索引文件记录实际保存的内容赫尔 checkpoint 编号，而 checkpoint 数据包含变量值及属性查询路径。

```sh
$ls my_checkpoint*
```

```sh
my_checkpoint.data-00000-of-00001  my_checkpoint.index
```

可以查看 checkpoint 内部，以确保整个变量集合被保存下来，按照包含这些变量的 Python 对象排序：

```python
tf.train.list_variables(chkp_path)
```

```sh
[('_CHECKPOINTABLE_OBJECT_GRAPH', []),
 ('model/dense_1/b/.ATTRIBUTES/VARIABLE_VALUE', [3]),
 ('model/dense_1/w/.ATTRIBUTES/VARIABLE_VALUE', [3, 3]),
 ('model/dense_2/b/.ATTRIBUTES/VARIABLE_VALUE', [2]),
 ('model/dense_2/w/.ATTRIBUTES/VARIABLE_VALUE', [3, 2])]
```

在分布式（多机）训练中，checkpoint 可以被分片，所以才给它们编号，例如 '00000-of-00001'。不过上例中，只有一个。

在重新加载模型时，将覆盖 Python 对象中的值。

```python
new_model = MySequentialModule()
new_checkpoint = tf.train.Checkpoint(model=new_model)
new_checkpoint.restore("my_checkpoint")

# Should be the same result as above
new_model(tf.constant([[2.0, 2.0, 2.0]]))
```

```sh
<tf.Tensor: shape=(1, 2), dtype=float32, numpy=array([[4.0598335, 0.       ]], dtype=float32)>
```

> 由于 checkpoint 是长时间训练流程的核心，因此提供了辅助类 `tf.checkpoint.CheckpointManager` 辅助 checkpoint 管理。

## 保存函数

TensorFlow 可以在没有原始 Python 对象的情况下运行模型，就像 TensorFlow Serving 和 TensorFlow Lite 中演示的。

TensorFlow 需要知道如何执行 Python 中描述的计算，但不需要原始代码。要做到这一点，需要用到 graph，在 [graph 执行](basic_graph.md) 中有详细讨论。

graph 包含实现函数的操作（`ops`）。

通过在函数上添加 `@tf.function` 装饰器可以让代码以 graph 运行。

```python
class MySequentialModule(tf.Module):
  def __init__(self, name=None):
    super().__init__(name=name)

    self.dense_1 = Dense(in_features=3, out_features=3)
    self.dense_2 = Dense(in_features=3, out_features=2)

  @tf.function
  def __call__(self, x):
    x = self.dense_1(x)
    return self.dense_2(x)

# You have made a model with a graph!
my_model = MySequentialModule(name="the_model")
```

这个模块和上面一个完全相同。传递到该函数的每个不同的参数都会创建一个单独的 graph。

```python
print(my_model([[2.0, 2.0, 2.0]]))
print(my_model([[[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]]]))
```

```sh
tf.Tensor([[0.62891716 0.        ]], shape=(1, 2), dtype=float32)
tf.Tensor(
[[[0.62891716 0.        ]
  [0.62891716 0.        ]]], shape=(1, 2, 2), dtype=float32)
```

可以在 TensorBoard 中可视化 graph：

```python
# Set up logging.
stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
logdir = "logs/func/%s" % stamp
writer = tf.summary.create_file_writer(logdir)

# Create a new model to get a fresh trace
# Otherwise the summary will not see the graph.
new_model = MySequentialModule()

# Bracket the function call with
# tf.summary.trace_on() and tf.summary.trace_export().
tf.summary.trace_on(graph=True)
tf.profiler.experimental.start(logdir)
# Call only one tf.function when tracing.
z = print(new_model(tf.constant([[2.0, 2.0, 2.0]])))
with writer.as_default():
  tf.summary.trace_export(
      name="my_func_trace",
      step=0,
      profiler_outdir=logdir)
```

```sh
tf.Tensor([[0.         0.01750386]], shape=(1, 2), dtype=float32)
```

启动 TensorBoard 以查看生成的 trace：

```python
%tensorboard --logdir logs/func
```

![](2021-12-27-16-04-49.png)

### 创建 SavedModel

推荐使用 `SavedModel` 共享训练好的模型。`SavedModel` 同时包含函数和 weight 集合。

可以按如下方式保存训练好的模型：

```python
tf.saved_model.save(my_model, "the_saved_model")
```

```sh
INFO:tensorflow:Assets written to: the_saved_model/assets
```

```sh
$ # Inspect the SavedModel in the directory
$ ls -l the_saved_model
total 24
drwxr-sr-x 2 kbuilder kokoro  4096 Oct 26 01:29 assets
-rw-rw-r-- 1 kbuilder kokoro 14702 Oct 26 01:29 saved_model.pb
drwxr-sr-x 2 kbuilder kokoro  4096 Oct 26 01:29 variables
```

```sh
$ # The variables/ directory contains a checkpoint of the variables
$ ls -l the_saved_model/variables
total 8
-rw-rw-r-- 1 kbuilder kokoro 408 Oct 26 01:29 variables.data-00000-of-00001
-rw-rw-r-- 1 kbuilder kokoro 356 Oct 26 01:29 variables.index
```

`saved_model.pb` 文件是描述 `tf.Graph` 的 [protocol buffer](https://developers.google.com/protocol-buffers)。

可以从此表示中加载模型和 layer，而无需最初创建该模型的 Python 代码，这对模型的分发十分有利。

可以将模型作为新对象加载：

```python
new_model = tf.saved_model.load("the_saved_model")
```

从加载保存的模型创建的 `new_model`，是 TensorFlow 内部对象，不是 `SequentialModule` 类型。

```python
isinstance(new_model, SequentialModule)
```

```sh
False
```

这个新模型适用于已定义的输入签名，不能使用新的签名。

```python
print(my_model([[2.0, 2.0, 2.0]]))
print(my_model([[[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]]]))
```

```sh
tf.Tensor([[0.62891716 0.        ]], shape=(1, 2), dtype=float32)
tf.Tensor(
[[[0.62891716 0.        ]
  [0.62891716 0.        ]]], shape=(1, 2, 2), dtype=float32)
```

因此，使用 `SavedModel`，可以使用 `tf.Module` 保存 TensorFlow weights 和 graphs，然后在需要时加载。

## Keras models and layers

目前为止没有提到 Keras。你可以在 `tf.Module` 的基础上构建自己的高级 API。

下面，我们将查看 Keras 如何使用 `tf.Module`。

### Keras layers

`tf.keras.layers.Layer` 是所有 keras layers 的基类，而它继承自 `tf.Module`。

将 module 转换为 keras layer，只需要将父类替换为 `tf.keras.layers.Layer`，将 `__call__` 修改为 `call`：

```python
class MyDense(tf.keras.layers.Layer):
  # Adding **kwargs to support base Keras layer arguments
  def __init__(self, in_features, out_features, **kwargs):
    super().__init__(**kwargs)

    # This will soon move to the build step; see below
    self.w = tf.Variable(
      tf.random.normal([in_features, out_features]), name='w')
    self.b = tf.Variable(tf.zeros([out_features]), name='b')
  def call(self, x):
    y = tf.matmul(x, self.w) + self.b
    return tf.nn.relu(y)

simple_layer = MyDense(name="simple", in_features=3, out_features=3)
```

Keras layers 有自己的 `__call__` 实现，在其中包含一些记录功能，然后调用 `call()`。功能上基本不变。

```python
simple_layer([[2.0, 2.0, 2.0]])
```

```sh
<tf.Tensor: shape=(1, 3), dtype=float32, numpy=array([[0.      , 0.179402, 0.      ]], dtype=float32)>
```

### build 步骤

如前所述，在确定输入 shape 之后再创建变量更方便。

Keras layer 带有额外的生命周期，从而可以更灵活地定义 layer。该额外步骤在 `build` 函数中定故意。

`build` 根据输入的 shape 被调用一次。它通常用于创建变量（weights）。

我们可以重写上面的 `MyDense` 层，使其输入的大小更灵活：

```python
class FlexibleDense(tf.keras.layers.Layer):
  # Note the added `**kwargs`, as Keras supports many arguments
  def __init__(self, out_features, **kwargs):
    super().__init__(**kwargs)
    self.out_features = out_features

  def build(self, input_shape):  # Create the state of the layer (weights)
    self.w = tf.Variable(
      tf.random.normal([input_shape[-1], self.out_features]), name='w')
    self.b = tf.Variable(tf.zeros([self.out_features]), name='b')

  def call(self, inputs):  # Defines the computation from inputs to outputs
    return tf.matmul(inputs, self.w) + self.b

# Create the instance of the layer
flexible_dense = FlexibleDense(out_features=3)
```

此时还没有构建模型，所以还没有变量：

```python
flexible_dense.variables
```

```sh
[]
```

调用函数会自动生成 size 正确的变量：

```python
# Call it, with predictably random results
print("Model results:", flexible_dense(tf.constant([[2.0, 2.0, 2.0], [3.0, 3.0, 3.0]])))
```

```sh
Model results: tf.Tensor(
[[-1.6998017  1.6444504 -1.3103955]
 [-2.5497022  2.4666753 -1.9655929]], shape=(2, 3), dtype=float32)
```

```python
flexible_dense.variables
```

```sh
[<tf.Variable 'flexible_dense/w:0' shape=(3, 3) dtype=float32, numpy=
 array([[ 1.277462  ,  0.5399406 , -0.301957  ],
        [-1.6277349 ,  0.7374014 , -1.7651852 ],
        [-0.49962795, -0.45511687,  1.4119445 ]], dtype=float32)>,
 <tf.Variable 'flexible_dense/b:0' shape=(3,) dtype=float32, numpy=array([0., 0., 0.], dtype=float32)>]
```

由于 `build` 只被调用一次，如果输入的 shape 与 layer 变量不兼容，会抛出错误：

```python
try:
  print("Model results:", flexible_dense(tf.constant([[2.0, 2.0, 2.0, 2.0]])))
except tf.errors.InvalidArgumentError as e:
  print("Failed:", e)
```

```sh
Failed: In[0] mismatch In[1] shape: 4 vs. 3: [1,4] [3,3] 0 0 [Op:MatMul]
```

keras layer 还有许多额外功能，包括：

- optional losses
- 支持 metrics
- 内置 optional `training` 参数，以区分 training 和 inference
- `get_config` 和 `from_config` 方法可以准确地存储配置信息，从而在 Python 中克隆模型。

### Keras models

你可以将模型定义为嵌套的 keras layer。

但是，keras 提供了一个功能完整的模型类 `tf.keras.Model`。它继承自 `tf.keras.layers.Layer`，因此 keras 模型可以和 keras layer 意义昂使用、嵌套以及保存。Keras 模型带有一些方便训练、评估、载入和保存的功能，还提供了在多台机器上进行训练的功能。

可以使用和上面`SequentialModule`几乎相同的代码定义模型，记得修改父类为 `tf.keras.Model`，将 `__call__` 转换为 `call()`：

```python
class MySequentialModel(tf.keras.Model):
  def __init__(self, name=None, **kwargs):
    super().__init__(**kwargs)

    self.dense_1 = FlexibleDense(out_features=3)
    self.dense_2 = FlexibleDense(out_features=2)
  def call(self, x):
    x = self.dense_1(x)
    return self.dense_2(x)

# You have made a Keras model!
my_sequential_model = MySequentialModel(name="the_model")

# Call it on a tensor, with random results
print("Model results:", my_sequential_model(tf.constant([[2.0, 2.0, 2.0]])))
```

```sh
Model results: tf.Tensor([[5.5604653 3.3511646]], shape=(1, 2), dtype=float32)
```

> 嵌套在 keras layer 或 model 中的原始 `tf.Module` 不会为训练或保存收集变量，但是 keras layer 嵌套在 keras layer 中可以。

```python
my_sequential_model.variables
```

```sh
[<tf.Variable 'my_sequential_model/flexible_dense_1/w:0' shape=(3, 3) dtype=float32, numpy=
 array([[ 0.05627853, -0.9386015 , -0.77410126],
        [ 0.63149   ,  1.0802224 , -0.37785745],
        [-0.24788402, -1.1076807 , -0.5956209 ]], dtype=float32)>,
 <tf.Variable 'my_sequential_model/flexible_dense_1/b:0' shape=(3,) dtype=float32, numpy=array([0., 0., 0.], dtype=float32)>,
 <tf.Variable 'my_sequential_model/flexible_dense_2/w:0' shape=(3, 2) dtype=float32, numpy=
 array([[-0.93912166,  0.77979285],
        [ 1.4049559 , -1.9380962 ],
        [-2.6039495 ,  0.30885765]], dtype=float32)>,
 <tf.Variable 'my_sequential_model/flexible_dense_2/b:0' shape=(2,) dtype=float32, numpy=array([0., 0.], dtype=float32)>]
```

```python
my_sequential_model.submodules
```

```sh
(<__main__.FlexibleDense at 0x7f7b48525550>,
 <__main__.FlexibleDense at 0x7f7b48508d10>)
```

覆盖 `tf.keras.Model` 构建 TensorFlow 模型是一种非常 Pythonic 的方法。如果要从其它框架迁移模型，会非常直接。

如果使用现有 layers 和 inputs 简单组合构建模型，则使用 functional API 会而非常方便，它围绕模型重建和架构提供了额外功能。

下面是使用 functional API 定义的相同模型：

```python
inputs = tf.keras.Input(shape=[3,])

x = FlexibleDense(3)(inputs)
x = FlexibleDense(2)(x)

my_functional_model = tf.keras.Model(inputs=inputs, outputs=x)

my_functional_model.summary()
```

```sh
Model: "model"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         [(None, 3)]               0         
_________________________________________________________________
flexible_dense_3 (FlexibleDe (None, 3)                 12        
_________________________________________________________________
flexible_dense_4 (FlexibleDe (None, 2)                 8         
=================================================================
Total params: 20
Trainable params: 20
Non-trainable params: 0
```

```python
my_functional_model(tf.constant([[2.0, 2.0, 2.0]]))
```

```sh
<tf.Tensor: shape=(1, 2), dtype=float32, numpy=array([[8.219393, 4.511119]], dtype=float32)>
```

这里的主要差别是，在 functional API 构建过程中要预先指定输入 shape。这里 `input_shape` 参数与不需要完全指定，不指定的部分设置为 `None`。

> 在子类化的模型中不需要指定 `input_shape` 或 `InputLayer`，这些参数和 layers 会被忽略。

## 保存 keras 模型

keras 模型可以设置 checkpoint，和 `tf.Module` 基本一样。

keas 模型也可以使用 `tf.saved_model.save()` 保存，因为它们也是模块。不过，keras 模型提供了更方便的方法：

```python
my_sequential_model.save("exname_of_file")
```

```sh
INFO:tensorflow:Assets written to: exname_of_file/assets
```

然后重新载入：

```python
reconstructed_model = tf.keras.models.load_model("exname_of_file")
```

```sh
WARNING:tensorflow:No training configuration found in save file, so the model was *not* compiled. Compile it manually.
```

keras 的 `SavedModels` 还保存了 metric, loss 和 optimizer 的状态。

这个重构的模型可以直接使用，在相同数据上调用可以生成和原模型一样的结果：

```python
reconstructed_model(tf.constant([[2.0, 2.0, 2.0]]))
```

```sh
<tf.Tensor: shape=(1, 2), dtype=float32, numpy=array([[5.5604653, 3.3511646]], dtype=float32)>
```

关于 keras 模型的保存和序列化的更多内容，可以参考 [保存和加载 keras 模型](keras/keras_save_load.md)。

## 参考

- https://www.tensorflow.org/guide/intro_to_modules
