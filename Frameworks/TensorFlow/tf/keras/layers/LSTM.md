# LSTM

- [LSTM](#lstm)
  - [简介](#简介)
  - [示例](#示例)
  - [参数](#参数)
  - [调用参数](#调用参数)
  - [属性](#属性)
  - [方法](#方法)
    - [get\_dropout\_mask\_for\_cell](#get_dropout_mask_for_cell)
    - [get\_recurrent\_dropout\_mask\_for\_cell](#get_recurrent_dropout_mask_for_cell)
    - [reset\_dropout\_mask](#reset_dropout_mask)
    - [reset\_recurrent\_dropout\_mask](#reset_recurrent_dropout_mask)
    - [reset\_states](#reset_states)
  - [参考](#参考)

Last updated: 2023-02-18, 13:18
****

## 简介

```python
tf.keras.layers.LSTM(
    units,
    activation='tanh',
    recurrent_activation='sigmoid',
    use_bias=True,
    kernel_initializer='glorot_uniform',
    recurrent_initializer='orthogonal',
    bias_initializer='zeros',
    unit_forget_bias=True,
    kernel_regularizer=None,
    recurrent_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    kernel_constraint=None,
    recurrent_constraint=None,
    bias_constraint=None,
    dropout=0.0,
    recurrent_dropout=0.0,
    return_sequences=False,
    return_state=False,
    go_backwards=False,
    stateful=False,
    time_major=False,
    unroll=False,
    **kwargs
)
```

基于可运行的硬件条件，`LSTM` 会选择不同实现以最大化性能，即选择 cuDNN 实现和纯 tf 实现。如果有 GPU 且所有参数满足下面的 cuDNN 内核要求，则选择更快的 cuDNN 实现。

使用 cuDNN 实现需要满足如下要求：

1. `activation` == `tanh`：默认
2. `recurrent_activation` == `sigmoid`：默认
3. `recurrent_dropout` == 0：默认，所以使用 `recurrent_dropout` 意味着放弃 cuDNN GPU 性能
4. `unroll` == `False`：默认
5. `use_bias` == `True`：默认
6. 输入如果使用 masking，必须为 right-padded.
7. Eager execution is enabled in the outermost context.

## 示例

```python
>>> inputs = tf.random.normal([32, 10, 8])
>>> lstm = tf.keras.layers.LSTM(4)
>>> output = lstm(inputs)
>>> print(output.shape) # batch, hidden
(32, 4)
>>> lstm = tf.keras.layers.LSTM(4, return_sequences=True, return_state=True)
>>> whole_seq_output, final_memory_state, final_carry_state = lstm(inputs)
>>> print(whole_seq_output.shape) # batch, time, hidden
(32, 10, 4)
>>> print(final_memory_state.shape)
(32, 4)
>>> print(final_carry_state.shape)
(32, 4)
```

## 参数

|参数|说明|
|---|---|
|units|Positive integer, dimensionality of the output space|
|activation|Activation function to use. Default: hyperbolic tangent (tanh). If you pass None, no activation is applied (ie. "linear" activation: a(x) = x)|
|recurrent_activation|Activation function to use for the recurrent step. Default: sigmoid. If you pass None, no activation is applied (ie. "linear" activation: a(x) = x)|
|use_bias|Boolean (default True), whether the layer uses a bias vector|
|kernel_initializer|Initializer for the **kernel** weights matrix, used for the linear transformation of the inputs. Default: **glorot_uniform**|
|recurrent_initializer|Initializer for the **recurrent_kernel** weights matrix, used for the linear transformation of the recurrent state. Default: **orthogonal**|
|bias_initializer|Initializer for the bias vector. Default: **zeros**|
|unit_forget_bias|Boolean (default `True`). If True, add 1 to the bias of the forget gate at initialization. Setting it to true will also force `bias_initializer="zeros"`. This is recommended in [Jozefowicz et al..](http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf)|
|kernel_regularizer|Regularizer function applied to the `kernel` weights matrix. Default: `None`|
|recurrent_regularizer|Regularizer function applied to the `recurrent_kernel` weights matrix. Default: `None`|
|bias_regularizer|Regularizer function applied to the bias vector. Default: `None`|
|activity_regularizer|Regularizer function applied to the output of the layer (its "activation"). Default: `None`|
|kernel_constraint|Constraint function applied to the `kernel` weights matrix. Default: `None`|
|recurrent_constraint|Constraint function applied to the recurrent_kernel weights matrix. Default: None|
|bias_constraint|Constraint function applied to the bias vector. Default: None.|
|dropout|Float between 0 and 1. Fraction of the units to drop for the linear transformation of the inputs. Default: 0.|
|recurrent_dropout|Float between 0 and 1. Fraction of the units to drop for the linear transformation of the recurrent state. Default: 0.|
|return_sequences|Boolean，是否返回完整序列的输出，默认 False|
|return_state|Boolean，是否额外返回最终状态，默认 False|
|go_backwards|Boolean (default False). If True, process the input sequence backwards and return the reversed sequence.|

- **stateful=False**

Boolean，表示当前 batch 样本 i 的最终状态用作下一个 batch 样本 i 的初始状态。

|time_major|The shape format of the inputs and outputs tensors. If True, the inputs and outputs will be in shape [timesteps, batch, feature], whereas in the False case, it will be [batch, timesteps, feature]. Using time_major = True is a bit more efficient because it avoids transposes at the beginning and end of the RNN calculation. However, most TensorFlow data is batch-major, so by default this function accepts input and emits output in batch-major form.|
|unroll|Boolean (default False). If True, the network will be unrolled, else a symbolic loop will be used. Unrolling can speed-up a RNN, although it tends to be more memory-intensive. Unrolling is only suitable for short sequences.|

- **dropout**

## 调用参数

- `inputs`

shape 为 `[batch, timesteps, feature]` 的 3D 张量。

- `mask`

shape 为 `[batch, timesteps]` 的 2D 张量，指定是否屏蔽指定时间步，默认 `None`。`True` 表示使用对应时间步，`False` 表示忽略对应时间步。

- `training`

Binary tensor of shape [batch, timesteps] indicating whether a given timestep should be masked (optional, defaults to None). An individual True entry indicates that the corresponding timestep should be utilized, while a False entry indicates that the corresponding timestep should be ignored.

- `initial_state`

List of initial state tensors to be passed to the first call of the cell (optional, defaults to None which causes creation of zero-filled initial state tensors).

## 属性

## 方法

### get_dropout_mask_for_cell

```python
get_dropout_mask_for_cell(
    inputs, training, count=1
)
```

返回 RNN cell 输入的 dropout mask。

如果没有已缓存的 mask，则基于 context 创建新的 mask，并以新创建的 mask 更新 cell 缓存的 mask。

**参数：**

- **inputs**

输入张量，将用其 shape 生成 dropout mask。

- **training**

Boolean 张量，是否处于训练模式，在非训练模式下忽略 dropout。

- **count**

int，生成多少个 dropout mask。对内部包含权重融合的 cell 有用。

**返回：**

mask 张量 list，根据 context 生成或缓存的 mask。

### get_recurrent_dropout_mask_for_cell

```python
get_recurrent_dropout_mask_for_cell(
    inputs, training, count=1
)
```

返回 RNN cell 的 recurreent dropout mask。

如果没有已缓存的 mask，则基于 context 创建新的 mask，并以新创建的 mask 更新 cell 缓存的 mask。

**参数：**

- **inputs**

输入张量，将用其 shape 生成 dropout mask。

- **training**

Boolean 张量，是否处于训练模式，在非训练模式下忽略 dropout。

- **count**

int，生成多少个 dropout mask。对内部包含权重融合的 cell 有用。

**返回：**

mask 张量 list，根据 context 生成或缓存的 mask。

### reset_dropout_mask

```python
reset_dropout_mask()
```

重置缓存的 dropout mask。

在 RNN 层的 `call()` 方法中调用该方法很重要，以便在调用 `cell.call()` 之前清除缓存的 mask。该 mask 在同一个 batch 中应该跨时间步缓存，但不应该在 batch 之间缓存。否则会在 batch 数据的特定 index 产生不合理的 bias。

### reset_recurrent_dropout_mask

```python
reset_recurrent_dropout_mask()
```

重置缓存的 recurrent dropout mask。

在 RNN 层的 `call()` 方法中调用该方法很重要，以便在调用 `cell.call()` 之前清除缓存的 mask。该 mask 在同一个 batch 中应该跨时间步缓存，但不应该在 batch 之间缓存。否则会在 batch 数据的特定 index 产生不合理的 bias。

### reset_states

```python
reset_states(
    states=None
)
```

重置 stateful RNN 层的状态。

仅当使用 `stateful = True` 构造 RNN 层时使用。参数 `states` 为表示初始状态的 numpy 数组，`None` 时将自动创建合适 shape 的全零 numpy 数组作为初始状态。

> 在处理长序列数据时，需要维持 RNN 的隐藏状态，这一功能通常用单词 "stateful" 表示，许多深度学习框架都支持该参数，表示是否保存上一时刻的隐藏状态。

## 参考

- https://www.tensorflow.org/api_docs/python/tf/keras/layers/LSTM
