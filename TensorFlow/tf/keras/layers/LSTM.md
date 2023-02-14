# LSTM

2022-02-21, 15:58
***

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

基于可运行的硬件条件，`LSTM` 会选择不同实现以最大化性能，即选择基于 cuDNN 的实现或纯 tensorflow 实现。如果有 GPU 可用，并且所有参数满足下面的 cuDNN 内核要求，则选择更快的 cuDNN 实现。

使用 cuDNN 实现需要满足如下要求：

1. `activation` == `tanh`
2. `recurrent_activation` == `sigmoid`
3. `recurrent_dropout` == 0
4. `unroll` == `False`
5. `use_bias` == `True`
6. 输入如果使用 masking，必须为 right-padded.
7. Eager execution is enabled in the outermost context.

例如：

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
|stateful|Boolean (default False). True 表示当前 batch 样本 i 的最终状态用作下一个 batch 的样本 i 的初始状态|
|time_major|The shape format of the inputs and outputs tensors. If True, the inputs and outputs will be in shape [timesteps, batch, feature], whereas in the False case, it will be [batch, timesteps, feature]. Using time_major = True is a bit more efficient because it avoids transposes at the beginning and end of the RNN calculation. However, most TensorFlow data is batch-major, so by default this function accepts input and emits output in batch-major form.|
|unroll|Boolean (default False). If True, the network will be unrolled, else a symbolic loop will be used. Unrolling can speed-up a RNN, although it tends to be more memory-intensive. Unrolling is only suitable for short sequences.|

- **dropout**



## 调用参数

- `inputs`

A 3D tensor with shape `[batch, timesteps, feature]`.

- `mask`

Binary tensor of shape [batch, timesteps] indicating whether a given timestep should be masked (optional, defaults to None). An individual True entry indicates that the corresponding timestep should be utilized, while a False entry indicates that the corresponding timestep should be ignored.

- `training`

Binary tensor of shape [batch, timesteps] indicating whether a given timestep should be masked (optional, defaults to None). An individual True entry indicates that the corresponding timestep should be utilized, while a False entry indicates that the corresponding timestep should be ignored.

- `initial_state`

List of initial state tensors to be passed to the first call of the cell (optional, defaults to None which causes creation of zero-filled initial state tensors).

## 方法

### get_dropout_mask_for_cell

```python
get_dropout_mask_for_cell(
    inputs, training, count=1
)
```

### reset_states

```python
reset_states(
    states=None
)
```

重置 stateful RNN layer 的状态。

仅当使用 `stateful = True` 构造 RNN 层时才能使用。参数 `states` 为表示初始状态的 numpy 数组，`None` 时将自动创建合适 shape 的全零 numpy 数组作为初始状态。

> 在处理长序列数据时，需要维持 RNN 的隐藏状态，这一功能通常用单词 "stateful" 表示，许多深度学习框架都支持该参数，表示是否保存上一时刻的隐藏状态。

## 参考

- https://www.tensorflow.org/api_docs/python/tf/keras/layers/LSTM
