# tf.keras.layers.GRU

2022-03-09, 22:51
***

## 简介

```python
tf.keras.layers.GRU(
    units, activation='tanh', recurrent_activation='sigmoid',
    use_bias=True, kernel_initializer='glorot_uniform',
    recurrent_initializer='orthogonal',
    bias_initializer='zeros', kernel_regularizer=None,
    recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None,
    kernel_constraint=None, recurrent_constraint=None, bias_constraint=None,
    dropout=0.0, recurrent_dropout=0.0, return_sequences=False, return_state=False,
    go_backwards=False, stateful=False, unroll=False, time_major=False,
    reset_after=True, **kwargs
)
```

门控循环单元。

基于可用的运行时硬件，GRU 会选择不同的实现（基于 cuDNN 或纯 TensorFlow）来最大化性能。如果 GPU 可用，且所有参数满足 cuDNN 内核要求（如下），则使用 cuDNN 实现。

使用 cuDNN 实现的要求如下：

1. `activation == tanh`
2. `recurrent_activation == sigmoid`
3. `recurrent_dropout == 0`
4. `unroll` is `False`
5. `use_bias` is `True`
6. `reset_after` is `True`
7. 输入如果使用 masking，则必须右边填充
8. 最外层 context 启用 eager 执行

GRU 实现有两种变体。默认是基于 v3 的实现，在矩阵乘法前对隐藏层应用 reset gate。另一个基于原始文献，顺序相反。

第二个变体兼容 CuDNNGRU (GPU-only)，并允许在 CPU 上进行推理。它对 `kernel` 和 `recurrent_kernel` 有不同偏好，使用该变体，需要设置 `reset_after=True` 和 `recurrent_activation='sigmoid'`。

|参数|说明|
|---|---|
|units|Positive integer, dimensionality of the output space.
|activation|Activation function to use. Default: hyperbolic tangent (tanh). If you pass None, no activation is applied (ie. "linear" activation: a(x) = x).
|recurrent_activation|Activation function to use for the recurrent step. Default: sigmoid (sigmoid). If you pass None, no activation is applied (ie. "linear" activation: a(x) = x).
|use_bias|Boolean, (default True), whether the layer uses a bias vector.|
|kernel_initializer|Initializer for the kernel weights matrix, used for the linear transformation of the inputs. Default: glorot_uniform.|
|recurrent_initializer|Initializer for the recurrent_kernel weights matrix, used for the linear transformation of the recurrent state. Default: orthogonal.|
|bias_initializer|Initializer for the bias vector. Default: zeros.|
|kernel_regularizer|Regularizer function applied to the kernel weights matrix. Default: None.|
|recurrent_regularizer|Regularizer function applied to the recurrent_kernel weights matrix. Default: None.|
|bias_regularizer|Regularizer function applied to the bias vector. Default: None.|
|activity_regularizer|Regularizer function applied to the output of the layer (its "activation"). Default: None.|
|kernel_constraint|Constraint function applied to the kernel weights matrix. Default: None.|
|recurrent_constraint|Constraint function applied to the recurrent_kernel weights matrix. Default: None.|
|bias_constraint|Constraint function applied to the bias vector. Default: None.|
|dropout|Float between 0 and 1. Fraction of the units to drop for the linear transformation of the inputs. Default: 0.|
|recurrent_dropout|Float between 0 and 1. Fraction of the units to drop for the linear transformation of the recurrent state. Default: 0.|
|return_sequences|Boolean. Whether to return the last output in the output sequence, or the full sequence. Default: False.|
|return_state|Boolean. Whether to return the last state in addition to the output. Default: False.|
|go_backwards|Boolean (default False). If True, process the input sequence backwards and return the reversed sequence.|
|stateful|Boolean (default False). If True, the last state for each sample at index i in a batch will be used as initial state for the sample of index i in the following batch.|
|unroll|Boolean (default False). If True, the network will be unrolled, else a symbolic loop will be used. Unrolling can speed-up a RNN, although it tends to be more memory-intensive. Unrolling is only suitable for short sequences.
|time_major|The shape format of the inputs and outputs tensors. If True, the inputs and outputs will be in shape [timesteps, batch, feature], whereas in the False case, it will be [batch, timesteps, feature]. Using time_major = True is a bit more efficient because it avoids transposes at the beginning and end of the RNN calculation. However, most TensorFlow data is batch-major, so by default this function accepts input and emits output in batch-major form.
|reset_after|GRU convention (whether to apply reset gate after or before matrix multiplication). False = "before", True = "after" (default and cuDNN compatible).

## 参考

- https://www.tensorflow.org/api_docs/python/tf/keras/layers/GRU
