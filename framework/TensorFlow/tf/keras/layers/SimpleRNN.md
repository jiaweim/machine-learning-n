# SimpleRNN

- [SimpleRNN](#simplernn)
  - [简介](#简介)
  - [使用注意事项](#使用注意事项)
  - [参考](#参考)

2022-01-30, 10:55
***

## 简介

```python
tf.keras.layers.SimpleRNN(
    units, activation='tanh', use_bias=True,
    kernel_initializer='glorot_uniform',
    recurrent_initializer='orthogonal',
    bias_initializer='zeros', kernel_regularizer=None,
    recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None,
    kernel_constraint=None, recurrent_constraint=None, bias_constraint=None,
    dropout=0.0, recurrent_dropout=0.0, return_sequences=False, return_state=False,
    go_backwards=False, stateful=False, unroll=False, **kwargs
)
```

全连接 RNN，输出送回输入。

|参数|说明|
|---|---|
|units|输出空间的维数|



## 使用注意事项

- SimpleRNN 不擅长处理长序列，比如问题；

## 参考

- https://keras.io/api/layers/recurrent_layers/simple_rnn/
