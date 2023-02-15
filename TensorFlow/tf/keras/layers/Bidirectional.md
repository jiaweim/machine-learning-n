# tf.keras.layers.Bidirectional

- [tf.keras.layers.Bidirectional](#tfkeraslayersbidirectional)
  - [简介](#简介)
  - [示例](#示例)
  - [参考](#参考)

Last updated: 2023-02-15, 10:34
****

## 简介

```python
tf.keras.layers.Bidirectional(
    layer,
    merge_mode='concat',
    weights=None,
    backward_layer=None,
    **kwargs
)
```

用于构建双向 RNN。

**参数：**

- **layer**

`keras.layers.RNN` 实例，如 `keras.layers.LSTM` 或 `keras.layers.GRU`。也可以是满足如下条件的 `keras.layers.Layer` 实例：

1. 是序列处理层，即支持 3D+ 输入；
2. 包含 `go_backwards`, `return_sequences` 和 `return_state` 属性，即与 RNN 类具有相同的语义；
3. 包含 `input_spec` 属性；
4. 通过 `get_config()` 和 `from_config()` 实现序列化。注意，创建新的 RNN layer 的推荐方法是自定义 RNN cell，然后将其与 `keras.layers.RNN` 一起使用，而不是直接继承 `keras.layers.Layer `；
5. 当 `returns_sequences=True`，无论 layer 的原 `zero_output_for_mask` 值是什么，mask 时间步的输出为 0

- **merge_mode**

前向和后向 RNN 的输出的组合方式。包括 {'sum', 'mul', 'concat', 'ave', None}。

None 表示不合并输出，直接以 list 返回。默认 'concat'。

- **backward_layer**, optional

用于处理反向输入的`keras.layers.RNN` 或 `keras.layers.Layer` 实例。如果不指定，则使用 `layer` 生成反向层。

若提供 `backward_layer`，则其属性要与 `layer` 匹配，如 `stateful`, `return_states`, `return_sequences` `等值要相同。另外，backward_layer` 和 `layer` 的 `go_backwards` 参数要不同。不满足这些要求将引发 `ValueError`。 

**调用参数：**

该层的调用参数与包装的 `RNN` 层相同。注意，在调用该层期间传如 `initial_state` 参数时，`initial_state` list 元素的前一半传给前向 RNN，后一半传递给反向 RNN。

**异常：**

`ValueError`:

- 如果 `layer` 或 `backward_layer` 不是 `Layer` 实例；
- `merge_mode` 参数无效；
- `backward_layer` 与 `layer` 的属性不匹配。

## 示例

```python
model = Sequential()
model.add(Bidirectional(LSTM(10, return_sequences=True),
                             input_shape=(5, 10)))
model.add(Bidirectional(LSTM(10)))
model.add(Dense(5))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

# With custom backward layer
model = Sequential()
forward_layer = LSTM(10, return_sequences=True)
backward_layer = LSTM(10, activation='relu', return_sequences=True,
                      go_backwards=True)
model.add(Bidirectional(forward_layer, backward_layer=backward_layer,
                        input_shape=(5, 10)))
model.add(Dense(5))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
```

## 参考

- https://www.tensorflow.org/api_docs/python/tf/keras/layers/Bidirectional
