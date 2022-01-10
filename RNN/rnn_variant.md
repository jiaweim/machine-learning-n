# RNN 类型

## 双向 RNN

TensorFlow 2.0 通过一个双向包装层提供对双向 RNN 的支持，以 LSTM 为例：

```python
self.lstm = tf.keras.layers.Bidirectional(
    tf.keras.layers.LSTM(10, return_sequences=True, input_shape=(5, 10))
)
```
