# tf.executing_eagerly

***

## 简介

```python
tf.executing_eagerly()
```

检查当前线程是否启用 eager 执行。

Eager 执行默认启用，所以该 API 在大多数情况下返回 `True`。在以下情况可能返回 `False`:

- 在 `tf.function` 内执行，除非之前调用了 `tf.init_scope` 或 `tf.config.run_functions_eagerly(True)`
- 在 `tf.dataset` 的 transformation 函数内执行
- 调用了 `tf.compat.v1.disable_eager_execution()`

**一般情况：**

```python
>>> print(tf.executing_eagerly())
True
```

- 在 `tf.function` 内：

```python
>>> @tf.function
... def fn():
...     with tf.init_scope():
...         print(tf.executing_eagerly()) # 前面有 tf.init_scope，为 True
...     print(tf.executing_eagerly()) # tf.function 内为 False
>>> fn()
True
False
```

- 调用 `tf.config.run_functions_eagerly(True)` 后在 `tf.function` 内

```python
>>> tf.config.run_functions_eagerly(True) # 后面所有内容都是 eager 执行
>>> @tf.function
... def fn():
...     with tf.init_scope():
...         print(tf.executing_eagerly())
...     print(tf.executing_eagerly())
>>> fn()
True
True
>>> tf.config.run_functions_eagerly(False)
```

- 在 tf.dataset 的 transformation 函数中

```python
>>> def data_fn(x):
>>>     print(tf.executing_eagerly())
>>>     return x
>>> dataset = tf.data.Dataset.range(100)
>>> dataset = dataset.map(data_fn)
False
```

**Returns**

如果当前线程启用了 eager 执行，返回 True。

## 参考

- https://www.tensorflow.org/api_docs/python/tf/executing_eagerly
