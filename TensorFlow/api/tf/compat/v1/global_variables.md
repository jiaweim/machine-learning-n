# tf.compat.v1.global_variables

```python
tf.compat.v1.global_variables(
    scope=None
)
```

返回 global 变量。

global 变量在分布式环境中跨机器共享。`Variable()` 构造函数或 `get_variable()` 自动将新变量添加到 graph 集合 `GraphKeys.GLOBAL_VARIABLES`。这个函数返回该集合的内容。

与 global 变量相对的是 local 变量。

## 参考

- https://www.tensorflow.org/api_docs/python/tf/compat/v1/global_variables
