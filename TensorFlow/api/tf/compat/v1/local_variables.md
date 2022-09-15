# tf.compat.v1.local_variables

Last updated: 2022-09-15, 14:40
****

```python
tf.compat.v1.local_variables(
    scope=None
)
```

返回 local 变量 list。

local 变量存在于单个进程中，通常不保存到 checkpoint，用于临时保存值。例如，local 变量可用来保存 epoch 数。`tf.contrib.framework.local_variable()` 函数自动将新的变量添加到集合 `GraphKeys.LOCAL_VARIABLES`，而本函数返回该集合内容。

和 local 变量相对的是 global 变量。参考 [tf.compat.v1.global_variables](https://www.tensorflow.org/api_docs/python/tf/compat/v1/global_variables)

|参数|说明|
|----|---|
|scope|(可选) a string。如果提供，则对 local 变量集合进行筛选，使用 `re.match` 匹配变量的 `name` 和 `scope`，没有 `name` 和没匹配上的变量被筛掉|

## 参考

- https://www.tensorflow.org/api_docs/python/tf/compat/v1/local_variables
