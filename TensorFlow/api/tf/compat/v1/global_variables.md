# tf.compat.v1.global_variables

Last updated: 2022-09-15, 14:58
****

```python
tf.compat.v1.global_variables(
    scope=None
)
```

返回 global 变量 list。

global 变量在分布式环境中跨机器共享。`Variable()` 构造函数或 `get_variable()` 自动将新变量添加到 graph 集合 `GraphKeys.GLOBAL_VARIABLES`，而本函数返回该集合的内容。

与 global 变量相对的是 local 变量。参考 [tf.compat.v1.local_variables](https://www.tensorflow.org/api_docs/python/tf/compat/v1/local_variables)

|参数|说明|
|----|---|
|scope|(可选) a string。如果提供，则对 global 变量集合进行筛选，使用 `re.match` 匹配变量的 `name` 和 `scope`，没有 `name` 和没匹配上的变量被筛掉|

## 参考

- https://www.tensorflow.org/api_docs/python/tf/compat/v1/global_variables
