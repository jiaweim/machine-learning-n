# tf.function

## 简介

```python
tf.function(
    func=None,
    input_signature=None,
    autograph=True,
    jit_compile=None,
    reduce_retracing=False,
    experimental_implements=None,
    experimental_autograph_options=None,
    experimental_relax_shapes=None,
    experimental_compile=None,
    experimental_follow_type_hints=None
) -> tf.types.experimental.GenericFunction
```

将函数编译为可调用 TF graph。

> **WARNING**：`experimental_compile` 参数已弃用，改用 `jit_compile`。
> 

## 参考

- https://tensorflow.google.cn/api_docs/python/tf/function
