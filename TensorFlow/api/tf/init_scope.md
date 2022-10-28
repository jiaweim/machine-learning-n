# tf.init_scope

## 简介

```python
@tf_contextlib.contextmanager
tf.init_scope()
```

> **aliases**: `tf.compat.v1.init_scope`

从控制流 scope 和函数构建 graph 中提取操作的上下文管理器。

通常需要将变量初始化操作从控制流 scope、函数构建 graph 以及梯度 tape 中提取出来。`init_scope` 是满足这种需求的一种机制，进入 `init_scope` 有三个效果：

1. 进入 `init_scope` 时，所有的控制依赖被清除，