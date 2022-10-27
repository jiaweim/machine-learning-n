# TensorFlow QAs

## 证书问题

在下载 tensorflow.keras 数据集时遇到证书问题下载失败，可以添加如下语句解决该问题：

```python
import ssl

ssl._create_default_https_context = ssl._create_unverified_context
```

## Symbolic

TF2 提供了两种构架神经网络的 API：

1. 符号样式（symbolic），通过 graph of layers 构建神经网络；
2. 命令样式（imperative），通过扩展 class 构建神经网络。

符号样式的 graph 可以是 DAG（有向无环图），也可以是 layer 堆栈。当以符号样式构建模型，其实就是在描述这个 graph 的结构。

TF2.0 提供了两套符号模型构建 API：keras Sequential 以及 keras Functional。Sequential 用于 layer 堆栈，Functional 用于 DAG。

显然，Functional API 更灵活，可以创建非线性模型、具有共享层的模型、包含多个输入或输出的模型。
