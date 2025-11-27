# 内存管理

## 简介

内存是深度学习领域，尤其是 Java 领域最大挑战之一。最大的问题是垃圾回收器无法回收本地内存。它不知道用了多少，也不知道怎么释放。除此之外，对高内存使用率（如 GPU 上训练）它太慢了。

如果没有自动内存管理，仍然可以手动释放内存。当时深度学习会创建很多 `NDArray`，手动释放不实用。

因此，DJL 使用 [NDManager](https://javadoc.io/doc/ai.djl/api/latest/ai/djl/ndarray/NDManager.html)。NDManager 即是 factory，又是 scope。所有由 manager 创建的数组都属于其 scope。通过释放 manager，它可以释放它创建的所有数组。这也包括通过 `NDArray.add(other)` 或 `Activation.relu(array)` 操作创建的数组。它们附加到输入参数所属的 manager。如果操作包含多个 `NDArray` 输入，则附加到第一个参数所属 manager。

NDManager 的分层结构也很常见，它们代表不同的抽象层次和数据释放节点。下面看一些常见用法。

## 推理示例

典型推理情形 NDManager 的结构：

<img src="./images/ndmanager_structure_for_inference.png" width="120" />

结构的顶层是系统 manager。



## 参考

- https://docs.djl.ai/master/docs/development/memory_management.html