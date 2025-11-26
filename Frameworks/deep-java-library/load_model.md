# 加载模型

## 简介

模型是通过训练创建的 artifact 集合。在深度学习中，使用模型推理通常涉及预处理和后处理。djl 提供 `ZooModel` 类，可以轻松地将数据处理与模型集合起来。

下面介绍如何在各种场景中加载预训练模型。

## 使用 ModelZoo 加载 Model

djl 推荐使用 `ModelZoo` API 加载模型。

`ModelZoo` 提供加载模型的统一方式。该 API 声明式的性质允许将模型信息存储在配置文件中。这为测试和部署模型提供了很大的灵活性。

### Criteria 类

可以使用 `Criteria` 类缩小搜索条件来查找要加载的模型。`Criteria` 类遵循 djl 的 Builder 模式。以 `set` 开头的方法为必填字段，以 `opt` 开头的为可选字段，创建 `Criteria` 必需调用  `setType` 方法：

```java
```

