# Alink 快速入门

## 简介

随着大数据时代的到来和人工智能的崛起，机器学习所能处理的场景更加广泛和多样。算法工程师们不单要处理好批式数据的模型训练与预测，也要能处理好流式数据，并需要具备将模型嵌入企业应用和微服务上的能力。为了取得更好的业务效果，算法工程师们需要尝试更多、更复杂的模型，并需要处理更大的数据集，因此他们使用分布式集群已经成为常态。为了及时应对市场的变化，越来越多的业务选用在线学习方式来直接处理流式数据、实时更新模型。

Alink 就是为了更好地满足这些实际应用场景而研发的机器学习算法平台，以帮助数据分析和应用开发人员轻松地搭建端到端的业务流程。

## 1. Alink 是什么

Alink是阿里巴巴计算平台事业部机器学习平台团队基于Flink计算引擎研发的批流一体的机器学习算法平台，该平台提供了丰富的算法组件库和便捷的操作框架。借此，开发者可以一键搭建覆盖数据处理、特征工程、模型训练、模型预测的算法模型开发全流程。Alink的名称取自相关英文名称，即Alibaba、Algorithm、AI、Flink和Blink中的公共部分。Alink提供了Java接口和Python接口（PyAlink），开发者不需要Flink的技术背景也可以轻松构建算法模型。

Alink在2019年11月的Flink Forword Asia 2019大会上宣布**开源**。欢迎大家下载使用、反馈意见、提出建议，以及贡献新的算法。

GitHub地址：https://github.com/alibaba/Alink

Alink文档及教程：https://alinklab.cn/

## 2. 下载、安装

可以在Alink开源网站获取其最新版本。为了方便用户查看Alink文档，解决Alink的本地安装、使用问题，以及解决Alink在集群上部署、运行等方面的问题，我们提供了如下专门的资料供读者参考。相关的网址如下：

- 【Github开源】
  完整的开源内容：代码、函数说明、注意事项、安装包、历史版本。
  https://github.com/alibaba/Alink

- 【使用手册、教程】
  各种资料汇总在 [https://alinklab.cn](https://alinklab.cn/index.html)

关于Alink的安装、运行、部署等方面的内容，可以参阅：

- [第1.2.1节 使用 Maven 快速构建 Alink Java 项目](https://alinklab.cn/tutorial/book_java_01_2_1.html)
- [第1.2.2节 在集群上运行 Alink Java 任务](https://alinklab.cn/tutorial/book_java_01_2_2.html)

【本教程配套代码】
下载地址：https://github.com/ALinkLab/alink_tutorial_java

如果运行示例报错：

```java
Unable to make field private final byte[] java.lang.String.value accessible: module java.base does not "opens java.lang" to unnamed module @35f760a4
```

需要添加如下 VM 参数：

```
--add-opens java.base/java.lang=ALL-UNNAMED
```





