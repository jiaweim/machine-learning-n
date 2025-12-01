# Tribuo 结构

## 简介

Tribuo 是一个用于创建机器学习（ML）模型的库。

ML 模型是对数据集应用某个训练算法的结果。Tribuo `Model` 可以看作稀疏特征空间到密集（dense）输出空间（如分类标签概率、回归输出）。每个输出和输出都有命名。

## 数据流

![Tribuo architecture diagram](./images/tribuo-data-flow.png)

Tribuo 使用 `DataSource` 实现加载数据，数据来源可能是数据库或文件。`DataSource` 处理输入数据，将其转换为 Tribuo 的存储格式 `Example`。`Example` 是包含输出 `Output` 的特征列表的 tuple，每个 `Feature` 包含 `String` 类型名称和 `double` 类型的特征值。

`DataSource` 被读入 `Dataset`，`Dataset` 会统计数据信息用于建模。 `Dataset` 可以拆分为训练集和测试集，或者根据某些条件过滤数据。`Example` 输入 `Dataset` 后，会记录特征，特征的统计信息保存在 `FeatureMap` 中。类似地，输出的统计信息被记录在 `Output` 子类的 `OutputInfo` 中。

准备好 `Dataset`，将其传递给 `Trainer`，`Trainer` 包含训练算法和必要的参数值（超参数），`Trainer` 训练完成后生成 `Model`。`Model` 包含预测所需的参数，以及一个记录模型构建过程的 `Provenance` 对象（如数据文件名、数据 hash、训练超参数、时间戳等）。数据和模型都可以通过 java 序列化保存到磁盘。

模型训练完成，就可以输入之前未见过的样本，生成 `Prediction`。如果已知新的样本的输出，则该 `Prediction` 可以传递给 `Evaluator`，用于计算准确性之类的统计数据。

## 结构

Tribuo 包含的顶层模块：

- tribuo-core 包含 Tribuo 的核心类和接口
- tribuo-data 提供文本、sql 和 csv 数据的加载器，以及处理 column 数据的工具
- tribuo-math 提供 tribuo 的线性代数库，以及 kernel 和梯度 optimizers
- tribuo-json 提供一个JSON 数据加载器，以及一个从训练模型剥离 provenance 的工具

Tribuo 对每个预测任务提供了单独的模块：

- tribuo-classification-core: 包含一个名为 `Label` 的 `Output` 实现，表示多类分类。`Label` 是 `String` 名称和 `double` 精度的元组。对每个 `OutputFactory`, `OutputInfo`, `Evaluator` 和 `Evaluation`，都提供特定于分类的实现，分别为 `LabelFactory`, `LabelInfo`, `LabelEvaluator` 和 `LabelEvaluation`
- tribuo-regression-core: 包含一个名为 `Regressor` 的 `Output` 实现，表示多维回归。每个 `Regressor` 是一个元组，包含维度名称、`double` 维度值，`double` 维度方差。对每个 `OutputFactory`, `OutputInfo`, `Evaluator` 和 `Evaluation`，都提供特定于回归的实现，分别为 `RegressionFactory`, `RegressionInfo`, `RegressionEvaluator` 和 `RegressionEvaluation`。维度名称默认为 "DIM-x"，其中 `x` 是非负整数
- tribuo-anomaly-core: 包含一个名为 `Event` 的 `Output` 实现，表示检测到的事件类型 `EventType` 符合预期 `EXPECTED` 或为异常 `ANOMALY`。`Event` 为元组类型，包含`EventType` 实例和代表事件类型 `double` 精度的打分值。对每个 `OutputFactory`, `OutputInfo`, `Evaluator` 和 `Evaluation`，都提供特定于异常检测的实现，分别为 `AnomalyFactory`, `AnomalyInfo`, `AnomalyEvaluator` 和 `AnomalyEvaluation`
- tribuo-clustering-core: `Output` 实现类为 `ClusterID`，代表分配的 cluster id。每个 `ClusterID` 包含一个非负整数 id，以及 `double` 精度的打分值。对 `OutputFactory`, `OutputInfo`, `Evaluator` 和 `Evaluation`，都提供特定于聚类的实现，分别为 `ClusteringFactory`, `ClusteringInfo`, `ClusteringEvaluator` 和 `ClusteringEvaluation`
- tribuo-multilabel-core:  `Output` 实现类为 `MultiLabel`，代表多类分类。每个 `MultiLabel` 包含一组 `Label` 实例以及对应打分。对 `OutputFactory`, `OutputInfo`, `Evaluator` 和 `Evaluation`，都提供特定于聚类的实现，分别为 `MultiLabelFactory`, `MultiLabelInfo`, `MultiLabelEvaluator` 和 `MultiLabelEvaluation`。它还提供了一个 `Trainer<MultiLabel>`，它接收 `Trainer<Label>` 通过使用内存 trainer 对每个 `Label` 单独进行预测生成 `Model<MultiLabel>`。这个多类分类问题的合理 baseline。

最后，还有一些跨领域模块集合：

- 为预测任务提供的基础类
- tribuo-interop-core: 处理大型外部库的基础类，如 TensorFlow 和 ONNX Runtime
- 用于特定任务的独立库。例如 InformationTheory 是一个用于信息理论的函数库，Tokens 提供 tokenization 相关的接口和实现