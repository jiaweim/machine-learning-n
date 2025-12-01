# Tribuo 包结构

2025-12-01⭐
@author Jiawei Mao
***

## 简介

核心模块提供 API定义、数据交互、数学库以及不同预测类型共享的通用模块。

- Core (`tribuo-core`, package-root `org.tribuo`) 提供主类和接口
  - `Dataset` - 包含 `Example` list 和相关特征信息，如分类特征的类别数，实值特征的均值和方差
  - `DataSource` - 包含从其它格式获得的 `Example` list，并附有数据来源和处理方式
  - `Example` - String-value tuple list。`Example` 类型与 `Output` 子类类型对应，代表响应类型
  - `Feature` - immutable tuple, `String` 和 value。`String` 为特征名称，作为 feature 的识别符
  - `FeatureMap` - 从 `String` 到 `VariableInfo` 的映射。当为 immutable 时，还包含 feature id 编号
  - `Model` - 能够预测特定 `Output` 类型的类
  - `Output` - 表示输出类型的接口：回归、多标签、多类、聚类或异常检测
  - `OutputInfo` - 表示与输出相关信息的接口
  - `Trainer` - 基于特定输出类型的 `Dataset` 生成 `Model` 的类
  - `Prediction` - 存储输入 `Example` 时 `Model` 输出的结果。包含每个 label 的打分。
  - `VariableInfo` - 表示特征信息的类，如它在数据集中出现的次数。
- Core 还包含一些其它 package
  - `dataset` - 提供对另一个数据集视图的 `Dataset`，可以执行 subsampling 或基于阈值的过滤
  - `datasource` - `DataSource` 的内存实现和简单文件格式实现
  - `ensemble` - 集成模型（ensemble）的基类和接口
  - `evaluation` - 所有输出类型评估基类。该 package 还包含与 evaluation 相关用于交叉验证和 train-test 拆分的类
  - `hash` - feature 的 hash 实现，混淆存储在 `Model` 中的特征名称。hash 可以防止特征名称泄漏数据集信息
  - `provenance` - provenance 记录数据集的位置和转换，trainer 的参数以及其它有用信息
  - `sequence` - 序列预测 API
  - `transform` - 一个特征转换 package，可应用于整个数据集，也可以通过正则表达式应用于匹配的单个特征。它还包含 trainer 和 model wrapper，以确保在预测时使用相同转换
  - `util` - 基本操作工具，如数据和随机样本
- Data (`tribuo-data`, package-root: `org.tribuo.data`) 提供数据处理、columnar 数据、csv 文件和文本输入。鼓励用户自行处理文本，因此这里的实现比较基础
  - `columnar` - 提供从 columnar 数据提取特征的类
  - `csv` - 在 columnar package 的基础上，提供用于处理 CSV 和其它分隔符文件的基础类
  - `sql` - 在 columnar package 的基础上，提供处理 JDBC 数据源的类
  - `text` - 文本处理基础接口和一个示例实现
- Json (`tribuo-json`, package-root: `org.tribuo.json`) 提供从 json 加载数据的功能，以及从模型剥离来源信息的功能
- Math (`tribuo-math`, package-root: `org.tribuo.math`) 提供线性代数库，用于稀疏、密集向量和矩阵
  - `kernel` - 一组用于 SGF 的 kernel 函数
  - `la` - 包含 SGD 中使用的线性代数库，不是完整的 BLAS
  - `optimisers` - 一套随机梯度下降算法，包括 SGD with momentum, AdaGra, AdaDelta, RMSProp 等。AdaGrad 应被视为默认算法，它在大多数线性 SGD 问题中性能最好
  - `util` - 用于数组、向量和矩阵的各种工具类

## Util 包



Tribuo 还有 3 个不依赖其它模块的库：

- InformationTheory - (`tribuo-util-infotheory`, package-root: `org.tribuo.util.infotheory`) 提供适用于计算聚类 metrices, 特征选择和结构选择的离散信息论函数
- ONNXExport - (`tribuo-util-onnx`, package-root: `org.tribuo.util.onnx`) 提供从 Java 构建 ONNX graph 的基础类。该 packge 适用于其它想要编写 ONNX 模型的JVM 库，相比直接输出 protobufs 提供了额外的类型安全和可用性
- Tokenization - (`tribuo-util-tokenization`, package-root: `org.tribuo.util.tokens`) 提供适合特征提取和信息检索的分词 API（tokenization），以及多个 tokenizer 实现，包括适合与 BERT 模型一起使用的 wordpiece 实现

## Multi-class Classification

Multi-class 分类为测试样本分配多个 labels 中的一个。分类模块包含多个子模块：

| Folder                  | ArtifactID                           | Package root                             | Description                                                  |
| ----------------------- | ------------------------------------ | ---------------------------------------- | ------------------------------------------------------------ |
| Core                    | `tribuo-classification-core`         | `org.tribuo.classification`              | 包含多类分类任务的 `Output` 子类、检查模型性能的 evaluation 代码以及 `Adaboost.SAMME` 的实现。还包含简单的 baseline classifiers |
| DecisionTree            | `tribuo-classification-tree`         | `org.tribuo.classification.dtree`        | CART 决策树实现                                              |
| Experiments             | `tribuo-classification-experiments`  | `org.tribuo.classification.experiments`  | 一组用于任何支持的数据集上训练和测试模型的 main 函数。该子模块依赖于所有 classifiers，便于在它们之间进行轻松比较。**不应**将其导入其它项目，它纯粹用于开发和测试 |
| Explanations            | `tribuo-classification-experiments`  | `org.tribuo.classification.explanations` | 用于分类任务的 LIME 实现。如果使用 columnar 数据 loader，LIME 可以提取优选特征 domain 的更多信息，并提供更好的解释 |
| FeatureSelection        | `tribuo-classification-fs`           | `org.tribuo.classification.fs`           | 用于分类问题的几种信息论特征选择算法的实现                   |
| LibLinear               | `tribuo-classification-liblinear`    | `org.tribuo.classification.liblinear`    | LibLinear-java 的 wrapper。提供 linear-SVMs 和其它 l1 或 l2 正则化线性 classifiers |
| LibSVM                  | `tribuo-classification-libsvm`       | `org.tribuo.classification.libsvm`       | LibSVM 的 wrapper。提供 sigmoid, gaussian 和 polynominal kernel 的 linear & kernel SVMs |
| Multinomial Naive Bayes | `tribuo-classification-mnnaivebayes` | `org.tribuo.classification.mnb`          | 多项式朴素贝叶斯 classifier 的实现。其目标是存储模型的简洁内存表示，因此只记录观察到的 feature/class pairs 的权重 |
| SGD                     | `tribuo-classification-sgd`          | `org.tribuo.classification.sgd`          | 基于随机梯度下降（SGD）的 classifier 实现。包含用于逻辑回归和线性 SVM（分别使用 log 和 hinge loss）的 linear package，一个使用 Pegasos 算法训练 kernel-SVM 的 kernel package，一个用于训练 linear-chain CRF 的 crf package，以及一个用于训练 pairwise factorization machines 的 fm package。这些实现依赖于 Main 包中的 SGD。linear, fm, crf packages 可使用提供的任何梯度 optimisers，这些 optimisers 强制执行各种类型的正则化和收敛指标。由于 SGD 方法的速度和可扩展性，这是线性分类和序列分类的首选 package |
| XGBoost                 | `tribuo-classification-xgboost`      | `org.tribuo.classification.xgboost`      | XGBoost API 的 wrapper。XGBoost 需要通过 JNI 防伪 C 库。XGBoost 是梯度提升树的可扩展实现 |

## Multi-label Classification

Multi-label 分类对测试样本输出多个标签，而不是单个标签。

独立的二元 predictor 将每个 multi-label 预测拆分为 n 个二元 predictions，每个可能的 label 对应一个。为此，提供的 trainer 采用一个 classification trainer 构建 n 个模型，每个 label 一个，然后依次在测试样本上运行，生成最终的 multi-label 输出。classifier chain 使用类似方法将 classification-trainer 转换为 multi-label trainere。

| Folder | ArtifactID               | Package root                | Description                                                  |
| ------ | ------------------------ | --------------------------- | ------------------------------------------------------------ |
| Core   | `tribuo-multilabel-core` | `org.tribuo.multilabel`     | 包含用于 multi-label 的 `Output` 子类、检查 multi-label 模型性能的 evaluation 代码，以及独立二元预测的基本实现。还包含 Classifier chains 和 classifier chain Ensemble 的实现，是用于 multi-label 预测的更强大的集成技术 |
| SGD    | `tribuo-multilabel-sgd`  | `org.tribuo.multilabel.sgd` | 基于 SGD 的 classifier 实现。包含用于独立逻辑回归和 linear-SVM 的 linear package，以及对每个输出 label 的损失分解机。这些实现依赖于 Math 包中的 SGD。linear 和 fm 包可使用任何梯度 optimisers，它们强制执行各种类型的正则化和收敛指标 |



## Regression

回归是预测实值输出的任务。提供的模块：

| Folder          | ArtifactID                    | Package root                      | Description                                                  |
| --------------- | ----------------------------- | --------------------------------- | ------------------------------------------------------------ |
| Core            | `tribuo-regression-core`      | `org.tribuo.regression`           | 包含用于回归数据的 `Output` 子类，以及用于评估模型性能的标准回归指标（如 $R^2$，explained variance, RMSE 和平均绝对误差）。该模块还包含一个简单的 baseline 回归 |
| LibLinear       | `tribuo-regression-liblinear` | `org.tribuo.regression.liblinear` | LibLinear-java 的 wrapper。提供 linear-SVM 和其它 l1 或 l2 正则化线性回归 |
| LibSVM          | `tribuo-regression-libsvm`    | `org.tribuo.regression.libsvm`    | LibSVM 的 wrapper。提供具有 sigmoid, gaussian 和 polynomial kernels 的 linear & kernel SVR |
| RegressionTrees | `tribuo-regression-tree`      | `org.tribuo.regression.rtree`     | 两种类型的 CART 回归树实现。第一种为每个输出维度构建一个单独的 tree，第二种为所有输出构建一个 tree |
| SGD             | `tribuo-regression-sgd`       | `org.tribuo.regression.sgd`       | 为用于线性回归和 factorization machine 回归实现的随机梯度下降。它使用 Math package 中的梯度优化器，这些优化器允许各种正则化和下降算法 |
| SLM             | `tribuo-regression-slm`       | `org.tribuo.regression.slm`       | 稀疏线性模型实现。包含 ElasticNet 的坐标下降实现，LARS 实现，使用 LARSS 的 LASSO 实现，以及几种顺序前向选择算法 |
| XGBoost         | `tribuo-regression-xgboost`   | `org.tribuo.regression.xgboost`   | XGBoost Java API 的 wrapper。XGBoost 需要通过 JNI 访问 C 库  |

## Clutering

聚类是对输出数据分组的任务。

| Folder  | ArtifactID                  | Package root                    | Description                                                  |
| ------- | --------------------------- | ------------------------------- | ------------------------------------------------------------ |
| Core    | `tribuo-clustering-core`    | `org.tribuo.clustering`         | 包含用于聚类的 `Output` 子类，以及用于衡量聚类性能的评估代码 |
| HDBSCAN | `tribuo-clustering-hdbscan` | `org.tribuo.clustering.hdbscan` | HDBSCAN 的实现，这是一种基于非参密度的聚类算法               |
| KMeans  | `tribuo-clustering-kmeans`  | `org.tribuo.clustering.kmeans`  | 使用 Java 8 Stream API 并行化的 K-Means 实现，以及 K-Means++ 初始化算法 |

## Anomaly Detection

异常检测（anomaly detection）使用非异常数据训练模型，用该模型发现异常值。

| Folder    | ArtifactID                 | Package root                   | Description                                       |
| --------- | -------------------------- | ------------------------------ | ------------------------------------------------- |
| Core      | `tribuo-anomaly-core`      | `org.tribuo.anomaly`           | 包含异常检测的 `Output` 子类                      |
| LibLinear | `tribuo-anomaly-liblinear` | `org.tribuo.anomaly.liblinear` | LibLinear-Java 的 wrapper，提供一个 one-class SVM |
| LibSVM    | `tribuo-anomaly-libsvm`    | `org.tribuo.anomaly.libsvm`    | LibSVM 的 wrapper，提供一个 one-class SVM         |

## Common

该模块包含在多种预测类型间共享的代码。它为 LibLinear, LibSVM, nearest-neighbor, tree 和 XGBoost 模型提供基础支持。nearest-neighbor 子模块是独立的，但其余模块需要专门用于预测的实现模块。common tree package 包含随机森林和极端随机树（ExtraTrees）的实现。

## Third party models

Tribuo 支持加载多个第三方模型，这些模型在系统外部（甚至其它编程语言）中训练，然后使用 Tribuo 从 Java 使用。目前支持加载 ONNX、TensorFlow 和 XGBoost 模型。此外，还支持将 [OCI 数据科学模型](https://www.oracle.com/artificial-intelligence/data-science/)部署在 Tribuo模型。

- OCI - 支持将 Tribuo 模型部署到 OCI 数据科学，以及将 OCI 数据科学模型包装在 Tribuo 外部模型，允许它们与其它 Tribuo 模型一起使用
- ONNX - ONNX (Open Neural Network eXchange) 格式被许多深度学习系统用作导出格式，而且有从 scikit-learn 等系统转换为 ONNX 格式的转换器。Tribuo 提供了一个 Microsoft ONNX runtime 包装，可以对 CPU 和 GPU 平台的 ONNX 模型进行打分。ONNX 支持在 `tribuo-onnx` 的 `org.tribuo.interop.onnx` package 中，还提供使用 BERT 嵌入模型的特征提取器。该 package 还包含加载 Tribuo 导出的 ONNX 模型，并从这些模型中提取存储的 Tribuo 来源信息
- TensorFlow - Tribuo 支持加载 TensorFlow 的 frozen graphs 和 saved model
- XGBoost - Tribuo 支持加载 XGBoost 分类和回归模型

## TensorFlow

Tribuo 在 `tribuo-tensorflow` artifact 的 `org.tribuo.interop.tensorflow` package 中包含对 TensorFlow-Java 0.4.0 (对应 TensorFlow 2.7.0)的实验性支持。可以使用 TensorFlow-Java 的 graph 构造机制定义模型，Tribuo 会管理梯度 optimizer 输出函数和损失函数。它包含 Java 序列化系统，因此所有 TensorFlow 模型能和其它 Tribuo 模型一样进行序列化和反序列化。如果有 GPU 且在 classpath 有相应的 GPU jar，则 TensorFlow 默认在 GPU 上运行。

在 TF JVM SIG 重写 TensorFlow Java API 期间，该支持依然出于实验阶段。Tribuo 团队参与了 TensorFlow JVM SIG，并致力于改进 TensorFlow，不仅是为了 Tribuo，也是为了整个 Java 社区。

Tribuo 包含了一个示例配置文件、几个实验模型生成函数和 MNIST mode graph 的 protobuf 来演示 TensorFlow 的互操行。

## 其它模块

Tribuo 还包含一些其它模块：

| Folder          | ArtifactID                 | Package root                   | Description                                                  |
| --------------- | -------------------------- | ------------------------------ | ------------------------------------------------------------ |
| Json            | `tribuo-json`              | `org.tribuo.json`              | 支持读写 Json 格式数据，以及用于检查和删除模型来源信息的工具 |
| ModelCard       | `tribuo-interop-modelcard` | `org.tribuo.interop.modelcard` | 支持读写 Json 格式的 model card，使用 Tribuo 模型中的来源信息指导 card 的构建 |
| Reproducibility | `tribuo-reproducibility`   | `org.tribuo.reproducibility`   | 用于重现 Tribuo 模型和数据集的工具                           |
