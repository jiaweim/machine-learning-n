# 聚类

- [聚类](#聚类)
  - [简介](#简介)
  - [构建聚类器](#构建聚类器)
    - [批聚类器](#批聚类器)
    - [增量聚类器](#增量聚类器)
  - [评估聚类器](#评估聚类器)
    - [classes to clusters](#classes-to-clusters)
  - [样本聚类](#样本聚类)

2023-12-12, 17:30
****

## 简介

聚类（clustering）是在数据中寻找模式的无监督机器学习技术，这类算法不需要类别信息。下面介绍如下内容：

- 构建聚类器：batch 和增量学习
- 评估聚类器
- 聚类新样本：确定新样本所属类别

## 构建聚类器

聚类器，和分类器一样，默认使用全部数据进行训练，即批训练。但是有少数聚类算法可以采用增量学习的方式更新内部表示。

### 批聚类器

构建批聚类器分两步：

- 设置选项：调用 `setOptions(String[])` 或 setter 方法配置选项
- 构建模型：调用 `buildClusterer(Instances)` 训练模型。根据定义，使用相同数据重复调用该方法，必须获得相同模型（可重复性）。换句话说，调用该方法必须首先完全重置模型。

**示例：** 构建 EM 聚类器（最多迭代 100 次）

```java
import weka.clusterers.EM;
import weka.core.Instances;
...
Instances data = ... // from somewhere
String[] options = new String[2];
options[0] = "-I"; // max. iterations
options[1] = "100";
EM clusterer = new EM(); // new instance of clusterer
clusterer.setOptions(options); // set the options
clusterer.buildClusterer(data); // build the clusterer
```

### 增量聚类器

增量聚类器实现 `UpdateableClusterer` 接口。训练增量聚类器分三步：

1. 初始化：调用 `buildClusterer(Instances)` 初始化模型。这里可以使用空的 `weka.core.Instances` 对象，或包含初始数据。
2. 更新：调用 `updateClusterer(Instance)` 逐个样本更新模型
3. 完成：调用 `updateFinished()` 完成模型。

**示例：** 使用 `ArffLoader` 迭代数据，增量构建 Cobweb 聚类器

```java
import weka.clusterers.Cobweb;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffLoader;
...
// 加载数据
ArffLoader loader = new ArffLoader();
loader.setFile(new File("/some/where/data.arff"));
Instances structure = loader.getStructure();
// train Cobweb
Cobweb cw = new Cobweb();
cw.buildClusterer(structure);
Instance current;
while ((current = loader.getNextInstance(structure)) != null)
    cw.updateClusterer(current);
cw.updateFinished();
```

## 评估聚类器

聚类的评估不如分类那么全面。由于聚类是无监督的，所以很难确定一个模型有多好。

由 `ClusterEvaluation` 类评估聚类。

**示例：** 为了生成和 Explorer 或命令行一样的输出，可以按如下方式使用 `evaluateClusterer` 方法：

```java
import weka.clusterers.EM;
import weka.clusterers.ClusterEvaluation;
...
String[] options = new String[2];
options[0] = "-t";
options[1] = "/some/where/somefile.arff";
System.out.println(ClusterEvaluation.evaluateClusterer(new EM(), options));
```

**示例：** 如果数据集已载入内存，可以使用如下方式

```java
import weka.clusterers.ClusterEvaluation;
import weka.clusterers.EM;
import weka.core.Instances;
...
Instances data = ... // from somewhere
EM cl = new EM();
cl.buildClusterer(data);
ClusterEvaluation eval = new ClusterEvaluation();
eval.setClusterer(cl);
eval.evaluateClusterer(new Instances(data));
System.out.println(eval.clusterResultsToString());
```

基于密度的聚类器，即实现 `DensityBasedClusterer` 接口的算法，可以交叉验证，获得 log-likelyhood。使用 `MakeDensityBasedClusterer` 可以将不是基于密度的聚类器转换为这类聚类器。

**示例：** 基于密度的聚类器的交叉验证，获得对数似然

```java
import weka.clusterers.ClusterEvaluation;
import weka.clusterers.DensityBasedClusterer;
import weka.core.Instances;
import java.util.Random;
...
Instances data = ... // from somewhere
DensityBasedClusterer clusterer = new ... // the clusterer to evaluate
double logLikelyhood =
ClusterEvaluation.crossValidateModel( // cross-validate
    clusterer, data, 10, // with 10 folds
    new Random(1)); // and random number generator
    // with seed 1
```

### classes to clusters

监督学习算法的数据集，也可以用来评估聚类算法。这种评估方式称为 classes-to-clusters，因为 clusterr 被映射回 classes。

这种方式的评估流程如下：

1. 创建包含 class 属性的数据集的副本，使用 Remove 过滤器删除 class 属性
2. 使用新的数据构建聚类器
3. 使用原始数据评估聚类器

具体代码：

1. 创建不包含 class 属性的数据副本

```java
Instances data = ... // from somewhere
Remove filter = new Remove();
filter.setAttributeIndices("" + (data.classIndex() + 1));
filter.setInputFormat(data);
Instances dataClusterer = Filter.useFilter(data, filter);
```

2. 构建聚类器

```java
EM clusterer = new EM();
// set further options for EM, if necessary...
clusterer.buildClusterer(dataClusterer);
```

3. 评估聚类器

```java
ClusterEvaluation eval = new ClusterEvaluation();
eval.setClusterer(clusterer);
eval.evaluateClusterer(data);
// print results
System.out.println(eval.clusterResultsToString());
```

## 样本聚类

使用如下样本判断新样本类别：

- `clusterInstance(Instance)`：返回 Instance 所属 cluster
- `distributionForInstance(Instance)`：返回 Instance 在不同 cluster 中的分布，加和为 1

**示例：** 在一个数据集上训练 EM，预测另一个数据集的 cluster

```java
import weka.clusterers.EM;
import weka.core.Instances;
...
Instances dataset1 = ... // from somewhere
Instances dataset2 = ... // from somewhere
// build clusterer
EM clusterer = new EM();
clusterer.buildClusterer(dataset1);
// output predictions
System.out.println("# - cluster - distribution");
for (int i = 0; i < dataset2.numInstances(); i++) {
    int cluster = clusterer.clusterInstance(dataset2.instance(i));
    double[] dist = clusterer.distributionForInstance(dataset2.instance(i));
    System.out.print((i+1));
    System.out.print(" - ");
    System.out.print(cluster);
    System.out.print(" - ");
    System.out.print(Utils.arrayToString(dist));
    System.out.println();
}
```