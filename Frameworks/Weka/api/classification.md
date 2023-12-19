# 分类

- [分类](#分类)
  - [简介](#简介)
  - [构建分类器](#构建分类器)
    - [批分类器](#批分类器)
    - [增量分类器](#增量分类器)
  - [评估分类器](#评估分类器)
    - [交叉验证](#交叉验证)
    - [测试集](#测试集)
    - [统计量](#统计量)
    - [收集预测结果](#收集预测结果)
  - [预测](#预测)

2023-12-12, 15:53
****

## 简介

分类和回归算法在 WEKA 中统称为分类器，相关类在 `weka.classifiers` 包中。下面介绍：

- 构建分类器：批量和增量学习
- 评估分类器：各种评估技术以及如何获得统计数据
- 执行分类：预测未知数据的类别

## 构建分类器

weka 的所有分类器都可以批训练，即一次在整个数据集上训练。如果内存能够容纳训练集，该方式很好；如果数据量太大，也有部分分类器支持增量训练。

### 批分类器

构建批分类器非常简单：

- 设置选项：使用 `setOptions(String[])` 方法或 setter 方法
- 训练：在训练集上调用 `buildClassifier(Instances)`
  - 根据定义，`buildClassifier(Instances)` 应该完全重置模型，以确保下次使用相同数据训练能得到相同模型

**示例：** 训练一个 J48 模型

```java
Instances data = ... // from somewhere
String[] options = new String[1];
options[0] = "-U"; // unpruned tree
J48 tree = new J48(); // new instance of tree
tree.setOptions(options); // set the options
tree.buildClassifier(data); // build classifier
```

### 增量分类器

实现 `UpdateableClassifier` 的分类器可以增量训练。由于不需要一次性将所有数据载入内容，这类分类器内存占用小。

训练增量分类器分两步：

1. 初始化：调用 `buildClassifier(Instances)` 初始化模型，`weka.core.Instances` 可以不包含或包含初始数据，需要通过 `Instances` 获取数据结构
2. 更新：调用 `updateClassifier(Instance)` 逐个使用样本更新模型

**示例：** 使用 `ArffLoader` 迭代 ARFF 文件数据，逐个使用样本训练 `NaiveBayesUpdateable` 分类器

```java
// 加载数据
ArffLoader loader = new ArffLoader();
loader.setFile(new File("/some/where/data.arff"));
Instances structure = loader.getStructure();
structure.setClassIndex(structure.numAttributes() - 1);

// 训练 NaiveBayes
NaiveBayesUpdateable nb = new NaiveBayesUpdateable();
nb.buildClassifier(structure);
Instance current;
while ((current = loader.getNextInstance(structure)) != null)
    nb.updateClassifier(current);
```

## 评估分类器

weka 支持两种评估方法：

- 交叉验证（cross-validation）：只有训练集没有测试集
- 专门的测试集：测试集仅用于评估分类器。

评估步骤，包括统计信息的手机，由 `Evaluation` 类执行。

### 交叉验证

`Evaluation` 类的 `crossValidateModel` 方法在单个数据集上对**未训练**的分类器执行交叉验证。

在执行交叉验证前，需使用随机数生成器 `java.util.Random` 随机化数据。建议指定 seed 创建 `Random`。否则，由于数据的随机化不同，后续在同一数据集上的交叉验证会产生不同的结果。

**示例**：使用 J48 决策树算法在数据集 `newData` 上执行 10-fold 交叉验证。随机数生成器的 seed 设为 1。

```java
Instances newData = ... // from somewhere
Evaluation eval = new Evaluation(newData);
J48 tree = new J48();
eval.crossValidateModel(tree, newData, 10, new Random(1));
System.out.println(eval.toSummaryString("\nResults\n\n", false));
```

这里使用评估过程中使用的数据集初始化 `Evaluation`，这样是为了正确设置数据结构。

!!! note
    传递给 `crossValidateModel` 方法的模型必须**没有经过训练**。因为，如果分类器不遵守 weka 规则，在 `buildClassifier` 时没有重置分类器，交叉验证会得到不一致的结果。`crossValidateModel` 在训练和评估分类器时，会创建原分类器的副本，因为不存在该问题。

### 测试集

使用专门的测试集评估模型。此时，需要提供训练过的分类器，而不是未经训练的分类器。

依然使用 `weka.classifiers.Evaluation` 类，调用 `evaluateModel` 方法。

**示例**：在训练集上使用默认配置训练 J48，然后再测试集上评估模型。

```java
Instances train = ... // from somewhere
Instances test = ... // from somewhere
// 训练模型
Classifier cls = new J48();
cls.buildClassifier(train);
// evaluate classifier and print some statistics
Evaluation eval = new Evaluation(train);
eval.evaluateModel(cls, test);
System.out.println(eval.toSummaryString("\nResults\n\n", false));
```

### 统计量

除了前面使用的 `Evaluation` 类的 `toSummaryString` 方法，还有其它输出统计结果的方法。

**nominal class 属性统计方法**

- `toMatrixString`：输出混淆矩阵
- `toClassDetailsString`：输出 TP/FP rates, precision, recall, F-measure, AUC
- `toCumulativeMarginDistributionString`：输出累计 margins distribution

如果不想使用这些汇总方法，也可以访问单个统计量。

**nominal class 属性**

- `correct()`：正确分类样本数。`incorrect()` 返回错误分类样本数。
- `pctCorrect()`：正确分类样本百分比；`pctIncorrect()` 返回错误分类样本百分比。
- `areaUnderROC(int)`：指定 class label index 的 AUC
- `kappa()`：Kappa 统计量

**numeric class 属性**

- `correlationCoefficient()`：相关稀疏

**常规**

- `meanAbsoluteError()`：平均绝对误差
- `rootMeanSquaredError()`：均方根误差
- `numInstances()`：指定类别的样本数
- `unclassified()`：未分类样本数
- `pctUnclassified()`：未分类样本的百分比

完整信息，可参考 `Evaluation` 类的 javadoc。

**示例：** 输出和命令行中相似的统计结果

```java
String[] options = new String[2];
options[0] = "-t";
options[1] = "/some/where/somefile.arff";
System.out.println(Evaluation.evaluateModel(new J48(), options));
```

### 收集预测结果

在评估分类器时，可以提供一个对象来打印预测结果。该模式的超类是 `weka.classifiers.evaluation.output.prediction.AbstractOutput`。

**示例：** 将 10-fold 交叉验证的结果保存到 CSV 文件

```java
// 加载数据
Instances data = DataSource.read("/some/where/file.arff");
data.setClassIndex(data.numAttributes() - 1);

// 配置分类器
J48 cls = new J48();

// 10-fold 交叉验证，预测结果保存到 CSV
Evaluation eval = new Evaluation(data);
StringBuffer buffer = new StringBuffer();
CSV csv = new CSV();
csv.setBuffer(buffer);
csv.setNumDecimals(8); // use 8 decimals instead of default 6
// 如果希望保存到文件，取消下面注释
//csv.setOutputFile(new java.io.File("/some/where.csv"));
eval.crossValidateModel(cls, data, 10, new Random(1), csv);
// output collected predictions
System.out.println(buffer.toString());
```

**示例：** 将预测结果保存到 Java 对象，采用 `InMemory` 类

```java
// 加载数据
Instances data = DataSource.read("/some/where/file.arff");
data.setClassIndex(data.numAttributes() - 1);

// 配置分类器
J48 cls = new J48();

// 10-fold 交叉验证
Evaluation eval = new Evaluation(data);
StringBuffer buffer = new StringBuffer();
InMemory store = new InMemory();
// additional attributes to store as well (eg ID attribute to identify instances)
store.setAttributes("1");
eval.crossValidateModel(cls, data, 10, new Random(1), store);
// output collected predictions
int i = 0;
for (PredictionContainer cont: store.getPredictions()) {
    i++;
    System.out.println("\nContainer #" + i);
    System.out.println("- instance:\n" + cont.instance);
    System.out.println("- prediction:\n" + cont.prediction);
}
```

## 预测

在评估分类器后证明其有用，就可以用来预测新数据的类别。

**示例：** 使用训练过的分类器 tree 对从磁盘加载的样本进行分类。分类完成后，将其写入新的文件。

加载文件 `/some/where/unlabeled.arff`，使用 tree 分类器预测样本类别，将标记数据输出到 `/some/where/labeled.arff` 文件。

```java
// 加载未标记数据
Instances unlabeled = DataSource.read("/some/where/unlabeled.arff");
// 设置 class 属性
unlabeled.setClassIndex(unlabeled.numAttributes() - 1);

// 复制数据集
Instances labeled = new Instances(unlabeled);

// 预测样本
for (int i = 0; i < unlabeled.numInstances(); i++) {
    double clsLabel = tree.classifyInstance(unlabeled.instance(i));
    labeled.instance(i).setClassValue(clsLabel);
}

// 保存标记数据
DataSink.write("/some/where/labeled.arff", labeled);
```

说明：

- `classifyInstance(Instance)` 返回 0-based 索引，如果想获得获得索引对应的标签，可以采用

```java
System.out.println(clsLabel + " -> " + unlabeled.classAttribute().value((int) clsLabel));
```

- 如果对类别分布感兴趣，可以使用 `distributionForInstance(Instance)`，返回样本属于每个分类的概率，该方法仅对分类任务有用。

**示例：** 输出分类分布

```java
// 加载训练集
Instances train = DataSource.read(args[0]);
train.setClassIndex(train.numAttributes() - 1);
// 加载测试集
Instances test = DataSource.read(args[1]);
test.setClassIndex(test.numAttributes() - 1);
// 训练分类器
J48 cls = new J48();
cls.buildClassifier(train);
// 输出预测结果
System.out.println("# - actual - predicted - distribution");
for (int i = 0; i < test.numInstances(); i++) {
    double pred = cls.classifyInstance(test.instance(i));
    double[] dist = cls.distributionForInstance(test.instance(i));
    System.out.print((i+1) + " - ");
    System.out.print(test.instance(i).toString(test.classIndex()) + " - ");
    System.out.print(test.classAttribute().value((int) pred) + " - ");
    System.out.println(Utils.arrayToString(dist));
}
```

