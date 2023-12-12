# 分类

- [分类](#分类)
  - [简介](#简介)
  - [构建 classifier](#构建-classifier)
  - [评估分类器](#评估分类器)
    - [交叉验证](#交叉验证)
    - [测试集](#测试集)
    - [统计值](#统计值)
    - [收集预测结果](#收集预测结果)
  - [预测](#预测)

2023-12-12, 15:53
****

## 简介

分类和回归算法在 WEKA 中称为 classifier，放在 weka.classifiers 包中。下面介绍：

- 构建 classifier - 批量和增量学习
- 评估 classifier - 各种评估技术以及如何获得生成的统计数据
- 执行分类 - 预测位置数据的类别

## 构建 classifier

weka 的所有 classifiers 都可以 batch 训练，即一次在整个数据集上训练。如果内存能够容纳训练集，该方式很好；如果数据量太大，weka 也提供了增量（incremental）classifier。

**Batch classifiers**

构建 batch classifier 非常简单：

- 配置选项：使用 `setOptions(String[])` 方法或 setter 方法
- 训练：在训练集上调用 `buildClassifier(Instances)`。根据定义，`buildClassifier(Instances)` 会完全重置模型，以确保下次使用相同数据训练能得到相同模型

下面训练一个 unpruned J48 模型：

```java
import weka.core.Instances;
import weka.classifiers.trees.J48;
...
Instances data = ... // from somewhere
String[] options = new String[1];
options[0] = "-U"; // unpruned tree
J48 tree = new J48(); // new instance of tree
tree.setOptions(options); // set the options
tree.buildClassifier(data); // build classifier
```

**Incremental classifiers**

weka 的所有 incremental classifiers 实现 `UpdateableClassifier` 接口。这类 classifier 能以很小的内存占用在大量数据上训练。

训练增量分类器分两步：

1. 初始化：调用 `buildClassifier(Instances)` 初始化模型，可以不包含数据或包含初始数据的 `weka.core.Instances` 对象
2. 更新：调用 `updateClassifier(Instance)` 逐个使用样本更新模型

**示例：** 使用 `ArffLoader` 类以增量方式加载 ARFF 文件，并逐个使用样本训练 `NaiveBayesUpdateable` 分类器

```java
import weka.core.converters.ArffLoader;
import weka.classifiers.bayes.NaiveBayesUpdateable;
import java.io.File;
...
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

- 交叉验证（cross-validation）：只有一个数据集的情况。
- 专门的测试集：测试集仅用于评估分类器。

评估步骤，包括统计信息的手机，由 `Evaluation` 类执行。

### 交叉验证

`Evaluation` 类的 `crossValidateModel` 方法在单个数据集上对**未训练**的分类器执行交叉验证：

- 使用未训练的分类器可确保没有信息泄露到评估过程。尽管要求 `buildClassifier` 实现重置分类器，但不能确保任何实现都遵守该要求。
- 使用未训练分类器避免了不必要的副作用，因为对每个 train/test 对，使用的原始分类器副本

在执行交叉验证前，需使用提供的随机数生成器 `java.util.Random` 随机化数据。建议使用指定的 seed 值创建 `Random`。否则，由于数据的随机化不同，后续在同一数据集上的交叉验证会产生不同的结果。

**示例**：使用 J48 决策树算法在数据集 `newData` 上执行 10-fold 交叉验证。随机数生成器的 seed 设为 1。

```java
import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import java.util.Random;
...
Instances newData = ... // from somewhere
Evaluation eval = new Evaluation(newData);
J48 tree = new J48();
eval.crossValidateModel(tree, newData, 10, new Random(1));
System.out.println(eval.toSummaryString("\nResults\n\n", false));
```

这里使用评估过程中使用的数据集初始化 `Evaluation`，这样是为了正确设置数据结构。

### 测试集

使用专门的测试集评估模型，其使用方式与交叉验证一样简单。

此时，必须提供经过训练的分类器，而不是提供一个未经训练的分类器。

依然使用 `weka.classifiers.Evaluation` 类，调用 `evaluateModel` 方法。

**示例**：在训练集上使用默认配置训练 J48，然后再测试集上评估模型。

```java
import weka.core.Instances;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
...
Instances train = ... // from somewhere
Instances test = ... // from somewhere
// train classifier
Classifier cls = new J48();
cls.buildClassifier(train);
// evaluate classifier and print some statistics
Evaluation eval = new Evaluation(train);
eval.evaluateModel(cls, test);
System.out.println(eval.toSummaryString("\nResults\n\n", false));
```

### 统计值

前面评估示例中均使用的 Evaluation 类的 toSummaryString 方法。对 nominal class 属性还有其它统计方法：

- `toMatrixString`：输出混淆矩阵
- `toClassDetailsString`：输出 TP/FP rates, precision, recall, F-measure, AUC
- `toCumulativeMarginDistributionString`：输出累计 margins distribution

如果不想使用这些汇总方法，也可以访问单个统计量。

**nominal class 属性**

- `correct()`：正确分类样本数。`incorrect()` 返回错误分类样本数。
- `pctCorrect()`：正确分类样本百分比；`pctIncorrect()` 返回错误分类样本百分比。
- `areaUnderROC(int)`：指定 class label index 的 AUC

**numeric class 属性**

- `correlationCoefficient()`：相关稀疏

**常规**

- `meanAbsoluteError()`：平均绝对误差
- `rootMeanSquaredError()`：均方根误差
- `numInstances()`：指定类别的样本数
- `unclassified()`：未分类样本数
- `pctUnclassified()`：未分类样本的百分比

完整信息，可参考 `Evaluation` 类的 javadoc。

### 收集预测结果

在评估分类器时，可以提供一个对象来打印预测结果。该模式的超类是 `weka.classifiers.evaluation.output.prediction.AbstractOutput`。

**示例：** 将 10-fold 交叉验证的结果保存到 CSV 文件

```java
import weka.classifiers.Evaluation;
import weka.classifiers.evaluation.output.prediction.CSV;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
...
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
import weka.classifiers.Evaluation;
import weka.classifiers.evaluation.output.prediction.InMemory;
import weka.classifiers.evaluation.output.prediction.InMemory.PredictionContainer;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
...
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

在评估分类器后证明其有用，可以使用构建的分类器预测新数据的类别。

**示例：** 使用训练过的分类器 tree 对从磁盘加载的样本进行分类。分类完成后，将其写入新的文件。

```java
// load unlabeled data and set class attribute
Instances unlabeled = DataSource.read("/some/where/unlabeled.arff");
unlabeled.setClassIndex(unlabeled.numAttributes() - 1);
// create copy
Instances labeled = new Instances(unlabeled);
// label instances
for (int i = 0; i < unlabeled.numInstances(); i++) {
double clsLabel = tree.classifyInstance(unlabeled.instance(i));
labeled.instance(i).setClassValue(clsLabel);
}
// save newly labeled data
DataSink.write("/some/where/labeled.arff", labeled);
```

该示例适用于分类和回归任务。因为 `classifyInstance(Instance)` 对回归返回数字，对分类返回 0-based 索引。

如果对类别分布感兴趣，则可以使用 `distributionForInstance(Instance)`。该方法仅对分类任务有用。

**示例：** 输出分类分布

```java
// load data
Instances train = DataSource.read(args[0]);
train.setClassIndex(train.numAttributes() - 1);
Instances test = DataSource.read(args[1]);
test.setClassIndex(test.numAttributes() - 1);
// train classifier
J48 cls = new J48();
cls.buildClassifier(train);
// output predictions
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

