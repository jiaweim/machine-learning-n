# 过滤

- [过滤](#过滤)
  - [简介](#简介)
  - [Batch filter](#batch-filter)
  - [实时过滤](#实时过滤)

2023-12-12, 15:32
****

## 简介

filters 用于预处理数据。所有过滤器都在 `weka.filters` 包中，filter 分为两类：

- supervised - 这类 filter 需要 class 属性（已分类）
- unsupervised - 不需要 class 属性

然后可以继续分为两个子类：

- 基于属性（attribute-based），columns 过滤，如添加和删除 columns
- 基于实例（instance-based）, row 过滤，如添加和删除样本

除了以上分类，过滤器还有基于流和基于批两种模式：

- 流过滤器可以直接处理数据
- 批过滤器需要一批数据来初始化内部状态

例如，`ReplaceMissingValues` filter 首先需要一批数据来确定每个属性的均值和模式。否则，无法将缺失值替换为有意义的值，完成初始化后，就可以和流过滤器一样逐行处理数据。

基于实例的过滤器处理数据的方式有点特殊。如前所述，在获得第一批数据后，所有过滤器都可以逐行处理数据。在批处理模式，过滤器会添加或删除 rows；在单行处理模式，过滤器不工作。这是有原因的，以 FilteredClassifier 分类器为例：在训练之后（第一批数据），分类器在测试集上进行评估，每次一个 row，如果此时过滤器添加或删除 row，就无法正确评估了。所以基于实例的过滤器后续不再处理样本。`Resample` 就是这类过滤器。

**示例：** 使用 `Remove` 过滤器删除第一个属性

```java
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;
...
String[] options = new String[2];
options[0] = "-R"; // "range"
options[1] = "1"; // first attribute
Remove remove = new Remove(); // new instance of filter
remove.setOptions(options); // set options
remove.setInputFormat(data); // inform filter about dataset
                            // **AFTER** setting options
Instances newData = Filter.useFilter(data, remove); // apply filter
```

一个常见错误是在调用 `setInputFormat(Instances)` 之后调用 `setOptions`。`setInputFormat(Instances)` 通常用于确定数据的输入格式，所有选项必须在调用它之前设置，之后的设置都会被忽略。

## Batch filter

如果不至一个数据集需要用相同过滤器处理，就需要批过滤器。

如果不使用批过滤器，例如使用 `StringToWordVector` 生成训练集和测试集时，两个完全独立的过滤器，可能导致训练集和测试集不兼容。在两个不同数据集上运行 `StringToWordVector` 会产生两个不同的单词词典，从而生成不同的属性。

**示例：** 使用 Standardize 顾虑器标准化训练集和测试集

```java
Instances train = ... // from somewhere
Instances test = ... // from somewhere
Standardize filter = new Standardize();
// 使用训练集初始化过滤器
filter.setInputFormat(train);

// 使用训练集配置过滤器
Instances newTrain = Filter.useFilter(train, filter);
// 创建测试集
Instances newTest = Filter.useFilter(test, filter);
```

## 实时过滤

实时过滤不需要实现过滤数据，而是只需要设置一个元模式。Weka 提供的实时过滤元模式包括：

- FilteredClassifier
- FilteredAssociator
- FilteredAttributeEval/FilteredSubsetEval

**示例**： 使用 `FilteredClassifier` 和 `Remove` 过滤器删除第一个属性，使用 J48 分类器。

首先使用训练集构建分类器，然后使用单独的测试集评估。实际值和预测值输出到控制台

```java
import weka.classifiers.meta.FilteredClassifier;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.filters.unsupervised.attribute.Remove;
...
Instances train = ... // from somewhere
Instances test = ... // from somewhere
// 过滤器
Remove rm = new Remove();
rm.setAttributeIndices("1"); // remove 1st attribute
// classifier
J48 j48 = new J48();
j48.setUnpruned(true); // using an unpruned J48
// meta-classifier
FilteredClassifier fc = new FilteredClassifier();
fc.setFilter(rm);
fc.setClassifier(j48);
// train and output model
fc.buildClassifier(train);
System.out.println(fc);
for (int i = 0; i < test.numInstances(); i++) {
    double pred = fc.classifyInstance(test.instance(i));
    double actual = test.instance(i).classValue();
    System.out.print("ID: "
        + test.instance(i).value(0));
    System.out.print(", actual: "
        + test.classAttribute().value((int) actual));
    System.out.println(", predicted: "
        + test.classAttribute().value((int) pred));
}
```
