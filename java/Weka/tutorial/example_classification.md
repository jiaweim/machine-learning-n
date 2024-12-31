# 分类示例

## 简介

使用 Weka 加载数据，选择特征，实现分类器，以及评价性能。

## 数据

zoo 数据集，包含 101 条数据，每条数据对应一个动物，包含 18 个属性：

|编号|名称|类型|
|---|---|---|
| 1|animal|name:	Unique for each instance|
| 2|hair		|Boolean|
| 3|feathers	|Boolean|
| 4|eggs		|Boolean|
| 5|milk		|Boolean|
| 6|airborne	|Boolean|
| 7|aquatic		|Boolean|
| 8|predator	|Boolean|
| 9|toothed		|Boolean|
|10|backbone	|Boolean|
|11|breathes	|Boolean|
|12|venomous	|Boolean|
|13|fins		|Boolean|
|14|legs		|Numeric (set of values: {0,2,4,5,6,8})|
|15|tail		|Boolean|
|16|domestic	|Boolean|
|17|catsize		|Boolean|
|18|type		|Numeric (integer values in range [1,7])|

下面构建一个 模型，来预测 animal。

## 加载数据

使用 `DataSource` 加载数据，数据集保存为 `Instances` 对象：

```java
InputStream stream = ZooClassifier.class.getResourceAsStream("zoo.arff");
Instances dataset = ConverterUtils.DataSource.read(stream);
System.out.println(dataset.numInstances()); // 样本个数
```

```
101
```

建模目的是预测 `animal` 属性，所以使用 `Remove` 过滤器删除该属性。

```java
Remove remove = new Remove();
remove.setOptions(Utils.splitOptions("-R 1"));
remove.setInputFormat(dataset);
dataset = Filter.useFilter(dataset, remove);
```

## 选择特征

特征选择算法评估不同特征的子集，计算打分，来判断哪些特征对建模比较有用。

特性选择算法包含两部分：

- 搜索算法
- 打分方法

下面使用 `InfoGainAttributeEval` 作为打分方法，`Ranker` 作为搜索算法

```java
InfoGainAttributeEval eval = new InfoGainAttributeEval();
Ranker search = new Ranker();
AttributeSelection attributeSelection = new AttributeSelection();
attributeSelection.setEvaluator(eval);
attributeSelection.setSearch(search);
attributeSelection.SelectAttributes(dataset);

int[] indices = attributeSelection.selectedAttributes();
System.out.println(Utils.arrayToString(indices));
```

```
12,3,7,2,0,1,8,9,13,4,11,5,15,10,6,14,16
```

可以看到，包含信息最多的属性有 12 (fins), 3 (eggs), 7 (aquatic), 2 (hair) 等。基本该排序，可以删除信息量不高的特征。

那么，保留多少特征？没有确切的规则，特征数量取决于数据和问题，所以最好的办法，是分析这些特征是否能改进模型。

## 学习算法

假设我们已经加载数据并选择了最佳特征，现在准备学习一个分类模型。我们从基本的决策树开始。在 Weka 中，J48 类实现决策树，它是著名的 Quinlan C4.5 决策树的重新实现。

生成决策树的步骤如下：

1. 初始化 J48 决策树

```java
J48 tree = new J48();
String[] options = new String[1];
options[0] = "-U";

tree.setOptions(options);
```

