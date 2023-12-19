# 选择属性

- [选择属性](#选择属性)
  - [简介](#简介)
  - [元分类器](#元分类器)
  - [过滤器](#过滤器)
  - [直接使用 API](#直接使用-api)

2023-12-12, 19:03
****

## 简介

合理准备数据对训练模型非常重要。减少属性数量不仅有助于加快模型训练，还有助于避免不管属性干扰。

目前 weka 有三种属性评估器：

- 单属性评估器：`weka.attributeSelection.AttributeEvaluator` 接口对应这类属性评估器，`Ranker` 检索算法通常与这类算法结合使用。
- 属性子集评估器：`weka.attributeSelection.SubsetEvaluator` 接口，对属性子集进行评估。
- 属性集评估器：评估属性集合，`weka.attributeSelection.AttributeSetEvaluator` 接口实现类。

目前大多数属性选择方案实现都是有监督的，即需要带 class 属性的数据集。

无监督评估算法需继承以下类：

- `weka.attributeSelection.UnsupervisedAttributeEvaluator`，如 `LatentSemanticAnalysis`, `PrincipalComponents`
- `weka.attributeSelection.UnsupervisedSubsetEvaluator`，目前还没有具体实现

属性选择提供即时过滤：

- `weka.attributeSelection.FilteredAttributeEval`：评估单值属性的过滤器
- `weka.attributeSelection.FilteredSubsetEval`：评估属性子集的过滤器

weka 提供了三种执行属性选择的方式：

- 使用 meta-classifier：用于即时属性选择，类似 `FilteredClassifier` 的即时过滤
- 使用 filter：用于预处理数据
- 底层 API：不使用 meta 模式（过滤器或分类器），直接使用属性选择 API

## 元分类器

元分类器 `AttributeSelectedClassifier` 类似与 FilteredClassifier 类似。但是，AttributeSelectedClassifier 不是基于分类器或过滤器执行过滤，而是使用搜索算法（派生自 weka.attributeSelection.ASEvaluation）和评估器（派生自 weka.attributeSelection.ASSearch）执行属性选择，然后使用一个基本分类器对简化后的数据进行训练。

**示例：** 使用 J48 作为基本分类器，CfsSubsetEval 作为评估器，GreedyStepwise 作为搜索算法

```java
Instances data = ... // from somewhere

CfsSubsetEval eval = new CfsSubsetEval();
GreedyStepwise search = new GreedyStepwise();
search.setSearchBackwards(true);
J48 base = new J48();

// 配置元分类器
AttributeSelectedClassifier classifier = new AttributeSelectedClassifier();
classifier.setClassifier(base);
classifier.setEvaluator(eval);
classifier.setSearch(search);
// cross-validate classifier
Evaluation evaluation = new Evaluation(data);
evaluation.crossValidateModel(classifier, data, 10, new Random(1));
System.out.println(evaluation.toSummaryString());
```

## 过滤器

如果数据只需要降维，不用于训练分类器，可以使用过滤器。

`AttributeSelection` 过滤器以评估器和搜索算法为参数。

**示例：** 以 `CfsSubsetEval` 为评估器，`GreedyStepwise` 为搜索算法，输出过滤后的简化数据

```java
Instances data = ... // from somewhere

CfsSubsetEval eval = new CfsSubsetEval();
GreedyStepwise search = new GreedyStepwise();
search.setSearchBackwards(true);

// 设置过滤器
AttributeSelection filter = new AttributeSelection();
filter.setEvaluator(eval);
filter.setSearch(search);
filter.setInputFormat(data);
// filter data
Instances newData = Filter.useFilter(data, filter);
System.out.println(newData);
```

## 直接使用 API

使用元分类器或过滤器选择属性很容易，但可能无法满足所有人的需求。例如，想获取属性排序（使用 Ranker）或检索所选属性的索引，而不是简化数据。

**示例：** 使用 `CfsSubsetEval` 评估器和 `GreedyStepwise` 检索算法，在控制台输出选择的索引

```java
Instances data = ... // from somewhere
// setup attribute selection
AttributeSelection attsel = new AttributeSelection();
CfsSubsetEval eval = new CfsSubsetEval();
GreedyStepwise search = new GreedyStepwise();
search.setSearchBackwards(true);
attsel.setEvaluator(eval);
attsel.setSearch(search);
// perform attribute selection
attsel.SelectAttributes(data);
int[] indices = attsel.selectedAttributes();
System.out.println(
    "selected attribute indices (starting with 0):\n"
    + Utils.arrayToString(indices));
```
