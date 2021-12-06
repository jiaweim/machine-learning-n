# 机器学习术语

- [机器学习术语](#机器学习术语)
  - [A/B 测试（A/B testing）](#ab-测试ab-testing)
  - [准确率（accuracy）](#准确率accuracy)
  - [行动（action）](#行动action)
  - [激活函数 (activation function)](#激活函数-activation-function)
  - [聚集聚类（agglomerative clustering）](#聚集聚类agglomerative-clustering)
  - [二元分类 (binary classification)](#二元分类-binary-classification)
  - [形心（centroid）](#形心centroid)
  - [基于质心的聚类（centroid-based clustering）](#基于质心的聚类centroid-based-clustering)
  - [分类模型（classification model）](#分类模型classification-model)
  - [聚类（clustering）](#聚类clustering)
  - [多类别分类 (multi-class classification)](#多类别分类-multi-class-classification)
  - [样本（example）](#样本example)
  - [特征（feature）](#特征feature)
  - [层次聚类（hierarchical clustering）](#层次聚类hierarchical-clustering)
  - [标签（label）](#标签label)
  - [多类别分类 (multi-class classification)](#多类别分类-multi-class-classification-1)
  - [预测（prediction）](#预测prediction)
  - [修正线性单元 (ReLU, Rectified Linear Unit)](#修正线性单元-relu-rectified-linear-unit)
  - [S 型函数 (sigmoid function)](#s-型函数-sigmoid-function)
  - [负例 (TN, true negative)](#负例-tn-true-negative)
  - [正例 (TP, true positive)](#正例-tp-true-positive)

2021-02-14, 23:37
***

## A/B 测试（A/B testing）

一种比较两种或多种技术的统计方法，通常是将当前采用的技术与新技术进行比较。

A/B 测试不仅用于确定哪种技术的效果更好，而且还有助于了解相应差异是否具有显著的统计学意义。A/B 测试通常采用一种衡量标准对两种技术进行比较，但也适用于任意数量的技术和衡量方式。

## 准确率（accuracy）

[分类模型](#分类模型classification-model)正确[预测](#预测prediction)所占的比例。在[多类别分类](#多类别分类-multi-class-classification)中，准确率的定义如下：

$$准确率=\frac{正确的预测数}{样本总数}$$

在[二元分类](#二元分类-binary-classification)中，准确率的定义如下：

$$准确率=\frac{True Positives+True Negatives}{样本总数}$$

请参考[正例](#正例-tp-true-positive)和[负例](#负例-tn-true-negative)。

## 行动（action）

在强化学习中，

## 激活函数 (activation function)

一种函数（例如 [ReLU](#修正线性单元-relu-rectified-linear-unit) 或 [S 型函数](#s-型函数-sigmoid-function)），用于对上一层的所有输入求加权和，然后生成一个输出值（通常为非线性值），并将其传递给下一层。

## 聚集聚类（agglomerative clustering）

参考 层次聚类。

## 二元分类 (binary classification)

二元分类 (binary classification)

## 形心（centroid）

聚类的中心，形心是由 k-means 或 k-median 算法确定。例如，如果 k 是 3，那么 k-means 或 k-median算法会找到 3 个形心。

## 基于质心的聚类（centroid-based clustering）

非层次聚类算法。k-means 是目前应用最广泛的基于质心的聚类算法。

和层次聚类算法相反。

## 分类模型（classification model）

一种机器学习模型，用于区分两种或多种离散类别。例如，某个自然语言处理分类模型可以确定输入的句子是法语、西班牙语还是意大利语。请与回归模型进行比较。

## 聚类（clustering）

将相关的样本分成一组，一般用于非监督学习。在所有样本分组完毕后，便可以选择性地为每个聚类赋予含义。

聚类算法有很多。例如，k-means 算法会基于样本与[形心](#形心centroid)

## 多类别分类 (multi-class classification)

区分两种以上类别的分类问题。例如，枫树大约有 128 种，因此，确定枫树种类的模型就属于多类别模型。反之，仅将电子邮件分为两类（“垃圾邮件”和“非垃圾邮件”）的模型属于二元分类模型。

## 样本（example）

数据集的一行。一个样本包含一个或多个[特征](#特征feature)，可能还包含一个标签。参阅有标签样本和无标签样本。

## 特征（feature）

在进行预测时使用的输入变量。

## 层次聚类（hierarchical clustering）

一种聚类算法，用于创建聚类树。层次聚类适合于层次数据（hierarchical data），例如植物分类。层次聚类算法有两种：

- 凝聚聚类（agglomerative clustering）首先为每个实例分配类别，然后迭代合并接近的类别，从而创建聚类树。
- 分裂聚类（divisive clustering）首先将所有示例分组到一个类别，然后迭代拆分类别为聚类树。

层次聚类与基于质心（centroid-based）聚类相反。

## 标签（label）

在监督式学习中，标签指[样本](#样本example)的“答案”或“结果”。有标签数据集中的每个样本都包含一个或多个特征以及一个标签。

例如，在住房数据集中，特征可能包括卧室数、卫生间数以及房龄，而标签可能是房价。在垃圾邮件检测数据集中，特征可能包括主题行、发件人以及电子邮件本身，而标签则可能是“垃圾邮件”或“非垃圾邮件”。

## 多类别分类 (multi-class classification)

区分两种以上类别的分类问题。例如，枫树大约有 128 种，因此，确定枫树种类的模型就属于多类别模型。反之，仅将电子邮件分为两类（“垃圾邮件”和“非垃圾邮件”）的模型属于二元分类模型。

## 预测（prediction）

模型对指定输入[样本](#样本example)的输出。

## 修正线性单元 (ReLU, Rectified Linear Unit)

一种激活函数，其规则如下：

- 如果输入为负数或 0，则输出 0。
- 如果输入为正数，则输出等于输入。

## S 型函数 (sigmoid function)

一种函数，可将逻辑回归输出或多项回归输出（对数几率）映射到概率，以返回介于 0 到 1 之间的值。S 型函数的公式如下：

在逻辑回归问题中， 非常简单：

换句话说，S 型函数可将  转换为介于 0 到 1 之间的概率。

在某些神经网络中，S 型函数可作为激活函数使用。

## 负例 (TN, true negative)

被模型正确地预测为负类别的样本。例如，模型推断出某封电子邮件不是垃圾邮件，而该电子邮件确实不是垃圾邮件。

## 正例 (TP, true positive)

被模型正确地预测为正类别的样本。例如，模型推断出某封电子邮件是垃圾邮件，而该电子邮件确实是垃圾邮件。
