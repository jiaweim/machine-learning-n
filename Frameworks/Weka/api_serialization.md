# 序列化

- [序列化](#序列化)
  - [简介](#简介)
  - [序列化分类器](#序列化分类器)
  - [反序列化分类器](#反序列化分类器)
  - [反序列化 Explorer 保存的分类器](#反序列化-explorer-保存的分类器)
  - [序列化 Explorer 模型](#序列化-explorer-模型)

2023-12-12, 19:51
****

## 简介

下面介绍如何序列化和反序列化一个 J48 分类器。

## 序列化分类器

`weka.core.SerializationHelper` 的 write 用于序列化：

```java
import weka.classifiers.Classifier;
import weka.classifiers.trees.J48;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.SerializationHelper;
...
// load data
Instances inst = DataSource.read("/some/where/data.arff");
inst.setClassIndex(inst.numAttributes() - 1);
// train J48
Classifier cls = new J48();
cls.buildClassifier(inst);
// serialize model
SerializationHelper.write("/some/where/j48.model", cls);
```

## 反序列化分类器

```java
import weka.classifiers.Classifier;
import weka.core.SerializationHelper;
...
// deserialize model
Classifier cls = (Classifier) SerializationHelper.read(
    "/some/where/j48.model");
```

## 反序列化 Explorer 保存的分类器

Explorer 保存的模型文件包含模型和训练集标题信息。通过检查数据集信息，可以很容易判断序列化的分类器是否适合当前数据集。readAll 方法返回文件中包含对象的数组：

```java
import weka.classifiers.Classifier;
import weka.core.Instances;
import weka.core.SerializationHelper;
...
// the current data to use with classifier
Instances current = ... // from somewhere
// deserialize model
Object o[] = SerializationHelper.readAll("/some/where/j48.model");
Classifier cls = (Classifier) o[0];
Instances data = (Instances) o[1];
// is the data compatible?
if (!data.equalHeaders(current))
    throw new Exception("Incompatible data!");
```

## 序列化 Explorer 模型

如果希望和 Explorer 一样，在序列化分类器时包含数据集的标题信息，可以使用 writeAll 方法：

```java
import weka.classifiers.Classifier;
import weka.classifiers.trees.J48;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.SerializationHelper;
...
// load data
Instances inst = DataSource.read("/some/where/data.arff");
inst.setClassIndex(inst.numAttributes() - 1);
// train J48
Classifier cls = new J48();
cls.buildClassifier(inst);
// serialize classifier and header information
Instances header = new Instances(inst, 0);
SerializationHelper.writeAll(
    "/some/where/j48.model", new Object[]{cls, header});
```