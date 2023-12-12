# Weka API
- [Weka API](#weka-api)
  - [简介](#简介)
  - [参数设置](#参数设置)
    - [setOptions](#setoptions)
    - [get/set](#getset)
- [Classifying instances](#classifying-instances)

***

## 简介

Weka API 使用选项：

- 设置参数
- 创建数据集
- 加载和保存数据
- 过滤
- 分类（Classifying）
- 聚类（Clustering）
- 选择属性
- 可视化
- 序列化

Weka API 常使用的组件包括：
- `Instances`，数据
- `Filter`，数据预处理
- `Classifier/Clusterer`，在处理后的数据上构建模型
- `Evaluating`，评估分类器/聚类器的性能
- `Attribute selection`，从数据中移除无关属性

下面解释如何使用这些内容。

## 参数设置

设置属性，例如设置 classifier 的属性，通常有两种方式：

- 通过对应的 get/set 方法
- 如果该类实现了 `weka.core.OptionHandler` 接口，还可以通过 `setOptions(String[])` 解析命令行选项，其对应的方法为 `getOptions()`。`setOptions(String[])` 的缺点是，无法增量设置选项，为设置选项采用默认值。

### setOptions

**例 1：** 为 `Remove` 过滤器设置选项

```java
import weka.filters.unsupervised.attribute.Remove;

String[] options = new String[]{"-R", "1"};
Remove rm = new Remove();
rm.setOptions(options);
```

`setOptions(String[])` 方法需要一个完全解析并正确分割的数组（由控制台完成），因此该方法有一些缺陷：

- 合并选项和参数会报错，例如使用 "-R 1" 作为 `String` 数组元素，WEKA 无法识别
- 无法识别命令中的空格，例如 "-R " 会报错。

为了避免该错误，可以使用 Weka 提供的 `Utils` 类来分割命令，例如：

```java
import weka.core.Utils;

String[] options = Utils.splitOptions("-R 1");
```

此方法会忽略多余空格，使用 " -R 1" 或 "-R 1 " 返回相同结果。

带**嵌套选项**的复杂命令行，例如，包含内核设置的支持向量机分类器 SMO（`weka.classifiers.functions`）有些麻烦，因为 Java 需要在字符串中转义双引号和反斜线。

### get/set

使用属性的 set 方法更直观：

```java
import weka.filters.unsupervised.attribute.Remove;

Remove rm = new Remove();
rm.setAttributeIndices("1");
```

要确定哪个选项对应哪个属性，建议查看 `setOptions(String[])` 和 `getOptions()` 方法。

使用 setter 方法时，可能会遇到需要 `weka.core.SelectedTag` 参数的方法。如 `GridSearch` 的 `setEvaluation` 方法。`SelectedTag` 用于 GUI 中显示下拉列表。

可以使用所有可能的 `weka.core.Tag` 数组，或 `Tag` 的 integer 或 string ID 数组创建 `SelectedTag`。例如，GridSearch 的 setOptions(String[]) 使用提供的 string ID 设置评估类型，如 "ACC" 表示 accuracy，如果缺失，默认为整数 ID EVALUATION ACC。两种情况都使用了定义所有可能选项的数组 TAGS EVALUATION：

```java
import weka.core.SelectedTag;
...
String tmpStr = Utils.getOption(’E’, options);
if (tmpStr.length() != 0)
    setEvaluation(new SelectedTag(tmpStr, TAGS_EVALUATION));
else
    setEvaluation(new SelectedTag(EVALUATION_CC, TAGS_EVALUATION));
```

# Classifying instances
如果你想使用新训练的分类器对未标记的数据集进行分类，可以使用如下的代码示例。

该示例载入数据文件 `/some/where/unlabeled.arff`，使用之前构建的分类器 `tree` 用来标记数据，然后将标记后的数据保存在 `/some/where/labeled.arff`。
```java
 import java.io.BufferedReader;
 import java.io.BufferedWriter;
 import java.io.FileReader;
 import java.io.FileWriter;
 import weka.core.Instances;
 ...
 // load unlabeled data
 Instances unlabeled = new Instances(
    new BufferedReader(new FileReader("/some/where/unlabeled.arff")));

 // set class attribute
 unlabeled.setClassIndex(unlabeled.numAttributes() - 1);

 // create copy
 Instances labeled = new Instances(unlabeled);

 // label instances
 for (int i = 0; i < unlabeled.numInstances(); i++) {
   double clsLabel = tree.classifyInstance(unlabeled.instance(i));
   labeled.instance(i).setClassValue(clsLabel);
 }
 // save labeled data
 BufferedWriter writer = new BufferedWriter(new FileWriter("/some/where/labeled.arff"));
 writer.write(labeled.toString());
 writer.newLine();
 writer.flush();
 writer.close();
```
