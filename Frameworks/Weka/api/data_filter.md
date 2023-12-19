# 过滤器

- [过滤器](#过滤器)
  - [简介](#简介)
  - [批过滤](#批过滤)
  - [动态过滤](#动态过滤)
  - [调用约定](#调用约定)
  - [Add 过滤器](#add-过滤器)

2023-12-12, 15:32
****

## 简介

过滤器用于**预处理数据**。所有过滤器都在 `weka.filters` 包中，分为：

- 监督和非监督：是否需要 class 属性
- 基于特征和基于实例：对特征（column）还是对样本（row）的操作

!!! note
    由于已经有 meta-classifier，单独应用过滤器不是很有必要。为了完整性，这里单独介绍。

**示例：** 使用 `Remove` 过滤器删除第一个特征

```java
String[] options = new String[2];
options[0] = "-R"; // "range"
options[1] = "1"; // first attribute
Remove remove = new Remove(); // new instance of filter
remove.setOptions(options); // set options
remove.setInputFormat(data); // inform filter about dataset
                            // **AFTER** setting options
Instances newData = Filter.useFilter(data, remove); // apply filter
```

除了以上分类，过滤器还有基于流和基于批两种模式：

- 流过滤器可以直接处理数据
- 批过滤器需要一批数据来初始化内部状态

例如，`ReplaceMissingValues`过滤器首先需要一批数据来确定每个属性的均值和模式。否则，无法将缺失值替换为有意义的值，完成初始化后，就可以和流过滤器一样逐行处理数据。

基于实例的过滤器处理数据的方式有点特殊。如前所述，在获得第一批数据后，所有过滤器都可以逐行处理数据。在批处理模式，过滤器会添加或删除 rows；在单行处理模式，过滤器不工作。这是有原因的，以 FilteredClassifier 分类器为例：在训练之后（第一批数据），分类器在测试集上进行评估，每次一个 row，如果此时过滤器添加或删除 row，就无法正确评估了。所以基于实例的过滤器后续不再处理样本。`Resample` 就是这类过滤器。

## 批过滤

批过滤指将相同配置的过滤器应用于多个数据集。在属性选择和标准化中，这是必要的。

例如使用 `StringToWordVector` 生成训练集和测试集时，如果使用两个独立的过滤器，可能导致训练集和测试集不兼容，在两个不同数据集上运行 `StringToWordVector` 会产生两个不同的单词词典，从而生成不同的属性。

批过滤实现起来很容易：

- 使用训练集调用 `setInputFormat(Instances)` 初始化过滤器
- 将过滤器应用于训练集和测试集

**示例：** 使用 `Standardize` 过滤器标准化训练集和测试集

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

## 动态过滤

动态过滤不需要立即过滤数据，而是预先配置好过滤器和分类器。Weka 提供的动态过滤元模式包括：

- `FilteredClassifier`
- `FilteredAssociator`
- `FilteredAttributeEval`/`FilteredSubsetEval`

`FilteredClassifier` 配置好过滤器和分类器后，在训练时会自动应用过滤器，在预测时会自动跳过过滤器。

**示例**： 使用 `FilteredClassifier` 和 `Remove` 过滤器删除 ID 属性，即第一个属性，使用 J48 分类器。

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

## 调用约定

方法调用顺序：

1. `setOptions`
2. `setInputFormat`
3. `Filter.useFilter(Instances,Filter)`

重点：`setInputFormat` 必须是应用过滤器前最后一个调用。因为许多过滤器在 `setInputFormat` 中根据当前设置选项生成输出格式的标题。所以在调用 `setInputFormat` 之前，应该完成所有设置选项。

## Add 过滤器

添加 1 个 nominal 属性，1 个 numeric 属性到数据集：

- 参数 1 是 arff 文件
- 参数 2 是 java 或 filter
  - filter 表示使用 Add 过滤器添加这 2 个属性
  - java 表示直接使用 java 代码添加这 2 个属性

```java
public static void main(String[] args) throws Exception {
    if (args.length != 2) {
        System.out.println("\nUsage: <file.arff> <filter|java>\n");
        System.exit(1);
    }

    // 加载数据
    Instances data = ConverterUtils.DataSource.read(args[0]);
    Instances newData = null;

    // 使用过滤器还是 java？
    if (args[1].equals("filter")) {
        Add filter;
        newData = new Instances(data);
        // 1. nominal attribute
        filter = new Add();
        filter.setAttributeIndex("last");
        filter.setNominalLabels("A,B,C,D");
        filter.setAttributeName("NewNominal");
        filter.setInputFormat(newData);
        newData = Filter.useFilter(newData, filter);
        // 2. numeric attribute
        filter = new Add();
        filter.setAttributeIndex("last");
        filter.setAttributeName("NewNumeric");
        filter.setInputFormat(newData);
        newData = Filter.useFilter(newData, filter);
    } else if (args[1].equals("java")) {
        newData = new Instances(data);
        // add new attributes
        // 1. nominal
        ArrayList<String> values = new ArrayList<String>();
        values.add("A");
        values.add("B");
        values.add("C");
        values.add("D");
        newData.insertAttributeAt(new Attribute("NewNominal", values), newData.numAttributes());
        // 2. numeric
        newData.insertAttributeAt(new Attribute("NewNumeric"), newData.numAttributes());
    } else {
        System.out.println("\nUsage: <file.arff> <filter|java>\n");
        System.exit(2);
    }

    // random values
    Random rand = new Random(1);
    for (int i = 0; i < newData.numInstances(); i++) {
        // 1. nominal
        newData.instance(i).setValue(newData.numAttributes() - 2, rand.nextInt(4));  // index of labels A:0,B:1,C:2,D:3
        // 2. numeric
        newData.instance(i).setValue(newData.numAttributes() - 1, rand.nextDouble());
    }

    // output on stdout
    System.out.println(newData);
}
```