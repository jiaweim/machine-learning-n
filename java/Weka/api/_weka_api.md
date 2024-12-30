# Weka API

- [Weka API](#weka-api)
  - [简介](#简介)
  - [1. 参数设置](#1-参数设置)
    - [setOptions](#setoptions)
      - [String 数组](#string-数组)
      - [String 字符串](#string-字符串)
      - [OptionsToCode](#optionstocode)
    - [get/set](#getset)
  - [2. 数据读写](#2-数据读写)
    - [从文件加载数据](#从文件加载数据)
      - [ARFF 文件](#arff-文件)
      - [CSV 文件](#csv-文件)
      - [设置分类属性](#设置分类属性)
    - [从数据库加载数据](#从数据库加载数据)
    - [保存到文件](#保存到文件)
      - [保存为 ARFF](#保存为-arff)
      - [保存为 CSV](#保存为-csv)
    - [保存到数据库](#保存到数据库)
  - [3. 创建数据集](#3-创建数据集)
    - [定义格式](#定义格式)
    - [添加数据](#添加数据)
    - [示例](#示例)
  - [4. 生成人工数据](#4-生成人工数据)
    - [生成 ARFF 文件](#生成-arff-文件)
    - [生成 Instances](#生成-instances)
  - [5. 数据随机化](#5-数据随机化)
  - [6. 过滤](#6-过滤)
    - [批过滤](#批过滤)
    - [动态过滤](#动态过滤)
    - [调用约定](#调用约定)
    - [Add 过滤器](#add-过滤器)
  - [7. 分类](#7-分类)
    - [构建分类器](#构建分类器)
      - [批分类器](#批分类器)
      - [增量分类器](#增量分类器)
    - [评估分类器](#评估分类器)
      - [交叉验证](#交叉验证)
      - [测试集](#测试集)
      - [统计量](#统计量)
      - [收集预测结果](#收集预测结果)
    - [预测](#预测)
  - [8. 聚类](#8-聚类)
    - [构建聚类器](#构建聚类器)
      - [批聚类器](#批聚类器)
      - [增量聚类器](#增量聚类器)
    - [评估聚类器](#评估聚类器)
      - [classes to clusters](#classes-to-clusters)
    - [样本聚类](#样本聚类)
  - [9. 特征选择](#9-特征选择)
    - [元分类器](#元分类器)
    - [过滤器](#过滤器)
    - [直接使用 API](#直接使用-api)
  - [10. 可视化](#10-可视化)
    - [ROC 曲线](#roc-曲线)
    - [Graph](#graph)
      - [Tree](#tree)
      - [BayesNet](#bayesnet)
  - [11. 序列化](#11-序列化)
    - [序列化分类器](#序列化分类器)
    - [反序列化分类器](#反序列化分类器)
    - [反序列化 Explorer 保存的分类器](#反序列化-explorer-保存的分类器)
    - [序列化 Explorer 模型](#序列化-explorer-模型)
  - [参考](#参考)

2024-12-30 ⭐
@author Jiawei Mao
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
- `Evaluating`，评估分类模型/聚类模型的性能
- `Attribute selection`，从数据中移除无关属性

下面解释如何使用这些内容。

完整内容可参考 Wek Manual 的 "Using the API" 这一章。


## 1. 参数设置

设置对象属性，例如设置 classifier 的属性，通常有两种方式：

- get/set 方法
- 如果该类实现了 `weka.core.OptionHandler` 接口，则可以通过 `setOptions(String[])` 解析命令行选项，其对应的方法为 `getOptions()`。

这两种方式的差别在于：`setOptions(String[])` 无法增量设置选项，**未设置选项**采用默认值。

### setOptions

`weka.core.OptionHandler` 接口提供了如下两个方法：

```java
void setOptions(String[] options)
String[] getOptions()
```

分类器、聚类器和过滤器均实现了该接口。

#### String 数组

直接使用 `String[]` 数组作为参数：

```java
String[] options = new String[2];
options[0] = "-R";
options[1] = "1";
```

**示例：** 为 `Remove` 过滤器设置选项

```java
import weka.filters.unsupervised.attribute.Remove;

String[] options = new String[]{"-R", "1"};
Remove rm = new Remove();
rm.setOptions(options);
```

#### String 字符串

使用单个字符串，用 `weka.core.Utils` 的 `splitOptions` 拆分出选项：

```java
String[] options = weka.core.Utils.splitOptions("-R 1");
```

`setOptions(String[])` 方法需要一个完全解析并正确分割的数组（由控制台完成），因此该方法有一些缺陷：

- 合并选项和参数会报错，例如使用 "-R 1" 作为 `String` 数组元素，WEKA 无法识别
- 无法识别命令中的空格，例如 "-R " 会报错。

为了避免该错误，Weka 提供了 `Utils` 类来分割命令，例如：

```java
import weka.core.Utils;

String[] options = Utils.splitOptions("-R 1");
```

此方法会忽略多余空格，使用 " -R 1" 或 "-R 1 " 返回相同结果。

#### OptionsToCode

带**嵌套选项**的复杂命令行，例如，包含内核设置的支持向量机分类器 SMO（`weka.classifiers.functions`）有些麻烦，因为 Java 需要在字符串中转义双引号和反斜线。使用 `OptionsToCode.java` 将命令选项自动转换为代码，对命令包含带选项的嵌套类时非常方便。例如 SMO 内核：

```java
java OptionsToCode weka.classifiers.functions.SMO
```

会生成：

```java
 // create new instance of scheme
 weka.classifiers.functions.SMO scheme = new weka.classifiers.functions.SMO();
 // set options
 scheme.setOptions(weka.core.Utils.splitOptions("-C 1.0 -L 0.0010 -P 1.0E-12 -N 0 -V -1 -W 1 -K \"weka.classifiers.functions.supportVector.PolyKernel -C 250007 -E 1.0\""));
```

`OptionsToCode` 的代码：

```java
public class OptionsToCode {

    /**
     * Generates the code and outputs it on stdout. E.g.:<p/>
     * <code>java OptionsToCode weka.classifiers.functions.SMO -K "weka.classifiers.functions.supportVector.RBFKernel" &gt; OptionsTest.java</code>
     */
    public static void main(String[] args) throws Exception {
        // output usage
        if (args.length == 0) {
            System.err.println("\nUsage: java OptionsToCode <classname> [options] > OptionsTest.java\n");
            System.exit(1);
        }

        // instantiate scheme
        String classname = args[0];
        args[0] = "";
        Object scheme = Class.forName(classname).getDeclaredConstructor().newInstance();
        if (scheme instanceof OptionHandler)
            ((OptionHandler) scheme).setOptions(args);

        // generate Java code
        StringBuffer buf = new StringBuffer();
        buf.append("public class OptionsTest {\n");
        buf.append("\n");
        buf.append("  public static void main(String[] args) throws Exception {\n");
        buf.append("    // create new instance of scheme\n");
        buf.append("    " + classname + " scheme = new " + classname + "();\n");
        if (scheme instanceof OptionHandler handler) {
            buf.append("    \n");
            buf.append("    // set options\n");
            buf.append("    scheme.setOptions(weka.core.Utils.splitOptions(\"" + Utils.backQuoteChars(Utils.joinOptions(handler.getOptions())) + "\"));\n");
            buf.append("  }\n");
        }
        buf.append("}\n");

        // output Java code
        System.out.println(buf);
    }
}
```

### get/set

使用属性的 set 方法更直观：

```java
import weka.filters.unsupervised.attribute.Remove;

Remove rm = new Remove();
rm.setAttributeIndices("1");
```

要确定哪个选项对应哪个属性，建议查看 `setOptions(String[])` 和 `getOptions()` 方法。

使用 setter 方法时，可能会遇到需要 `weka.core.SelectedTag` 参数的方法。如 `GridSearch` 的 `setEvaluation` 方法。`SelectedTag` 用于 GUI 中显示的下拉列表，用户可以从预定义的值列表中选择。

可以使用所有可能的 `weka.core.Tag` 数组，或 `Tag` 的 integer 或 string ID 数组创建 `SelectedTag`。例如，`GridSearch` 的 `setOptions(String[])` 使用提供的 string ID 设置评估类型，如 "ACC" 表示 accuracy，如果缺失，默认为整数 **ID EVALUATION ACC**。两种情况都使用了定义所有可能选项的数组 **TAGS EVALUATION**：

```java
import weka.core.SelectedTag;
...
String tmpStr = Utils.getOption(’E’, options);
if (tmpStr.length() != 0)
    setEvaluation(new SelectedTag(tmpStr, TAGS_EVALUATION));
else
    setEvaluation(new SelectedTag(EVALUATION_CC, TAGS_EVALUATION));
```


## 2. 数据读写


在应用过滤器、分类器或聚类之前，首先需要有数据。WEKA 支持从文件和数据库加载数据。在 `wekaexamples.core.converters` 包可以找到相关示例。

Weka 使用如下类存储数据：

- `weka.core.Instances` - 表示完整数据集。该数据结构基于行
  - 使用 `instance(int)` 访问单行（0-based）
  - 使用 `attribute(int)` 返回 `weka.core.Attribute`，包含列的信息
- `weka.core.Instance` - 表示单行，即一个样本。它封装了一个 double 数组
  - `Instance` 不包含 column 类型信息，需要访问 `weka.core.Instances` 获取（参考 `dataset` 和 `setDataset` 方法）
  - `weka.core.SparseInstance` 用于存储稀疏数据
- `weka.core.Attribute` - 保存单个 column 的类型信息，还包括 nominal 属性的标签，字符串属性的可能值，关系属性的数据集。

### 从文件加载数据

`weka.core.converters.ConverterUtils` 的内部类 `DataSource` 可以根据文件扩展名识别文件类型，然后调用特定的加载器读取数据。

从任意文件类型加载数据：

```java
Instances data1 = DataSource.read("/some/where/dataset.arff");
Instances data2 = DataSource.read("/some/where/dataset.csv");
Instances data3 = DataSource.read("/some/where/dataset.xrff");
```

如果文件扩展名与加载器关联的文件类型不匹配，则需要**显式调用特定加载器**。

#### ARFF 文件

使用 `ArffLoader` 加载 arff 文件。

```java
ArffLoader loader = new ArffLoader();
loader.setFile(new File("C:\\Program Files\\Weka-3-9-6\\data\\iris.arff"));
Instances dataSet = loader.getDataSet();
System.out.println(new Instances(dataSet, 0));
```

```
@relation iris

@attribute sepallength numeric
@attribute sepalwidth numeric
@attribute petallength numeric
@attribute petalwidth numeric
@attribute class {Iris-setosa,Iris-versicolor,Iris-virginica}

@data
```

说明：

- `ArffLoader` 支持多种数据源，包括 `File`, `URL`, `InputStream` 等；
- `new Instances(dataSet, 0)` 以 `dataSet` 为模板创建新的**空数据集**，用来打印标题信息，否则会输出所有样本数据。

#### CSV 文件

使用 `CSVLoader` 加载 CSV 文件。

```java
CSVLoader loader = new CSVLoader();
loader.setSource(new File("D:\\iris.csv"));
Instances dataSet = loader.getDataSet();
System.out.println("\nHeader of dataset:\n");
System.out.println(new Instances(dataSet, 0));
```

```
Header of dataset:

@relation iris

@attribute sepallength numeric
@attribute sepalwidth numeric
@attribute petallength numeric
@attribute petalwidth numeric
@attribute class {Iris-setosa,Iris-versicolor,Iris-virginica}

@data
```

说明：

- iris.csv 文件是用 weka 安装包自带的 iris.arff 转换而来

#### 设置分类属性

不是所有文件格式都存储类别属性信息，如 ARFF 不包含类别属性信息，而 XRFF 包含。如果需要类别属性信息，如使用分类器，可以使用 `setClassIndex(int)` 显式设置。

```java
// 将第一个属性设置为类别属性
if (data.classIndex() == -1)
    data.setClassIndex(0);
...
// 将最后一个属性设置为类别属性
if (data.classIndex() == -1)
    data.setClassIndex(data.numAttributes() - 1);
```

### 从数据库加载数据

使用以下两个类从数据库加载数据：

- `weka.experiment.InstanceQuery`：支持检索稀疏数据
- `weka.core.converters.DatabaseLoader`：可以增量检索数据

`InstanceQuery` 使用示例：

```java
InstanceQuery query = new InstanceQuery();
query.setDatabaseURL("jdbc_url");
query.setUsername("the_user");
query.setPassword("the_password");
query.setQuery("select * from whatsoever");
Instances data = query.retrieveInstances();
```

在批处理检索中使用 `DatabaseLoader` 的示例：

```java
DatabaseLoader loader = new DatabaseLoader();
loader.setSource("jdbc_url", "the_user", "the_password");
loader.setQuery("select * from whatsoever");
Instances data = loader.getDataSet();
```

在增量模式中使用 `DatabaseLoader` 的示例：

```java
DatabaseLoader loader = new DatabaseLoader();
loader.setSource("jdbc_url", "the_user", "the_password");
loader.setQuery("select * from whatsoever");
Instances structure = loader.getStructure();
Instances data = new Instances(structure);
Instance inst;
while ((inst = loader.getNextInstance(structure)) != null)
    data.add(inst);
```

> [!NOTE]
>
> - 并非所有数据库支持增量检索
>
> - 并非所有查询都有唯一键来增量检索 row。此时，可以使用 `setKeys(String)` 提供必要的 columns
> - 如果不能增量检索，则可以将其完全加载到内存


### 保存到文件

`DataSink` （`weka.core.converters.ConverterUtils` 内部类）根据扩展名自动保存到指定类型文件：

```java
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSink;
...
// data structure to save
Instances data = ...
// save as ARFF
DataSink.write("/some/where/data.arff", data);
// save as CSV
DataSink.write("/some/where/data.csv", data);
```

#### 保存为 ARFF

```java
Instances dataset = ...;
ArffSaver saver = new ArffSaver();
saver.setInstances(dataset);
saver.setFile(new File("out/path"));
saver.writeBatch();
```

#### 保存为 CSV

**示例**：显式调用 `CSVSaver` 保存到 CSV 文件

```java
Instances data = ...
CSVSaver saver = new CSVSaver();
saver.setInstances(data);
saver.setFile(new File("/some/where/data.csv"));
saver.writeBatch();
```

### 保存到数据库

除了 KnowledgeFlow，在 weka 中将数据保存到数据库的功能不明显。和 DatabaseLoader 一样，保存数据也分为 batch 模式和增量模式。

**示例：** 批量模式，比较简单

```java
// data structure to save
Instances data = ...
// store data in database
DatabaseSaver saver = new DatabaseSaver();
saver.setDestination("jdbc_url", "the_user", "the_password");
// we explicitly specify the table name here:
saver.setTableName("whatsoever2");
saver.setRelationForTableName(false);
// or we could just update the name of the dataset:
// saver.setRelationForTableName(true);
// data.setRelationName("whatsoever2");
saver.setInstances(data);
saver.writeBatch();
```

**示例：** 增量模式，稍微麻烦一点

```java
// data structure to save
Instances data = ...
// store data in database
DatabaseSaver saver = new DatabaseSaver();
saver.setDestination("jdbc_url", "the_user", "the_password");
// we explicitly specify the table name here:
saver.setTableName("whatsoever2");
saver.setRelationForTableName(false);
// or we could just update the name of the dataset:
// saver.setRelationForTableName(true);
// data.setRelationName("whatsoever2");
saver.setRetrieval(DatabaseSaver.INCREMENTAL);
saver.setStructure(data);
count = 0;
for (int i = 0; i < data.numInstances(); i++) {
    saver.writeIncremental(data.instance(i));
}
// notify saver that we’re finished
saver.writeIncremental(null);
```

## 3. 创建数据集

创建数据集 `weka.core.Instances` 对象分两步：

1. 通过设置属性定义数据格式
2. 逐行添加数据

可以参考 `wekaexamples.core.CreateInstances` 示例，该示例中场景一个 `Instances` 对象，其中包含 WEKA 支持的所有属性类型。

### 定义格式

WEKA 支持 5 种类型的数据。

|类型|描述|
|---|---|
|numeric|连续数值变量|
|date|日期|
|nominal|预定义标签|
|string|文本数据|
|relational|关系数据|

对这些数据，统一用 `weka.core.Attribute` 类定义，只是构造函数不同。

**numeric**

数字是最简单的属性类型，只需提供属性名称定义：

```java
Attribute numeric = new Attribute("name_of_attr");
```

**date**

日期属性在内部以数字处理，但是为了正确解析和显示日期，需要指定日期格式。具体格式可参考 `java.text.SimpleDateFormat` 类。例如：

```java
Attribute date = new Attribute("name_of_attr", "yyyy-MM-dd");
```

**nominal**

nominal 属性包含**预定义标签**，需要以 `java.util.ArrayList<String>` 提供，例如：

```java
ArrayList<String> labels = new ArrayList<String>();
labels.addElement("label_a");
labels.addElement("label_b");
labels.addElement("label_c");
labels.addElement("label_d");

Attribute nominal = new Attribute("name_of_attr", labels);
```

**string**

与 nominal 属性不同，此类型没有预定义标签，通用用于存储文本数据。构造函数与创建 nominal 属性相同，但是后面的 `java.util.ArrayList<String>` 提供 null 值：

```java
Attribute string = new Attribute("name_of_attr", (ArrayList<String>)null);
```

**relational**

定义流程：

- 定义 `ArrayList<Attribute>`
- 定义 `Instances`
- 定义 relational `Attribute`

这类属性使用另一个 `weka.core.Instances` 对象定义关系属性。例如，下面生成一个包含 numeric 属性和 nominal 属性之间的关系属性：

```java
ArrayList<Attribute> atts = new ArrayList<Attribute>();
atts.addElement(new Attribute("rel.num"));

ArrayList<String> values = new ArrayList<String>();
values.addElement("val_A");
values.addElement("val_B");
values.addElement("val_C");
atts.addElement(new Attribute("rel.nom", values));

Instances rel_struct = new Instances("rel", atts, 0);
Attribute relational = new Attribute("name_of_attr", rel_struct);
```

然后通过包含所有属性的 `java.util.ArrayList<Attribute>` 创建 `weka.core.Instances`。

**示例：** 创建包含 2 个数值属性，一个包含 2 个标签的 nominal 属性

```java
Attribute num1 = new Attribute("num1");
Attribute num2 = new Attribute("num2");

ArrayList<String> labels = new ArrayList<String>();
labels.addElement("no");
labels.addElement("yes");
Attribute cls = new Attribute("class", labels);

ArrayList<Attribute> attributes = new ArrayList<Attribute>();
attributes.addElement(num1);
attributes.addElement(num2);
attributes.addElement(cls);

Instances dataset = new Instances("Test-dataset", attributes, 0);
```

`Instances` 构造函数的最后一个参数指定初始大小，设置为合适的值可以避免昂贵的扩展操作。设置太大也没关系，最后可以使用 `compactify()` 紧凑。

### 添加数据

定义好数据集结构后，就可以开始逐行添加数据。

- `weka.core.Instance` 接口表示一个数据实例
- `weka.core.AbstractInstance` 实现该接口，提供了 `weka.core.DenseInstance` 和 `weka.core.SparseInstance` 的共有功能
  - `DenseInstance` 表示常规样本
  - `SparseInstance` 则只保存非 0 值

下面主要说明 `DenseInstance` 的使用，`SparseInstance` 的使用方法类似。

**构造函数**

`DenseInstance` 有两个构造函数：

```java
DenseInstance(double weight, double[] attValues)
```

以指定 weight 值和 double 数组创建 `DenseInstance`。WEKA 对**所有属性类型**都采用 double 存储，对 nominal, string 和 relational 属性以其 index 存储。
```java
DenseInstance(int numAttributes)
```

以 weight 1.0 和 all missing values 创建 `DenseInstance`。

这个构造函数使用可能更方便，但是设置属性值操作比较昂贵，特别是添加大量 row 的情况。因此**建议使用第一个构造函数**。

**创建数据**

对每个 `Instance`，第一步是创建存储属性值的 `double[]` 数组。一定不要重用该数组，而是创建新数组。因为在实例化 `DenseInstance` 时，Weka 只是引用，而不是创建它的副本。重用意味着改变之前生成的 `DenseInstance` 对象。

创建数组：

```java
double[] values = new double[data.numAttributes()];
```

为 `double[]` 填充值：

- **numeric**

```java
values[0] = 1.23;
```

- **date**

需将其转换为 double 值：

```java
values[1] = data.attribute(1).parseDate("2001-11-09");
```

- **nominal**

采用其索引值：

```java
values[2] = data.attribute(2).indexOf("label_b");
```

- **string**

使用 `addStringValue` 方法确定字符串的索引（Weka 内部使用哈希表保存所有字符串）：

```java
values[3] = data.attribute(3).addStringValue("This is a string");
```

- **relational**

首先，必须根据属性的关系定义创建一个新的 `Instances` 对象，然后才能使用 `addRelation` 确定它的索引：

```java	
Instances dataRel = new Instances(data.attribute(4).relation(),0);
valuesRel = new double[dataRel.numAttributes()];
valuesRel[0] = 2.34;
valuesRel[1] = dataRel.attribute(1).indexOf("val_C");
dataRel.add(new DenseInstance(1.0, valuesRel));
values[4] = data.attribute(4).addRelation(dataRel);
```

最后，使用初始化的 double 数组创建 Instance 对象：

```java
Instance inst = new DenseInstance(1.0, values);
data.add(inst);
```

### 示例

```java
public class CreateInstances {

    /**
     * 生成不同类型的 Instance
     * Generates the Instances object and outputs it in ARFF format to stdout.
     */
    public static void main(String[] args) throws Exception {
        // 1. 配置属性
        ArrayList<Attribute> atts = new ArrayList<>();
        // - numeric
        atts.add(new Attribute("att1"));
        // - nominal
        ArrayList<String> attVals = new ArrayList<>();
        for (int i = 0; i < 5; i++)
            attVals.add("val" + (i + 1));
        atts.add(new Attribute("att2", attVals));
        // - string
        atts.add(new Attribute("att3", (ArrayList<String>) null));
        // - date
        atts.add(new Attribute("att4", "yyyy-MM-dd"));
        // - relational
        ArrayList<Attribute> attsRel = new ArrayList<Attribute>();
        // -- numeric
        attsRel.add(new Attribute("att5.1"));
        // -- nominal
        ArrayList<String> attValsRel = new ArrayList<String>();
        for (int i = 0; i < 5; i++)
            attValsRel.add("val5." + (i + 1));
        attsRel.add(new Attribute("att5.2", attValsRel));
        Instances dataRel = new Instances("att5", attsRel, 0);
        atts.add(new Attribute("att5", dataRel, 0));

        // 2. 创建 Instances 对象
        Instances data = new Instances("MyRelation", atts, 0);

        // 3. 填充数据
        // first instance
        double[] vals = new double[data.numAttributes()];
        // - numeric
        vals[0] = Math.PI;
        // - nominal
        vals[1] = attVals.indexOf("val3");
        // - string
        vals[2] = data.attribute(2).addStringValue("This is a string!");
        // - date
        vals[3] = data.attribute(3).parseDate("2001-11-09");
        // - relational
        dataRel = new Instances(data.attribute(4).relation(), 0);
        // -- first instance
        double[] valsRel = new double[2];
        valsRel[0] = Math.PI + 1;
        valsRel[1] = attValsRel.indexOf("val5.3");
        dataRel.add(new DenseInstance(1.0, valsRel));
        // -- second instance
        valsRel = new double[2];
        valsRel[0] = Math.PI + 2;
        valsRel[1] = attValsRel.indexOf("val5.2");
        dataRel.add(new DenseInstance(1.0, valsRel));
        vals[4] = data.attribute(4).addRelation(dataRel);
        // add
        data.add(new DenseInstance(1.0, vals));

        // second instance
        vals = new double[data.numAttributes()];  // important: needs NEW array!
        // - numeric
        vals[0] = Math.E;
        // - nominal
        vals[1] = attVals.indexOf("val1");
        // - string
        vals[2] = data.attribute(2).addStringValue("And another one!");
        // - date
        vals[3] = data.attribute(3).parseDate("2000-12-01");
        // - relational
        dataRel = new Instances(data.attribute(4).relation(), 0);
        // -- first instance
        valsRel = new double[2];
        valsRel[0] = Math.E + 1;
        valsRel[1] = attValsRel.indexOf("val5.4");
        dataRel.add(new DenseInstance(1.0, valsRel));
        // -- second instance
        valsRel = new double[2];
        valsRel[0] = Math.E + 2;
        valsRel[1] = attValsRel.indexOf("val5.1");
        dataRel.add(new DenseInstance(1.0, valsRel));
        vals[4] = data.attribute(4).addRelation(dataRel);
        // add
        data.add(new DenseInstance(1.0, vals));

        // 4. output data
        System.out.println(data);
    }
}
```

```
@relation MyRelation

@attribute att1 numeric
@attribute att2 {val1,val2,val3,val4,val5}
@attribute att3 string
@attribute att4 date yyyy-MM-dd
@attribute att5 relational
@attribute att5.1 numeric
@attribute att5.2 {val5.1,val5.2,val5.3,val5.4,val5.5}
@end att5

@data
3.141593,val3,'This is a string!',2001-11-09,'4.141593,val5.3\n5.141593,val5.2'
2.718282,val1,'And another one!',2000-12-01,'3.718282,val5.4\n4.718282,val5.1'
```

## 4. 生成人工数据


使用 Weka 的数据生成器可以生成人工数据集。有两种方式。

### 生成 ARFF 文件

使用静态方法 `DataGenerator.makeData` 生成 ARFF 文件。

示例：

```java
import weka.datagenerators.DataGenerator;
import weka.datagenerators.classifiers.classification.RDG1;
...
// configure generator
RDG1 generator = new RDG1();
generator.setMaxRuleSize(5);
// set where to write output to
java.io.PrintWriter output = new java.io.PrintWriter(
    new java.io.BufferedWriter(new java.io.FileWriter("rdg1.arff")));
generator.setOutput(output);
DataGenerator.makeData(generator, generator.getOptions());
output.flush();
output.close();
```

### 生成 Instances

根据生成器不同，可以逐个生成 `Instance`，也可以生成整个数据集 `Instances`。

**示例：** 使用 Agrawal 生成器

```java
import weka.datagenerators.classifiers.classification.Agrawal;
...
// configure data generator
Agrawal generator = new Agrawal();
generator.setBalanceClass(true);
// initialize dataset and get header
generator.setDatasetFormat(generator.defineDataFormat());
Instances header = generator.getDatasetFormat();
// generate data
if (generator.getSingleModeFlag()) {
    for (int i = 0; i < generator.getNumExamplesAct(); i++) {
        Instance inst = generator.generateExample();
    }
} else {
    Instances data = generator.generateExamples();
}
```

## 5. 数据随机化

由于机器学习算法可能受数据顺序影响，对数据进行随机化是缓解此问题的常用方法。特别是重复随机化，如在交叉验证期间，有助于生成更真实的统计量。

Weka 提供了两种随机化数据集的方式：

- 使用 `weka.core.Instances` 对象的 `randomize(Random)` 方法。该方法需要 `java.util.Random` 实例。
- 使用 `Randomize` 过滤器

机器学习试验一个非常重要的方面，就是可重复。相同设置的多次运行，必须产生完全相同的结果。在这种情况下仍然可能随机化。随机数生成器不会返回一个完全随机的数字序列，而是伪随机数。为了实现可重复的随机序列，使用 seed 生成器。相同的 seed 总会得到相同的序列。

所以不要使用 `java.util.Random` 的默认构造函数，而使用 `Random(long)` 构造函数，指定 seed 值。

为了获得更依赖于数据集的随机随机化，可以使用 `weka.core.Instances` 的 `getRandomNumberGenerator(int)` 方法。该方法返回一个 `java.util.Random` 对象，其 seed 为提供的 seed 和从 `Instances` 中随机选择的 `weka.core.Instance`的 hashcode 的加和。

所以随机化数据可以分为两步：

1. 获得随机数生成器

```java
Instances dataSet = source.getDataSet();

Random random = data.getRandomNumberGenerator(1);
```

2. 随机化数据

```java
dataSet.randomize(random);
```

或者合并为一步：

```java
dataset.randomize(new Random(1));
```


## 6. 过滤

过滤器用于**预处理数据**。所有过滤器都在 `weka.filters` 包中，分为两类：

- 监督和非监督：是否需要 class 属性
- 基于特征和基于实例：对特征（column）还是对样本（row）的操作

监督过滤器考虑类别及其在数据集中的分布，以确定最佳的 bin 数和 bin-size；而无监督过滤器依赖于用户指定的 bin 数。

除了以上分类，过滤器有基于流和基于批两种模式：

- 流过滤器可以直接处理数据
- 批过滤器需要一批数据来初始化内部状态

`Add` 过滤器是一个流过滤器，添加只有缺失值的新属性不需要任何复杂设置。但是`ReplaceMissingValues`过滤器首先需要一批数据来确定每个属性的均值和众数，否则无法将缺失值替换为有意义的值，完成初始化后，就可以和流过滤器一样逐行处理数据。

**基于实例**的过滤器处理数据的方式有点特殊。如前所述，在获得第一批数据后，所有过滤器都可以逐行处理数据。在批处理模式，过滤器会添加或删除 rows；在单行处理模式，过滤器不工作。这是有原因的，以 `FilteredClassifier` 分类器为例：在训练后（第一批数据），分类器在测试集上进行评估，每次一个 row，如果此时过滤器添加或删除 row，就无法正确评估了。所以基于实例的过滤器不处理后序样本。`Resample` 就是这类过滤器。

`wekaexamples.filters` 中包含许多过滤器的示例类。

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

一定要在调用 `setInputFormat` 之前设置选项，因为该方法通常用于确定数据的输出格式，因此必须在调用之前设置好所有选项。

> [!NOTE]
>
> 由于已经有 meta-classifier，单独应用过滤器不是很有必要。为了完整性，这里单独介绍。   

### 批过滤

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

### 动态过滤

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

### 调用约定

方法调用顺序：

1. `setOptions`
2. `setInputFormat`
3. `Filter.useFilter(Instances,Filter)`

> [!IMPORTANT]
>
> `setInputFormat` 必须是应用过滤器前最后一个调用。因为许多过滤器在 `setInputFormat` 中根据当前设置选项生成输出格式的标题。所以在调用 `setInputFormat` 之前，应该完成所有设置选项。

### Add 过滤器

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

## 7. 分类

**分类和回归**算法在 WEKA 中统称为分类器，相关类在 `weka.classifiers` 包中。下面介绍：

- 构建分类器：批量和增量学习
- 评估分类器：各种评估技术以及如何获得统计量
- 执行分类：预测未知数据的类别

### 构建分类器

weka 的所有分类器都可以批训练，即一次性在整个数据集上训练。如果内存能够容纳训练集，该方式很好；如果数据量太大，也有部分分类器支持增量训练。

#### 批分类器

构建批分类器非常简单：

- 设置选项：使用 `setOptions(String[])` 或 setter 方法
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

#### 增量分类器

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

### 评估分类器

weka 支持两类评估方法：

- 交叉验证（cross-validation）：只有训练集没有测试集。如果 folds 数等于数据集的样本数，得到 leave-one-out cross-validation (LOOCV)；
- 专门的测试集：测试集仅用于评估分类器。

评估步骤，包括统计信息的收集，由 `Evaluation` 类执行。

#### 交叉验证

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

> [!NOTE]
>
> 传递给 `crossValidateModel` 方法的模型必须**没有经过训练**。因为，如果分类器不遵守 weka 规则，在 `buildClassifier` 时没有重置分类器，交叉验证会得到不一致的结果。`crossValidateModel` 在训练和评估分类器时，会创建原分类器的副本，因为不存在该问题。    

#### 测试集

使用专门的测试集评估模型。此时，需要提供训练过的分类器，而不是未经训练的分类器。

依然使用 `weka.classifiers.Evaluation` 类，调用 `evaluateModel` 方法。

**示例**：在训练集上使用默认选项训练 J48，然后在测试集上评估模型。

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

#### 统计量

除了前面使用的 `Evaluation` 类的 `toSummaryString` 方法，还有其它输出统计结果的方法。

**nominal class 属性统计方法**

- `toMatrixString`：输出混淆矩阵
- `toClassDetailsString`：输出 TP/FP rates, precision, recall, F-measure, AUC
- `toCumulativeMarginDistributionString`：输出累计 margins distribution

如果不想使用这些汇总方法，也可以访问单个统计量。

**nominal class 属性**

- `correct()`：正确分类样本数
- `incorrect()`： 错误分类样本数
- `pctCorrect()`：正确分类样本百分比
- `pctIncorrect()`：错误分类样本百分比
- `areaUnderROC(int)`：指定 class label index 的 AUC
- `kappa()`：Kappa 统计量

**numeric class 属性**

- `correlationCoefficient()`：相关系数

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

#### 收集预测结果

在评估分类器时，还可以查看哪些实例被错误分类。该模式的超类是 `weka.classifiers.evaluation.output.prediction.AbstractOutput`。

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

### 预测

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

## 8. 聚类


聚类（clustering）是在数据中寻找模式的无监督机器学习技术，这类算法不需要类别信息。下面介绍如下内容：

- 构建聚类器（cluster）：batch 和增量学习
- 评估聚类器
- 聚类新样本：确定新样本所属类别

`wekeexamples.clusters` 中有聚类的完整示例。

### 构建聚类器

聚类器和分类器一样，默认使用全部数据进行训练，即批训练。但是有少数聚类算法可以采用增量学习的方式更新内部表示。

#### 批聚类器

构建批聚类器分两步：

- 设置选项：调用 `setOptions(String[])` 或 setter 方法配置选项
- 构建模型：调用 `buildClusterer(Instances)` 训练模型。根据定义，使用相同数据重复调用该方法，必须获得相同模型（可重复性）。换句话说，调用该方法必须首先完全重置模型。

**示例：** 构建 EM 聚类器（最多迭代 100 次）

```java
Instances data = ... // from somewhere
String[] options = new String[2];
options[0] = "-I"; // max. iterations
options[1] = "100";
EM clusterer = new EM(); // new instance of clusterer
clusterer.setOptions(options); // set the options
clusterer.buildClusterer(data); // build the clusterer
```

#### 增量聚类器

增量聚类器实现 `UpdateableClusterer` 接口。训练增量聚类器分三步：

1. 初始化：调用 `buildClusterer(Instances)` 初始化模型。这里可以使用空的 `weka.core.Instances` 对象，或包含初始数据。
2. 更新：调用 `updateClusterer(Instance)` 逐个样本更新模型
3. 完成：调用 `updateFinished()` 完成模型。

**示例：** 使用 `ArffLoader` 迭代数据，增量构建 `Cobweb` 聚类器

```java
// 加载数据
ArffLoader loader = new ArffLoader();
loader.setFile(new File("/some/where/data.arff"));
Instances structure = loader.getStructure();

// 训练 Cobweb
Cobweb cw = new Cobweb();
cw.buildClusterer(structure);
Instance current;
while ((current = loader.getNextInstance(structure)) != null)
    cw.updateClusterer(current);
cw.updateFinished();
```

### 评估聚类器

聚类的评估不如分类那么全面。由于聚类是无监督的，所以很难确定一个模型有多好。使用 `ClusterEvaluation` 评估聚类模型。

**示例：** 查看 cluster 数目

```java
Clusterer clusterer = new EM();     // new clusterer instance, default options
clusterer.buildClusterer(data);     // build clusterer

ClusterEvaluation eval = new ClusterEvaluation();
eval.setClusterer(clusterer);       // the cluster to evaluate
eval.evaluateClusterer(newData);     // data to evaluate the clusterer on
System.out.println("# of clusters: " + eval.getNumClusters());  // output # of clusters
```

**示例：** 基于密度的聚类器的交叉验证，获得对数似然

基于密度的聚类器，即实现 `DensityBasedClusterer` 接口的算法，可以交叉验证，获得 log-likelyhood。`MakeDensityBasedClusterer` 可以将不是基于密度的聚类器转换为这类聚类器。

```java
Instances data = ... // from somewhere
DensityBasedClusterer clusterer = new ... // the clusterer to evaluate
double logLikelyhood = ClusterEvaluation.crossValidateModel( // cross-validate
            clusterer, data, 
            10, // with 10 folds
            new Random(1));
```

**示例：** 生成和 Explorer 或命令行一样的输出。

```java
String[] options = new String[2];
options[0] = "-t";
options[1] = "/some/where/somefile.arff";
System.out.println(ClusterEvaluation.evaluateClusterer(new EM(), options));
```

**示例：** 如果数据集已载入内存，可以使用如下方式

```java
Instances data = ... // from somewhere
EM cl = new EM();
cl.buildClusterer(data);
ClusterEvaluation eval = new ClusterEvaluation();
eval.setClusterer(cl);
eval.evaluateClusterer(new Instances(data));
System.out.println(eval.clusterResultsToString());
```

#### classes to clusters

如果数据集包含 class 属性，那么可以分析生成的 clusters 与 classes 的匹配情况。这种评估方式称为 **classes-to-clusters**。

这种方式的评估流程如下：

1. 创建包含 class 属性的数据集副本，使用 `Remove` 过滤器删除 class 属性
2. 使用新数据构建聚类器
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

### 样本聚类

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


## 9. 特征选择

合理准备数据对训练模型非常重要。减少属性数量不仅有助于加快模型训练，还有助于避免无关属性的干扰。

目前 weka 有三种属性评估器：

- 单属性评估器：`weka.attributeSelection.AttributeEvaluator` 接口对应这类属性评估器，`Ranker` 检索算法通常与这类算法结合使用。
- 属性子集评估器：`weka.attributeSelection.SubsetEvaluator` 接口，对属性子集进行评估。
- 属性集评估器：评估属性集合，`weka.attributeSelection.AttributeSetEvaluator` 接口实现类。

目前实现的大多数属性选择方案实现都是有监督的，即需要带 class 属性的数据集。

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

### 元分类器

元分类器 `AttributeSelectedClassifier` 与 `FilteredClassifier` 类似。但是，`AttributeSelectedClassifier` 不是基于分类器或过滤器执行过滤，而是使用搜索算法（派生自 `weka.attributeSelection.ASEvaluation`）和评估器（派生自 `weka.attributeSelection.ASSearch`）执行属性选择，然后使用 base 分类器对简化后的数据进行训练。

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

### 过滤器

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

### 直接使用 API

使用元分类器或过滤器选择属性很容易，但可能无法满足所有人的需求。例如，想获取属性排序（使用 `Ranker`）或检索所选属性的索引，而不是简化数据。

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

## 10. 可视化

`wekaexamples.gui`  包含可视化相关示例。

### ROC 曲线

weka 根据分类器评估结果生成 ROC 曲线。显示 ROC 曲线，需要如下步骤：

1. 根据 `Evaluation` 收集的预测结果，使用 `ThresholdCurve` 类生成绘图所需数据
2. 将绘图数据放入 `PlotData2D` 类
3. 将 `PlotData2D` 放入数据可视化面板 `ThresholdVisualizePanel` 类中
4. 将可视化面板放入 `JFrame` 中

实际代码：

1. 生成绘图数据

```java
Evaluation eval = ... // from somewhere
ThresholdCurve tc = new ThresholdCurve();
int classIndex = 0; // ROC for the 1st class label
Instances curve = tc.getCurve(eval.predictions(), classIndex);
```

2. 创建 `PlotData2D` 类

```java
PlotData2D plotdata = new PlotData2D(curve);
plotdata.setPlotName(curve.relationName());
plotdata.addInstanceNumberAttribute();
```

3. 创建面板

```java
ThresholdVisualizePanel tvp = new ThresholdVisualizePanel();
tvp.setROCString("(Area under ROC = " +
    Utils.doubleToString(ThresholdCurve.getROCArea(curve),4)+")");
tvp.setName(curve.relationName());
tvp.addPlot(plotdata);
```

4. 将面板加入 `JFrame`

```java
final JFrame jf = new JFrame("WEKA ROC: " + tvp.getName());
jf.setSize(500,400);
jf.getContentPane().setLayout(new BorderLayout());
jf.getContentPane().add(tvp, BorderLayout.CENTER);
jf.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
jf.setVisible(true);
```

### Graph

实现 `weka.core.Drawable` 接口的类可以生成显示内部模型的图形。目前有两种类型的图：

- Tree：决策树
- BayesNet：贝叶斯网络

#### Tree

显示 J48 和 M5P 等决策树的内部结构非常容易。

**示例：** 构建 J48 分类器，使用 `TreeVisualizer` 类显示决策树结构。

```java
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.gui.treevisualizer.PlaceNode2;
import weka.gui.treevisualizer.TreeVisualizer;
import java.awt.BorderLayout;
import javax.swing.JFrame;
...
Instances data = ... // from somewhere
// train classifier
J48 cls = new J48();
cls.buildClassifier(data);
// display tree
TreeVisualizer tv = new TreeVisualizer(
    null, cls.graph(), new PlaceNode2());
JFrame jf = new JFrame("Weka Classifier Tree Visualizer: J48");
jf.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
jf.setSize(800, 600);
jf.getContentPane().setLayout(new BorderLayout());
jf.getContentPane().add(tv, BorderLayout.CENTER);
jf.setVisible(true);
// adjust tree
tv.fitToScreen();
```

#### BayesNet

`BayesNet` 分类器生成的图形可以用 `GraphVisualizer` 显示。`GraphVisualizer`可以显示 `GraphViz` 的 DOT 语言 或 XML BIF 格式的 graphs：

- 对 DOT 格式，需要调用 readDOT
- 对 BIF 格式，需要调用 readBIF

**示例：** 训练 BayesNet 分类器，然后显示 graph 结构 

```java
import weka.classifiers.bayes.BayesNet;
import weka.core.Instances;
import weka.gui.graphvisualizer.GraphVisualizer;
import java.awt.BorderLayout;
import javax.swing.JFrame;
...
Instances data = ... // from somewhere
// train classifier
BayesNet cls = new BayesNet();
cls.buildClassifier(data);
// display graph
GraphVisualizer gv = new GraphVisualizer();
gv.readBIF(cls.graph());
JFrame jf = new JFrame("BayesNet graph");
jf.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
jf.setSize(800, 600);
jf.getContentPane().setLayout(new BorderLayout());
jf.getContentPane().add(gv, BorderLayout.CENTER);
jf.setVisible(true);
// layout graph
gv.layoutGraph();
```

## 11. 序列化

下面介绍如何序列化和反序列化一个 J48 分类器。

### 序列化分类器

`weka.core.SerializationHelper` 的 `write` 用于序列化：

```java
// 加载数据
Instances inst = DataSource.read("/some/where/data.arff");
inst.setClassIndex(inst.numAttributes() - 1);
// train J48
Classifier cls = new J48();
cls.buildClassifier(inst);
// serialize model
SerializationHelper.write("/some/where/j48.model", cls);
```

### 反序列化分类器

```java
// deserialize model
Classifier cls = (Classifier) SerializationHelper.read(
    "/some/where/j48.model");
```

### 反序列化 Explorer 保存的分类器

Explorer 保存的模型文件包含模型和训练集标头信息。通过检查数据集信息，可以很容易判断序列化的分类器是否适合当前数据集。`readAll` 方法返回文件中包含对象的数组：

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

### 序列化 Explorer 模型

如果希望和 Explorer 一样，在序列化分类器时包含数据集的标题信息，可以使用 `writeAll` 方法：

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

## 参考

-  https://waikato.github.io/weka-wiki/use_weka_in_your_java_code/