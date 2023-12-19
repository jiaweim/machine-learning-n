# 数据读写

- [数据读写](#数据读写)
  - [简介](#简介)
  - [从文件加载数据](#从文件加载数据)
    - [ARFF 文件](#arff-文件)
    - [CSV 文件](#csv-文件)
    - [设置分类属性](#设置分类属性)
  - [从数据库加载数据](#从数据库加载数据)
  - [保存到文件](#保存到文件)
    - [保存为 ARFF](#保存为-arff)
    - [保存为 CSV](#保存为-csv)
  - [保存到数据库](#保存到数据库)

2023-12-12, 10:00
@author Jiawei Mao
****

## 简介

在应用过滤器、分类器或聚类之前，首先需要有数据。WEKA 支持从文件和数据库加载数据。

Weka 使用如下类存储数据：

- `weka.core.Instances` - 表示数据集。该数据结构基于行
  - 使用 `instance(int)` 访问单行（0-based）
  - 使用 `attribute(int)` 返回 `weka.core.Attribute`，包含列的信息
- `weka.core.Instance` - 表示单行，即一个样本。它封装了一个 double 数组
  - `Instance` 不包含 column 的类型信息，需要访问 `weka.core.Instances` 获取（参考 `dataset` 和 `setDataset` 方法）
  - `weka.core.SparseInstance` 用于存储稀疏数据
- `weka.core.Attribute` - 保存单个 column 的类型信息，还包括 nominal 属性的标签，字符串属性的可能值，关系属性的数据集。

## 从文件加载数据

`weka.core.converters.ConverterUtils` 的内部类 `DataSource` 可以根据文件扩展名识别文件类型，然后调用特定的加载器读取数据。例如：

```java
Instances data1 = DataSource.read("/some/where/dataset.arff");
Instances data2 = DataSource.read("/some/where/dataset.csv");
Instances data3 = DataSource.read("/some/where/dataset.xrff");
```

### ARFF 文件

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
- `new Instances(dataSet, 0)` 以 `dataSet` 为模板创建新的空数据集，用来打印标题信息，否则还会输出所有样本数据。

### CSV 文件

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

### 设置分类属性

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

## 从数据库加载数据

使用以下两个类从数据库加载数据：

- `weka.experiment.InstanceQuery`：支持检索稀疏数据
- `weka.core.converters.DatabaseLoader`：可以增量检索数据

`InstanceQuery` 使用示例：

```java
import weka.core.Instances;
import weka.experiment.InstanceQuery;
...
InstanceQuery query = new InstanceQuery();
query.setDatabaseURL("jdbc_url");
query.setUsername("the_user");
query.setPassword("the_password");
query.setQuery("select * from whatsoever");
// if your data is sparse, then you can say so, too:
// query.setSparseData(true);
Instances data = query.retrieveInstances();
```

在批处理检索中使用 `DatabaseLoader` 的示例：

```java
import weka.core.Instances;
import weka.core.converters.DatabaseLoader;
...
DatabaseLoader loader = new DatabaseLoader();
loader.setSource("jdbc_url", "the_user", "the_password");
loader.setQuery("select * from whatsoever");
Instances data = loader.getDataSet();
```

在增量模式中使用 `DatabaseLoader` 的示例：

```java
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.DatabaseLoader;
...
DatabaseLoader loader = new DatabaseLoader();
loader.setSource("jdbc_url", "the_user", "the_password");
loader.setQuery("select * from whatsoever");
Instances structure = loader.getStructure();
Instances data = new Instances(structure);
Instance inst;
while ((inst = loader.getNextInstance(structure)) != null)
    data.add(inst);
```

!!! note
    - 并非所有数据库支持增量检索
    - 并非所有查询都有唯一键来增量检索 row。此时，可以使用 `setKeys(String)` 提供必要的 columns
    - 如果不能增量检索，则可以将其完全加载到内存


## 保存到文件

使用 `DataSink` （`weka.core.converters.ConverterUtils` 内部类）根据扩展名自动保存到指定文件：

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

### 保存为 ARFF

```java
Instances dataset = ...;
ArffSaver saver = new ArffSaver();
saver.setInstances(dataset);
saver.setFile(new File("out/path"));
saver.writeBatch();
```

### 保存为 CSV

**示例**：显式调用 `CSVSaver` 保存到 CSV 文件

```java
Instances data = ...
CSVSaver saver = new CSVSaver();
saver.setInstances(data);
saver.setFile(new File("/some/where/data.csv"));
saver.writeBatch();
```

## 保存到数据库

除了 KnowledgeFlow，在 weka 中将数据保存到数据库的功能不明显。和 DatabaseLoader 一样，保存数据也分为 batch 模式和增量模式。

**示例：** 批量模式，比较简单

```java
import weka.core.Instances;
import weka.core.converters.DatabaseSaver;
...
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
import weka.core.Instances;
import weka.core.converters.DatabaseSaver;
...
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