# 保存数据

- [保存数据](#保存数据)
  - [简介](#简介)
  - [保存到文件](#保存到文件)
  - [保存到数据库](#保存到数据库)

2023-12-12, 19:14
****

## 简介

保存 `weka.core.Instances` 与读取数据一样简单。

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

**示例**：显式调用 CSVSaver 保存到 CSV 文件

```java
import weka.core.Instances;
import weka.core.converters.CSVSaver;
import java.io.File;
...
// data structure to save
Instances data = ...
// save as CSV
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