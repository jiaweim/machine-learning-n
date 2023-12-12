# 生成人工数据

- [生成人工数据](#生成人工数据)
  - [简介](#简介)
  - [生成 ARFF 文件](#生成-arff-文件)
  - [生成 Instances](#生成-instances)

2023-12-12, 11:08
****

## 简介

使用 Weka 的数据生成器可以生成人工数据集。有两种方式。

## 生成 ARFF 文件

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

## 生成 Instances

根据生成器不同，可以逐个生成 Instance，也可以生成整个数据集 Instances。

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
