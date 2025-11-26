# 数据预处理

2025-11-26⭐
@author Jiawei Mao
***
## 简介

在 ndarray 中已经介绍多种处理 `NDArra` 数据的技术。为了将深度学习应用于现实问题，我们通常需要从预处理原始数据开始，而不是整理好的 `NDArray` 数据。在流行的 Java 数据分析工具中，tablesaw 被广泛使用，它与 Python 的 pandas 类似。下面简要介绍使用 tablesaw 预处理原始数据并将其转换为 `NDArray` 格式的步骤。在后面会介绍更多数据预处理技术。

## 添加依赖项

```xml
<dependency>
    <groupId>tech.tablesaw</groupId>
    <artifactId>tablesaw-jsplot</artifactId>
    <version>0.44.4</version>
</dependency>
```

## 读取数据集

下面首先创建一个数据集，保存到 `../data/house_tiny.csv` 文件。其它格式的数据处理方法类似。

```java
File file = new File("../data/");
file.mkdir();

String dataFile = "../data/house_tiny.csv";
File f = new File(dataFile);
f.createNewFile();

try (FileWriter fw = new FileWriter(dataFile)) {
    fw.write("NumRooms,Alley,Price\n"); // Column names
    fw.write("NA,Pave,127500\n");  // Each row represents a data example
    fw.write("2,NA,106000\n");
    fw.write("4,NA,178100\n");
    fw.write("NA,NA,140000\n");
}
```

使用 tablesaw 读取 csv 文件。该数据集包含 4 行 3 列。

```java
Table data = Table.read().file("../data/house_tiny.csv");
System.out.println(data);
```

```
 NumRooms  |  Alley  |  Price   |
---------------------------------
           |   Pave  |  127500  |
        2  |         |  106000  |
        4  |         |  178100  |
           |         |  140000  |
```

## 处理缺失值

上面数据有一些缺失值，处理缺失值的典型方法包括插入（imputation）和删除（deletion）。插入将指定值替换缺失值，而删除则忽略缺失值。这里使用插入策略。

下面通过创建新 tables 将 `data` 拆分为 `inputs` 和 `outputs`，前者包含前两列，后者包含最后 一列。对缺失的数值类型，用同一列的平均值替换缺失值。

```java
Table inputs = data.create(data.columns());
inputs.removeColumns("Price");
Table outputs = data.selectColumns("Price");

Column col = inputs.column("NumRooms");
col.set(col.isMissing(), (int) inputs.nCol("NumRooms").mean());

System.out.println(inputs);
```

```
 NumRooms  |  Alley  |
----------------------
        3  |   Pave  |
        2  |         |
        4  |         |
        3  |         |
```

对分类值或离散值，我们将缺失值或 null 视为一个类别。由于 "Alley" 列只有两种分类值 "Pave" 和 null，tablesaw 可以自动将其转换为两列，我们将这两列分别命名为 "Alley_Pave" 和 "Alley_nan"。之后，将这两列添加到原始数据中，并转换为 double 类型，并去掉原来的 "Alley" 列。

```java
StringColumn alleyCol = (StringColumn) inputs.column("Alley");
List<BooleanColumn> dummies = alleyCol.getDummies();
inputs.removeColumns(alleyCol);
inputs.addColumns(
        DoubleColumn.create("Alley_Pave", dummies.get(0).asDoubleArray()),
        DoubleColumn.create("Alley_nan", dummies.get(1).asDoubleArray())
);
System.out.println(inputs);
```

```
 NumRooms  |  Alley_Pave  |  Alley_nan  |
-----------------------------------------
        3  |           1  |          0  |
        2  |           0  |          1  |
        4  |           0  |          1  |
        3  |           0  |          1  |
```

## 转换为 NDArray 格式

现在 `inputs` 和 `outputs` 的数据都是数字，可以转换为 `NDArray` 格式。

```java
try (NDManager nd = NDManager.newBaseManager()) {
    NDArray x = nd.create(inputs.as().doubleMatrix());
    NDArray y = nd.create(outputs.as().doubleMatrix());
    System.out.println(x);
    System.out.println(y);
}
```

```
ND: (4, 3) gpu(0) float64
[[3., 1., 0.],
 [2., 0., 1.],
 [4., 0., 1.],
 [3., 0., 1.],
]

ND: (4, 1) gpu(0) float64
[[127500.],
 [106000.],
 [178100.],
 [140000.],
]
```

## 总结

- 与  Java 生态的许多扩展包一样，tablesaw 可以与 `NDArray` 一起使用
- 插入和删除可用于处理缺失数据



 