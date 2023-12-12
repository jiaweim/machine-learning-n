# 创建数据集

- [创建数据集](#创建数据集)
  - [简介](#简介)
  - [定义格式](#定义格式)
    - [numeric](#numeric)
    - [date](#date)
    - [nominal](#nominal)
    - [string](#string)
    - [relational](#relational)
  - [添加数据](#添加数据)
    - [构造函数](#构造函数)
    - [创建数据](#创建数据)
  - [示例](#示例)

2023-12-12, 10:40
@author Jiawei Mao
****

## 简介

创建数据集 `weka.core.Instances` 对象分两步：

1. 通过设置属性定义数据格式
2. 逐行添加数据

## 定义格式

WEKA 支持 5 种类型的数据。

|类型|描述|
|---|---|
|numeric|连续数值变量|
|date|日期|
|nominal|预定义标签|
|string|文本数据|
|relational|关系数据|

对这些数据，同一用 `weka.core.Attribute` 类定义，只是构造函数不同。

### numeric

最简单的属性类型，只需要提供属性名称：

```java
Attribute numeric = new Attribute("name_of_attr");
```

### date

日期属性在内部以数字处理，但是为了正确解析和显示日期，需要指定日期格式。具体格式可参考 `java.text.SimpleDateFormat` 类。例如：

```java
Attribute date = new Attribute("name_of_attr", "yyyy-MM-dd");
```

### nominal

nominal 属性包含**预定义标签**，需要以 `java.util.ArrayList<String>` 提供，例如：

```java
ArrayList<String> labels = new ArrayList<String>();
labels.addElement("label_a");
labels.addElement("label_b");
labels.addElement("label_c");
labels.addElement("label_d");

Attribute nominal = new Attribute("name_of_attr", labels);
```

### string

与 nominal 属性不同，此类型没有预定义标签。构造函数与创建 nominal 属性相同，但是后面的 `java.util.ArrayList<String>` 提供 null 值：

```java
Attribute string = new Attribute("name_of_attr", (ArrayList<String>)null);
```

### relational

这类属性使用另一个 `weka.core.Instances` 对象定义关系。例如，下面生成一个包含 numeric 属性和 nominal 属性之间的关系属性：

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

`Instances` 构造函数的最后一个参数指定初始大小。

## 添加数据

定义好数据集结构后，可以开始添加数据。

- `weka.core.Instance` 接口表示一个数据实例
- `weka.core.AbstractInstance` 实现该接口，提供了 `weka.core.DenseInstance` 和 `weka.core.SparseInstance` 的共有功能
  - `DenseInstance` 表示常规样本
  - `SparseInstance` 则只保存非 0 值

下面主要说明 `DenseInstance` 的使用，`SparseInstance` 的使用方法类似。

### 构造函数

`DenseInstance` 有两个构造函数：

```java
DenseInstance(double weight, double[] attValues)
```

以指定 weight 值和 double 数组创建 `DenseInstance`。WEKA 对所有属性类型都采用 double 存储，对 nominal, string 和 relational 属性以其 index 存储。
	
```java
DenseInstance(int numAttributes)
```

以 weight 1.0 和 all missing values 创建 `DenseInstance`。

这个构造函数使用可能更方便，但是设置属性值操作比较昂贵，特别是添加大量 row 的情况。因此建议使用第一个构造函数。

### 创建数据

对每个 `Instance`，第一步是创建存储属性值的 double[] 数组。一定不要重用该数组，而是创建新数组。因为在实例化 `DenseInstance` 时，Weka 只是引用，而不是创建它的副本。重用意味着改变之前生成的 `DenseInstance` 对象。

创建数组：

```java
double[] values = new double[data.numAttributes()];
```

为 double[] 填充值：

**numeric**

```java
values[0] = 1.23;
```

**date**

需将其转换为 double 值：

```java
values[1] = data.attribute(1).parseDate("2001-11-09");
```

**nominal**

采用其索引值：

```java
values[2] = data.attribute(2).indexOf("label_b");
```

**string**

使用 addStringValue 方法确定字符串的索引（Weka 内部使用哈希表保存所有字符串）：

```java
values[3] = data.attribute(3).addStringValue("This is a string");
```

**relational**

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

## 示例

```java
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instances;

import java.util.ArrayList;

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
