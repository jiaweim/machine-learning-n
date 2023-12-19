# Weka API
- [Weka API](#weka-api)
  - [总结](#总结)
  - [简介](#简介)
  - [参数设置](#参数设置)
    - [setOptions](#setoptions)
      - [String\[\] 数组](#string-数组)
      - [String 字符串](#string-字符串)
      - [OptionsToCode](#optionstocode)
    - [get/set](#getset)

2023-12-13, 09:21
****

## 总结

- [数据随机化](data_randomizing.md)
- [数据预处理](data_filter.md)
- [分类](classification.md)
- [模型序列化](serialization.md)

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

`weka.core.OptionHandler` 接口提供了如下两个方法：

```java
void setOptions(String[] options)
String[] getOptions()
```

分类器、聚类器和过滤器均实现了该接口。

#### String[] 数组

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

使用 `OptionsToCode.java` 将命令自动转换为代码，当命令包含带选项的嵌套类时，会非常有用。例如 SMO 内核：

```java
java OptionsToCode weka.classifiers.functions.SMO
```

会生成：

```
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
