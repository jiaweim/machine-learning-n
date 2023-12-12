# 可视化

- [可视化](#可视化)
  - [简介](#简介)
  - [ROC 曲线](#roc-曲线)
  - [图](#图)
    - [Tree](#tree)
    - [BayesNet](#bayesnet)

2023-12-12, 19:40
****

## 简介

## ROC 曲线

weka 根据分类器评估结果生成 ROC 曲线。显示 ROC 曲线，需要如下步骤：

1. 根据 Evaluation 收集的预测结果，使用 ThresholdCurve 类生成绘图所需数据
2. 将绘图数据放入 PlotData2D 类
3. 将 PlotData2D 放入数据可视化面板 ThresholdVisualizePanel 类中
4. 将可视化面板放入 JFrame 中

实际代码：

1. 生成数据

```java
Evaluation eval = ... // from somewhere
ThresholdCurve tc = new ThresholdCurve();
int classIndex = 0; // ROC for the 1st class label
Instances curve = tc.getCurve(eval.predictions(), classIndex);
```

2. 创建 PlotData2D 类

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

4. 将面板加入 JFrame

```java
final JFrame jf = new JFrame("WEKA ROC: " + tvp.getName());
jf.setSize(500,400);
jf.getContentPane().setLayout(new BorderLayout());
jf.getContentPane().add(tvp, BorderLayout.CENTER);
jf.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
jf.setVisible(true);
```

## 图

实现 weka.core.Drawable 接口的类可以生成显示内部模型的图形。目前有两种类型的图：

- Tree：决策树
- BayesNet：贝叶斯网络

### Tree

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

### BayesNet

BayesNet 分类器生成的图形可以用 GraphVisualizer 显示。GraphVisualizer可以显示 GraphViz 的 DOT 语言 或 XML BIF 格式的 graphs：

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