# LIBSVM 指南

2025-05-13⭐
@author Jiawei Mao
***

## 简介
支持向量机（SVMs）是一种用于数据分类的高性能监督学习算法。虽然 SVM 比神经网络简单，但是不熟悉 SVM 的用户开始上手时难以获得满意的结果。下面介绍如何正确使用 SVM 以获得合理的结果。

该文章不适合于 SVM 研究人员（过于简单），也无法保证结果准确性最高，只是方便新手入门。

虽然用户不需要了解 SVM 的基本理论，不过我们还是简单介绍一点基本知识，方便后续的解释说明。

### 分类问题
一个分类算法一般将数据分为训练集和测试集。训练集中的每个实例都包含一个目标值（或分类标签）和多个属性（也称为 features, observed variables）。SVM 的目标：基于训练集生成一个模型，然后使用该模型推断测试数据的目标值。

对训练数据集 $(X_i, y_i)$, $i=1,..,l$，其中 $X_i\in R^n$, $y\in \{1, -1\}^l$，SVM 需要解决如下的优化问题：
$$
\underset{w,b,\xi}{min} (\frac{1}{2}\bold{w}^T\bold{w}+C\displaystyle\sum_{i=1}^l\xi_i)
$$
且满足：
$$
y_i(w^T\varnothing(X_i)+b)\geq1-\xi_i
$$
其中：$\xi_i\geq0$。

在这里训练数据向量 $X_i$ 由函数 $\varnothing$ 映射到高维空间。SVM 在此高维空间中找到具有最大边界的线性分隔超平面。$C > 0$ 是误差项的**惩罚**参数。

另外，$K(X_i, X_j) \equiv \varnothing(X_i)^T\varnothing(X_j)$ 称为**核函数**（kernel）。四个最基本的核函数：
- 线性：$K(X_i, X_j)=X_i^TX_j$
- 多项式：$K(X_i, K_j)=(\gamma X_i^TX_j+r)^d$, $\gamma > 0$
- 径向基函数（Radial basic function ,RBF）: $K(X_i, X_j)=exp(-\gamma\lVert X_i-X_j\rVert^2)$, $\gamma > 0$
- S形函数（sigmoid）: $K(X_i, X_j)=tanh(\gamma X_i^TX_j+r)$

其中，$\gamma, r, d$ 为核函数的参数。

### 实例
下面是一个具体实例。数据集由我们用户提供，他们一开始无法获得合理的准确性。

|Applications|#training data|#testing data|#features|#classes|Accuracy by users|Accuracy by our procedure|
|---|---|---|----|----|---|---|
|Astroparticle|3,089|4,000|4|2|75.2%|96.9%|
|Bioinformatics|391|0|20|3|36%|85.2%|
|Vehicle|1,243|41|21|2|4.88%|87.8%|

通过该教程他们最终获得了较合理的准确性，[测试数据集位置](https://www.csie.ntu.edu.tw/~cjlin/papers/guide/data/)。

### 推荐使用流程
大多数初学者上手使用如下流程：
- 将数据转换为 SVM 包支持的格式；
- 随机尝试一些核函数和参数
- 测试

我们建议大家采用如下的**流程**：
1. 将数据转换为 SVM 包支持的格式

2. 对数据进行简单的缩放

3. 考虑使用 RBF 核函数 $K(x, y)=e^{-\gamma\lVert x-y\rVert^2}$

4. 通过交叉验证获得最佳的参数 $C$ 和 $\gamma$

5. 使用最佳的 $C$ 和 $\gamma$ 训练整个数据集

6. 测试

## 数据预处理
### 分类特征（Categorical Feature）
SVM 要求每个样本数据都以实数向量的形式表示。因此，如果有分类属性（categorical attribute），我们需要将它们转换为数值形式。

建议使用 m 个数值表示包含 m 个类别的属性：对一个样本，一个值为1，其它都是0（one hot）。例如，对包含三个分类的 {red, green, blue} 的属性，可以表示为 (0,0,1), (0,1,0), (1,0,0)。

根据以往的经验，如果属性的值不太大，这种编码比使用单个数字要稳定。

### 缩放
在使用 SVM 前对数据进行缩放十分重要。[Sarle 的神经网络FAQ](http://www.faqs.org/faqs/ai-faq/neural-nets/) 解释了这种做法的重要性，该解释大多数因素也适用于 SVM。

缩放（scaling）最大的优点是避免数值较大的特征产生比小数值特征更大的影响。且避免了计算过程的数值困难。因为内核函数通常依赖于特征向量的内积，如线性内核和多项式内核，因此属性值太大还可能导致数值问题。

建议：将属性值线性缩放到范围 $[-1,+1]$ 或 $[0, 1]$。

> [!IMPORTANT]
>
> 针对训练集和测试集必须使用相同的缩放规则。例如，如果将训练集的一个属性从 $[-10,+10]$ 缩放到 $[-1,+1]$，那么测试集数据该属性范围是 $[-11,+8]$，则必须缩放到 $[-1.1,+0.8]$。

## 选择模型
通用的内核只有四种，我们要先选择一个 kernel，然后选择惩罚参数 $C$ 和 kernel 参数。

### RBF 内核
通常首选 RBF 内核。该内核以**非线性**的形式将样本映射到高维空间，因此，相对线性内核，RBF可以处理类别和属性是非线性的情况。另外，线性内核只是 RBF 内核的一个特例，因为，具有惩罚参数 $\stackrel{\textasciitilde}{C}$ 的线性 kernel 与某些参数 $(C,\gamma)$ 的 RBF kernel 具有相同性能。另外，sigmoid kernel 在特定参数下和 RBF 的效果类似。

第二个原因是超参数的数目十分影响模型选择的复杂性。**多项式内核**比 RBF 内核超参数多很多。

最后，RBF内核具有较少的数值困难。相对RBF的 $0<K_{ij}\leq1$，多项式 kernel 值在次数较大时会出现无穷大 $(\gamma X_i^TX_j + r > 1)$ 或 0 $(\gamma X_i^TX_j + r < 1>)$ 的情况。此外，在某些参数下，sigmoid kernel 会出现无效情况（不是两个向量的内积）。

RBF 在某些情况也不合适，特别是特征数目特别多的时候，此时可以使用 linear kernel。

### 交叉验证和网格搜索（Grid-search）
RBF 内核有两个参数： $C$ 和 $\gamma$。对一个问题，我们无法提前知道 $C$ 和 $\gamma$ 的最佳取值，因此某些模型选择（parameter search）是必要的。其目的是获得较好的 $(C, \gamma)$ 参数值，使得分类器可以准确的预测未知数据（如测试数据）。

> [!NOTE]
>
> 在训练集上实现较高的 accuracy 是没有意义的。我们通常将训练集分为两部分，测试集被认为是未知的。只有在测试集上获得的预测精度才能反应模型的性能。

在 **v-fold** 交叉验证中，首先将训练集等分为 v 份。将一份作为测试集，余下的 $v-1$ 份作为训练数据，训练 $v$ 个模型。这样，每个样本都被推断一次，交叉验证的准确度就是被准确分类数据的百分比。

使用交叉验证可以避免过拟合问题。下面使用一个二分类问题解释过拟合问题。

<img src="images/2019-10-27-17-01-47.png" style="zoom:60%;" />

实心的圆和三角为训练数据，空心的圆和三角为测试数据。

Figures 1a, 1b 由于过拟合，在测试集上的准确率不好。而 1c, 1d 中由于没有过拟合训练数据，其交叉验证和测试准确率都更好。

我们推荐在交叉验证中使用 "grid-search" 寻找最佳的 $C$ 和 $\gamma$ 值。测试不同的 $(C, \gamma)$ 组合，选择交叉验证准确度最好的那一对。我们发现成指数的尝试 $C$ 和 $\gamma$ 值比较实用，例如 $C=2^{-5},2^{-3},...,2^{15}, \gamma=2^{-15},2^{-13},...,2^{3}$

grid-search 简单直接，看着很傻。实际上，有好几种可以节省计算成本的高级方法。不过依然推荐 grid-search，主要有两点原因：

- 首先，对那些通过近似或启发式方法避免完整枚举参数的方式不放心。
- 然后，因为只有两个参数，所以通过 grid-search 找到合适参数的时间不比那些高级方法多多少

并且，grid-search 很容易并行化，因为每一对参数都是相互独立的。而大多数高级方法都采用迭代的形式，比如沿着 path 优化，因此很难并行化。

由于实现完整的 grid-search 十分耗时，因此建议先采用粗粒度网格。在获得一个相对较好的网格区间，再采用细粒度搜索。为了更好的解释该方法，我们针对 Statlog collection 的 german 问题进行处理。如下图所示，首先采用粗粒度的网格检索$(C=2^{-5},2^{-3},...,2^{15}, \gamma=2^{-15},2^{-13},...,2^{3})$，发现最佳参数组合 $(C, \gamma)$ 为 $(2^3, 2^{-5})$，对应的交叉验证率为 77.5%。

<img src="images/2019-10-27-19-17-58.png" width="500" />

然后在 $(2^3, 2^{-5})$ 附近采用细粒度网格检索$(C=2^{1},2^{1.25},...,2^{5}, \gamma=2^{-7},2^{-6.75},...,2^{-3})$，在 $(2^{3.25}, 2^{-5.25})$ 获得更好的交叉验证率 77.6%.

<img src="images/2019-10-27-19-21-48.png" width="500" />

再用获得的最佳 $(C, \gamma)$ 参数重新对整个训练集进行训练，得到最终分类器。

以上方法对 1000+ 数据效果很好。对超大型数据集，则随机的从数据集中抽取部分数据进行网格搜索更可行。

## 讨论
在某些情况上，上面推荐的分析流程不好使，此时就需要采用其他的技术，比如**特征选择**（feature selection）。这些问题就有些超纲了，不在本节范围内。经验表面，该流程适用于特征不t太多的数据。如果有成千上万的属性，则在提供给 SVM 前，需要提取选择数据子集。

下面对比使用推荐的流程与一般初学者使用的流程所得模型的准确性。使用 LIVSVM 对 table 1 中提到的三个问题进行实验。对每个问题，首先通过直接训练和测试获得 accuracy。然后，展示 scaling 对 accuracy 的影响。根据前面讨论的内容可知，我们必须保存训练集中属性的范围，以对测试集应用相同操作。接着，使用推荐流程（scaling+模型选择）的 accuracy。最后，演示如何在 LIBSVM 自动执行整个过程。请注意，下面使用的 grid.py 在 R-LIBSVM 也有类似参数选择工具。

### 1. Astroparticle Physics

1. **原始数据集，默认参数**

```sh
SVMDataset trainSet = SVMDataset.read(new File("svmguide1"));
SVMDataset testSet = SVMDataset.read(new File("svmguide1.t"));

SVMParameter parameter = new SVMParameter();
SVMModel model = SVMUtils.train(trainSet, parameter);
double[] predicted = model.predict(testSet);
System.out.println("Accuracy = " +
        DecimalFormatUtils.F4.format(SVMUtils.getAccuracy(testSet.y, predicted) * 100) + "%");
```

```java
Accuracy = 66.9250%
```

2. **缩放数据集，默认参数**

```java
SVMDataset trainSet = SVMDataset.read(new File("svmguide1"));
SVMDataset testSet = SVMDataset.read(new File("svmguide1.t"));

Scaler scaler = Scaler.fit(trainSet);

SVMDataset trainScaled = scaler.apply(trainSet, -1, 1);
SVMDataset testScaled = scaler.apply(testSet, -1, 1);

SVMParameter parameter = new SVMParameter();
SVMModel model = SVMUtils.train(trainScaled, parameter);
double[] predicted = model.predict(testScaled);
System.out.println("Accuracy = " +
        DecimalFormatUtils.F4.format(SVMUtils.getAccuracy(testScaled.y, predicted) * 100) + "%");
```

```
Accuracy = 96.1500%
```

**3. 参数选择**

首先，使用 grid-search 检索参数：

```java
SVMDataset trainSet = SVMDataset.read(new File("svmguide1"));
Scaler scaler = Scaler.fit(trainSet);
SVMDataset trainScaled = scaler.apply(trainSet, -1, 1);

for (int c = -5; c <= 15; c += 2) {
    for (int g = -15; g <= 3; g += 2) {
        double C1 = Math.pow(2, c);
        double g1 = Math.pow(2, g);

        SVMParameter parameter = new SVMParameter();
        parameter.C = C1;
        parameter.gamma = g1;

        double[] out = SVMUtils.cvClassification(5, trainScaled, parameter);
        double accuracy = SVMUtils.getAccuracy(trainScaled.y, out);
        System.out.println(C1 + "\t" + g1 + "\t" + accuracy);
    }
}
```

```
2	2	0.970864357
512	0.125	0.970216899
2	8	0.969893169
8	2	0.969893169
512	0.5	0.96956944
...
```

C 值为正则化参数：

- C 值较大，则尽可能好的分离数据，优化器趋向于选择边距较小的超平面
- C 值较小，优化器会寻找分隔较大的超平面，即使该超平面会导致更多错误分类

Gamma 是 RBF kernel 的参数：

- gamma 值越小，决策边界越平滑，容易欠拟合
- gamma 值越大，决策边界越复杂，容易过拟合

> [!TIP]
>
> 在保证分类准确的前提下，C 和 gamma 都是越小越好。

可以发现，有许多参数组合性能近似，均在 96% 以上。这里采用值较小的参数组合，即取 $C=$, $g=2$。

然后用最佳参数组合训练模型，测试性能：

```java
SVMDataset trainSet = SVMDataset.read(new File("svmguide1"));
SVMDataset testSet = SVMDataset.read(new File("svmguide1.t"));

Scaler scaler = Scaler.fit(trainSet);

SVMDataset trainScaled = scaler.apply(trainSet, -1, 1);
SVMDataset testScaled = scaler.apply(testSet, -1, 1);

SVMParameter parameter = new SVMParameter();
parameter.gamma = 2;
parameter.C = 2;
SVMModel model = SVMUtils.train(trainScaled, parameter);
double[] predicted = model.predict(testScaled);
System.out.println("Accuracy = " +
        DecimalFormatUtils.F4.format(SVMUtils.getAccuracy(testScaled.y, predicted) * 100) + "%");
```

```
Accuracy = 96.8750%
```

### 2. Bioinformatics

该示例只有一个一个数据集，因此采用 CV 来评估。

**1. 原始数据集，默认参数**

```java
SVMDataset dataset = SVMDataset.read("train.2");
SVMParameter parameter = new SVMParameter();
double[] ys = SVMUtils.cvClassification(5, dataset, parameter);
double accuracy = SVMUtils.getAccuracy(dataset.y, ys);
System.out.println("Accuracy = " 
                   + DecimalFormatUtils.F4.format(accuracy * 100) + "%");
```

```
Accuracy = 56.5217%
```

**2. 缩放数据集，默认参数**

```java
SVMDataset dataset = SVMDataset.read("train.2");
Scaler scaler = Scaler.fit(dataset);
SVMDataset scaledDataset = scaler.apply(dataset, -1, 1);

SVMParameter parameter = new SVMParameter();
double[] ys = SVMUtils.cvClassification(5, scaledDataset, parameter);
double accuracy = SVMUtils.getAccuracy(scaledDataset.y, ys);
System.out.println("Accuracy = " 
                   + DecimalFormatUtils.F4.format(accuracy * 100) + "%");
```

```
Accuracy = 78.5166%
```

**3. 参数选择**

```java
SVMDataset dataset = SVMDataset.read("train.2");
Scaler scaler = Scaler.fit(dataset);
SVMDataset scaledDataset = scaler.apply(dataset, -1, 1);

GridSearch.create().grid(scaledDataset);
```

```
C	g	Accuracy
2.0	0.5	0.8542199488491049
128.0	0.03125	0.8439897698209718
8.0	0.125	0.8414322250639387
128.0	0.001953125	0.8337595907928389
512.0	0.0078125	0.8337595907928389
...
```

最佳参数为 $C=2.0$, $g=0.5$，准确度为 85.42%。

### 3. Vehicle

包含训练集和测试集。

**1. 原始数据集，默认参数**

```java
SVMDataset trainSet = SVMDataset.read("train.3");
SVMDataset testSet = SVMDataset.read("test.3");

SVMParameter parameter = new SVMParameter();
SVMModel model = SVMUtils.train(trainSet, parameter);
double[] predicted = model.predict(testSet);
System.out.println("Accuracy = " +
        DecimalFormatUtils.F4.format(SVMUtils.getAccuracy(testSet.y, predicted) * 100) + "%");
```

```
Accuracy = 2.4390%
```

**2. 缩放数据集，默认参数**

```java
SVMDataset trainSet = SVMDataset.read("train.3");
SVMDataset testSet = SVMDataset.read("test.3");

Scaler scaler = Scaler.fit(trainSet);

SVMDataset trainScaled = scaler.apply(trainSet, -1, 1);
SVMDataset testScaled = scaler.apply(testSet, -1, 1);

SVMParameter parameter = new SVMParameter();
SVMModel model = SVMUtils.train(trainScaled, parameter);
double[] predicted = model.predict(testScaled);
System.out.println("Accuracy = " +
        DecimalFormatUtils.F4.format(SVMUtils.getAccuracy(testScaled.y, predicted) * 100) + "%");
```

```
Accuracy = 12.1951%
```

**3. 参数选择**

```java
SVMDataset trainSet = SVMDataset.read("train.3");
Scaler scaler = Scaler.fit(trainSet);
SVMDataset trainScaled = scaler.apply(trainSet, -1, 1);

GridSearch.create().grid(trainScaled);
```

```
C	g	Accuracy
32768.0	0.0078125	0.8527755430410298
32.0	0.125	0.8439259855189059
128.0	0.03125	0.8431214802896219
2048.0	0.0078125	0.8415124698310539
8192.0	0.0078125	0.8399034593724859
8.0	0.125	0.8382944489139179
128.0	0.125	0.8382944489139179
512.0	0.03125	0.8358809332260659
512.0	0.0078125	0.835076427996782
2048.0	0.03125	0.83266291230893
2048.0	0.001953125	0.831858407079646
32768.0	4.8828125E-4	0.831858407079646
8192.0	4.8828125E-4	0.831053901850362
32768.0	0.001953125	0.831053901850362
...
```

靠前的准确性相差不大，这里采用较小的参数组合，$C=8$, $g=0.125$。

用该参数建模：

```java
SVMDataset trainSet = SVMDataset.read("train.3");
SVMDataset testSet = SVMDataset.read("test.3");

Scaler scaler = Scaler.fit(trainSet);

SVMDataset trainScaled = scaler.apply(trainSet, -1, 1);
SVMDataset testScaled = scaler.apply(testSet, -1, 1);

SVMParameter parameter = new SVMParameter();
parameter.C = 8;
parameter.gamma = 0.125;
SVMModel model = SVMUtils.train(trainScaled, parameter);
double[] predicted = model.predict(testScaled);
System.out.println("Accuracy = " +
        DecimalFormatUtils.F4.format(SVMUtils.getAccuracy(testScaled.y, predicted) * 100) + "%");
```

```
Accuracy = 73.1707%
```

用 $C=128$, $g=0.125$：

```
Accuracy = 87.8049%
```

> [!NOTE]
>
> CV 计算结果有一定随机性，对参数选择有一定影响。
>
> 所以，为了达到更好效果，是否应该用 CV 筛选出几组参数，然后分别训练模型，在测试集上，以最终效果为准？

## Linear vs RBF

当 **features 很多**，则可能不需要将数据映射到高维空间。也就是说，非线性映射可能不会提高性能，此时**使用 linear-kernel** 就足够了，因此只需要搜索参数 $C$。

虽然前面说了，RBF 至少与 linear-kernel 一样好，但有个前提：搜索到合适的 $(C,\gamma)$ 参数组合。

### 样本数 << feature 数

生物信息学中许多 microarray 数据都是这种类型，即样本数远小于 feature 数。以 [Leukemia](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/) 为例，training-set 和 testing-set 分别包含 38 和 34  个样本。Features 数为 7129，远大于样本数。下面将训练集和测试集合并，然后比较使用 RBF 和 linear 的 CV 准确性。

- RBF kernel 参数选择

```java
SVMDataset dataset = SVMDataset.read("leu.combined");
GridSearch.create().grid(dataset);
```

```
C	g	Accuracy
128.0	3.0517578125E-5	0.9861111111111112
8192.0	3.0517578125E-5	0.9861111111111112
32768.0	3.0517578125E-5	0.9861111111111112
8.0	3.0517578125E-5	0.9722222222222222
32.0	3.0517578125E-5	0.9722222222222222
512.0	3.0517578125E-5	0.9722222222222222
128.0	1.220703125E-4	0.9583333333333334
2.0	1.220703125E-4	0.9444444444444444
32.0	1.220703125E-4	0.9444444444444444
512.0	1.220703125E-4	0.9444444444444444
2048.0	3.0517578125E-5	0.9444444444444444
8.0	1.220703125E-4	0.9305555555555556
...
```

可以发现，最佳的 gamma 参数相同，对 $C$，我们选择靠前最小的 8。

- Linear kernel 参数选择

```java
SVMDataset dataset = SVMDataset.read("leu.combined");
SVMParameter parameter = new SVMParameter();
parameter.kernel_type = SVMParameter.LINEAR;

GridSearch.create()
        .parameters(parameter)
        .setLog2g(1, 1, 1) // 固定 gamma 值，linear-kernel 不使用 gamma
        .setLog2C(-1, 2, 1)
        .grid(dataset);
```

```
C	g	Accuracy
0.5	2.0	0.9861111111111112
1.0	2.0	0.9861111111111112
2.0	2.0	0.9722222222222222
4.0	2.0	0.9722222222222222
```

最佳 $C=0.5$，准确性 98.6%。

使用 Linear-kernel 的 CV 准确性与使用 RBF kernel 相当。显然，当 features 数量很大，不需要映射数据。

除了 SVM，[LIBLINEAR](http://www.csie.ntu.edu.tw/~cjlin/liblinear) 专门为这类情况情况设计，不过其 Java 版有点过时。

### 样本数和 feature 数都很大

这类数据通常出现在文档分类中。LIBSVM 对这类问题不是很合适。LIBLINEAR 更适合这类问题。

### 样本数 >> feature 数

由于 feature 数量少，人们通常将数据映射到高维空间，即使用非线性 kernel。但是，如果你更想使用 linear-kernel，则可以使用带 `-s 2` 的 LIBLINEAR，当 feature 数量少，`-s 2` 通常比默认的 `-s ` 更快。

 以 [covtype.libsvm.binary.scale](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html) 数据为例，样本数为 581012，feature 数为 54。使用 LIBLINEAR：

```sh
$ time liblinear-2.47/train -c 4 -v 5 -s 2 covtype.libsvm.binary.scale
```

```
Cross Validation Accuracy = 75.6683%
25.148s
```

```sh
$ time liblinear-2.47/train -c 4 -v 5 -s 1 covtype.libsvm.binary.scale
```

```
Cross Validation Accuracy = 75.6728%
134.857s
```

用 LIBSVM 测试：太慢了。。。

```java
SVMDataset dataset = SVMDataset.read("covtype.libsvm.binary.scale");
SVMParameter parameter = new SVMParameter();
parameter.kernel_type = SVMParameter.LINEAR;
parameter.C = 4;
parameter.cache_size = 1000;

double[] predicted = SVMUtils.cvClassification(5, dataset, parameter);
System.out.println("Accuracy = " +
        DecimalFormatUtils.F4.format(SVMUtils.getAccuracy(dataset.y, predicted) * 100) + "%");
```

## 参数

C 值为正则化参数：

- C 值较大，则尽可能好的分离数据，优化器趋向于选择边距较小的超平面
- C 值较小，优化器会寻找分隔较大的超平面，即使该超平面会导致更多错误分类

Gamma 是 RBF kernel 的参数：

- gamma 值越小，决策边界越平滑，容易欠拟合
- gamma 值越大，决策边界越复杂，容易过拟合

> [!TIP] 
>
> 在保证分类准确的前提下，C 和 gamma 都是越小越好。



- kernel cache size

对 SVC, SVR, NuSVC 和 NuSVR，kernel 缓存大小对较大问题的运行时间有很大影响。如果有足够 RAM，建议将 `cache_size` 设置高于默认的 200 M，如 500 MB 或 1000 MB。

## 参考

- https://www.csie.ntu.edu.tw/~cjlin/papers/guide/guide.pdf
- 数据集：https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/ 