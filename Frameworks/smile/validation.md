# 模型验证

2025-06-06⭐
@author Jiawei Mao
***
## 简介

在训练监督模型时，需要评估模型性能。这有助于模型选择和超参数调整。我们必须直到，在训练集上测量的模型误差有可能低于实际的泛化误差。

## 评估指标

对分类问题，smile 提供以下评估指标：

- Accuracy：预测正确的比例，包括 true positive 和 true negative

$$
\text{accuracy}=\frac{TP+TN}{N+P}
$$

N 指真实 negative 样本数，P 指真实 positive 样本数。

- Sensitivity 或 true positive rate (TPR)，也称为 recall, hit rate：所有 positive 样本中，正确分类的比例

$$
TPR=\frac{TP}{P}=\frac{TP}{TP+FN}
$$

- specificity 或 true negative rate：指所有 negative 样本中，正确分类的比例

$$
SPC=\frac{TN}{N}=\frac{TN}{FP+TN}=1-FPR
$$

- precision 或 positive predictive value (PPV)：预测为 positive 样本中，正确的比例

$$
PPV=\frac{TP}{TP+FP}
$$

- false discovery rate (FDR)：预测为 positive 样本中，预测错误的比例

$$
FDR=\frac{FP}{TP+FP}
$$

- false positive rate (FPR)：所有 negative 样本中，被错误分类为 positive 的比例

$$
FPR=\frac{FP}{N}=\frac{FP}{FP+TN}
$$

- F-score：同时考虑 precision 和 recall，传统的 F-score 称为 F1-score，是 precision 与 recall 的调和平均值

在 smile 中，class-label 1 视为 positive，0 视为 negative。需要注意的是，并非所有指标适合多分类数据。例如，将 specificity 或 sensitivity 用于多分类数据就不合理。此时，只有 label 1 视为 positive，其它任何值都视为 negative。

计算多分类问题的 accuracy：

```java
var segTrain = Read.arff("data/weka/segment-challenge.arff");
var segTest = Read.arff("data/weka/segment-test.arff");

var model = RandomForest.fit(Formula.lhs("class"), segTrain);
var pred = model.predict(segTest);

smile> Accuracy.of(segTest.column("class").toIntArray(), pred)
$161 ==> 0.9617283950617284
```

Sensitivity 和 Specificity 与 Type I error, Type II error 相关。对任何检验，都需要在这两个指标之间权衡。这种权衡可以用 ROC 图形化方式表示。

示例：计算二分类问题的各种指标

```java

var toyTrain = Read.csv("data/classification/toy200.txt", CSVFormat.DEFAULT.withDelimiter('\t'));
var toyTest = Read.csv("data/classification/toy20000.txt", CSVFormat.DEFAULT.withDelimiter('\t'));

var x = toyTrain.select(1, 2).toArray();
var y = toyTrain.column(0).toIntArray();
var model = LogisticRegression.fit(x, y, 0.1, 0.001, 100);

var testx = toyTest.select(1, 2).toArray();
var testy = toyTest.column(0).toIntArray();
var pred = Arrays.stream(testx).mapToInt(xi -> model.predict(xi)).toArray();

smile> Accuracy.of(testy, pred)
$171 ==> 0.81435

smile> Recall.of(testy, pred)
$172 ==> 0.7828

smile> Sensitivity.of(testy, pred)
$173 ==> 0.7828

smile> Specificity.of(testy, pred)
$174 ==> 0.8459

smile> Fallout.of(testy, pred)
$175 ==> 0.15410000000000001

smile> FDR.of(testy, pred)
$176 ==> 0.16447859963710107

smile> FScore.F1.score(testy, pred)
$177 ==> 0.808301925757654

// Calculate posteriori probability for AUC computation.
var posteriori = new double[2];
var prob = Arrays.stream(testx).mapToDouble(xi -> {
        model.predict(xi, posteriori);
        return posteriori[1];
    }).toArray();

smile> AUC.of(testy, prob)
$180 ==> 0.8650958
```

对回归问题，smile 提供了以下评估指标：

- MSE (mean squared error) 和 RMSE (root mean squared error)
- MAD (mean absolute deviation error)
- RSS (residual sum of squares)

## Out-of-sample Evaluation

泛化误差，又称为样本外误差（out-of-sample），使用训练集中没有的数据评估预测 accuracy。理想情况下，测试集在统计上应该独立于训练集。但是在实践中，我们通常只有一个数据集，并且学习算法可能对抽样误差敏感。接下来，我们讨论各种测试原理。

smile 提供了 java 和 scala 辅助测试函数。java 测试函数位于 `smile.validation.Validation` 类中，scala 辅助函数位于 `smile.validation` 包中。

### Hold-out Testing

hold-out 测试假设所有样本独立同分布（这也是大多数学习算法的基本假设）。留出一部分数据用于测试。许多 benchmark 数据包含一个单独的测试集。

```java
public class ClassificationValidation {
    public static <T, M extends Classifier<T>> ClassificationValidation<M>
        of(T[] x, int[] y, T[] testx, int[] testy,
           BiFunction<T[], int[], M> trainer);

    public static <M extends DataFrameClassifier> ClassificationValidation<M>
        of(Formula formula, DataFrame train, DataFrame test,
           BiFunction<Formula, DataFrame, M> trainer);
}

public class RegressionValidation {
    public static <T, M extends Regression<T>> RegressionValidation<M>
        of(T[] x, double[] y, T[] testx, double[] testy,
           BiFunction<T[], double[], M> trainer);

    public static <M extends DataFrameRegression> RegressionValidation<M>
        of(Formula formula, DataFrame train, DataFrame test,
           BiFunction<Formula, DataFrame, M> trainer);
}
```

例如：

```java
var segTrain = Read.arff("data/weka/segment-challenge.arff");
var segTest = Read.arff("data/weka/segment-test.arff");
var formula = Formula.lhs("class");
var model = RandomForest.fit(formula, segTrain);
var pred = model.predict(segTest);

smile> ConfusionMatrix.of(formula.y(segTest).toIntArray(), pred)
$187 ==> ROW=truth and COL=predicted
class  0 |     124 |       0 |       0 |       0 |       1 |       0 |       0 |
class  1 |       0 |     110 |       0 |       0 |       0 |       0 |       0 |
class  2 |       3 |       0 |     115 |       1 |       3 |       0 |       0 |
class  3 |       2 |       0 |       0 |     106 |       2 |       0 |       0 |
class  4 |       2 |       0 |      10 |       6 |     108 |       0 |       0 |
class  5 |       0 |       0 |       0 |       0 |       0 |      94 |       0 |
class  6 |       2 |       0 |       1 |       0 |       0 |       0 |     120 |
```

```scala
val toyTrain = read.csv("data/classification/toy200.txt", delimiter="\t", header=false)
val toyTest = read.csv("data/classification/toy20000.txt", delimiter="\t", header=false)

val x = toyTrain.select(1, 2).toArray()
val y = toyTrain.column(0).toIntArray()

val testx = toyTest.select(1, 2).toArray()
val testy = toyTest.column(0).toIntArray()

smile> validate.classification(x, y, testx, testy) { case (x, y) => lda(x, y) }
val res13: smile.validation.ClassificationValidation[smile.classification.LDA] =
{
  fit time: 360.135 ms,
  score time: 22.309 ms,
  validation data size: 20000,
  error: 3755,
  accuracy: 81.23%,
  sensitivity: 78.28%,
  specificity: 84.17%,
  precision: 83.18%,
  F1 score: 80.66%,
  MCC: 62.56%,
  AUC: 86.35%,
  log loss: 0.4999
}

smile> validate.classification(x, y, testx, testy) { case (x, y) => logit(x, y, 0.1, 0.001) }
val res14: smile.validation.ClassificationValidation[smile.classification.LogisticRegression] =
{
  fit time: 3.960 ms,
  score time: 4.046 ms,
  validation data size: 20000,
  error: 3713,
  accuracy: 81.44%,
  sensitivity: 78.28%,
  specificity: 84.59%,
  precision: 83.55%,
  F1 score: 80.83%,
  MCC: 63.00%,
  AUC: 86.51%,
  log loss: 0.4907
}
```

### Out-of-bag Error

out-of-bag (OOB) error，也称为 out-of-bag estimate，是一种评估随机森林、提升决策树等机器学习模型预测误差的方法，这些模型使用 bootstrap 对训练集进行采样。OOB 指仅使用 bootstrap 样本中不包含 $x_i$ 的 tree，对每个样本 $x_i$ 取平均值。

```java
var iris = Read.arff("data/weka/iris.arff");
var rf = smile.classification.RandomForest.fit(Formula.lhs("class"), iris);
System.out.println("OOB metrics = " + rf.metrics());
```

OOB 不需要独立验证数据集，但通常会低估性能改进和最佳迭代次数。

## Cross Validation

在 k 折交叉验证中，数据集被随机划分为 k 个部分。将每个部分视为一个 hold-set，然后再剩余数据上训练模型，并在 hold-set 在评估模型质量。整体性能取决于所有 k 个部分的平均值。

```java
public class CrossValidation {
    public static <T, M extends Classifier<T>> ClassificationValidations<M>
        classification(int k, T[] x, int[] y, BiFunction<T[], int[], M> trainer);

    public static <M extends DataFrameClassifier> ClassificationValidations<M>
        classification(int k, Formula formula, DataFrame data, BiFunction<Formula, DataFrame, M> trainer);

    public static <T, M extends Regression<T>> RegressionValidations<M>
        regression(int k, T[] x, double[] y, BiFunction<T[], double[], M> trainer);

    public static <M extends DataFrameRegression> RegressionValidations<M>
        regression(int k, Formula formula, DataFrame data, BiFunction<Formula, DataFrame, M> trainer);
}
```

如果不提供指标，这些方法默认使用 accuracy 或 R2 进行分类和回归。

```java
DataFrame iris = Read.arff(Path.of("\\data\\weka\\iris.arff"));

ClassificationValidations<DecisionTree> cv = 
    CrossValidation.classification(10, Formula.lhs("class"), iris, DecisionTree::fit);
System.out.println(cv);
```

```
{
  fit time: 2.349 ms ± 4.197,
  score time: 0.080 ms ± 0.116,
  validation data size: 15 ± 0,
  error: 1 ± 1,
  accuracy: 96.00% ± 6.44,
  cross entropy: 0.1973 ± 0.1798
}
```

对 Iris 数，10-fold 交叉验证的 accuracy 大约为 84.7%，由于分区的随机性，每次运行得到的结果可能不同。

一种特殊情况是留一法 (LOOCV)，即使用原始样本中单个样本作为测试集，其它数据作为训练集。重复该过程，直到每个样本都做过一次测试集。留一法交叉验证从计算角度来看非常昂贵，因为重复训练此时太多。

```java
public class LOOCV {
    public static <T, M extends Classifier<T>> ClassificationMetrics
        classification(T[] x, int[] y, BiFunction<T[], int[], M> trainer);

    public static <M extends DataFrameClassifier> ClassificationMetrics
        classification(Formula formula, DataFrame data, BiFunction<Formula, DataFrame, M> trainer);

    public static <T, M extends Regression<T>> RegressionMetrics
        regression(T[] x, double[] y, BiFunction<T[], double[], M> trainer);

    public static <M extends DataFrameRegression> RegressionMetrics
        regression(Formula formula, DataFrame data, BiFunction<Formula, DataFrame, M> trainer);
}
```

在 Iris 数据上，LOOCV 的 accuracy 为 85.33%，比 10 折交叉验证略高，这是因为用于训练的数据较多，而用于测试的数据较少。

```java
smile> var x = iris.drop("class").toArray();
x ==> double[150][] { double[4] { 5.099999904632568, 3. ... 68, 1.7999999523162842 } }

smile> var loocv = LOOCV.classification(x, y, (x, y) -> LDA.fit(x, y));
loocv ==> {
  fit time: 1.967 ms,
  score time: 0.014 ms,
  validation data size: 150,
  error: 22,
  accuracy: 85.33%,
  cross entropy: 0.4803
}
```

## Bootstrap

Bootstrap 是一种评估统计准确性的通用工具。其基本思想是，从训练集随机放回抽样，每个 bootstrap 样本大小与原数据集相同。在 bootstrap 样本中，unique 样本比例大约为 $1-1/e\approx 63.2\%$。重复多次，生成 $k$ 个 bootstrap 数据集，比如 $k=100$。然后对每个 bootstrap 数据集拟合模型，并查看 $k$ 次重复的拟合效果。

```java
public class Bootstrap {
    public static <T, M extends Classifier<T>> ClassificationValidations<M>
        classification(int k, T[] x, int[] y, BiFunction<T[], int[], M> trainer);

    public static <M extends DataFrameClassifier> ClassificationValidations<M>
        classification(int k, Formula formula, DataFrame data, BiFunction<Formula, DataFrame, M> trainer);

    public static <T, M extends Regression<T>> RegressionValidations<M>
        regression(int k, T[] x, double[] y, BiFunction<T[], double[], M> trainer);

    public static <M extends DataFrameRegression> RegressionValidations<M>
        regression(int k, Formula formula, DataFrame data, BiFunction<Formula, DataFrame, M> trainer);
}
```

在 Iris 数据集上，100 次 bootstrap 的 accuracy 大约为 83.7%，略低于 10 折交叉验证的 accuracy。

```java
smile> Bootstrap.classification(100, x, y, (x, y) -> LDA.fit(x, y))
$43 ==> {
  fit time: 0.057 ms ± 0.020,
  score time: 0.163 ms ± 0.236,
  validation data size: 55 ± 4,
  error: 9 ± 3,
  accuracy: 83.96% ± 4.68,
  cross entropy: 0.4847 ± 0.0530
}
```

bootstrap 分布与真实分布可能存在系统偏差，这种偏差会导致执行区间的 bias。

## 超参数调整

超参数可以分为两类：

1. 模型超参数：由于涉及模型选择，因此在拟合时无法推断
2. 算法超参数：原则上对模型性能没有影响，但会影响学习的速度和质量

例如，神经网络的拓扑结构和大小是模型超参数，而学习率和 batch 大小是算法超参数。

`Hyperparameters` 类提供了两种参数搜索策略。使用 `add()` 方法可以定义具有指定分布（固定某个值、数组或范围）的参数空间：

- `grid()` 方法会穷举所有参数组合
- `random()` 会随机生成一系列候选参数

```java
import smile.io.*;
import smile.data.formula.Formula;
import smile.validation.*;
import smile.classification.RandomForest;

var hp = new Hyperparameters()
    .add("smile.random.forest.trees", 100) // a fixed value
    .add("smile.random.forest.mtry", new int[] {2, 3, 4}) // an array of values to choose
    .add("smile.random.forest.max.nodes", 100, 500, 50); // range [100, 500] with step 50


var train = Read.arff("data/weka/segment-challenge.arff");
var test = Read.arff("data/weka/segment-test.arff");
var formula = Formula.lhs("class");
var testy = formula.y(test).toIntArray();

hp.grid().forEach(prop -> {
    var model = RandomForest.fit(formula, train, prop);
    var pred = model.predict(test);
    System.out.println(prop);
    System.out.format("Accuracy = %.2f%%%n", (100.0 * Accuracy.of(testy, pred)));
    System.out.println(ConfusionMatrix.of(testy, pred));
});
```

grid-search 很流行，random-search 的优势在于能够独立于参数数量和可能值进行选择。需要注意的是，`random()` 返回一个无限流。因此，应该使用 `limit()` 方法来决定需要测试的次数。

```java
hp.random().limit(20).forEach(prop -> {
    var model = RandomForest.fit(formula, train, prop);
    var pred = model.predict(test);
    System.out.println(prop);
    System.out.format("Accuracy = %.2f%%%n", (100.0 * Accuracy.of(testy, pred)));
    System.out.println(ConfusionMatrix.of(testy, pred));
});
```

在 `Hyperparameters` 的 lambda 函数中，用户可以训练任何模型，甚至多个算法，并使用一个或多个指标进行评估。除了像上例那样在测试数据进行评估，还可以使用交叉验证或 boosting 方法。

grid-search 和 random-search 独立评估每套参数设置。因此，使用并行没问题。注意，有些算法，如 random-forest， logistic regression 已经使用并行，此时不应继续使用并行，避免潜在的死锁。

## 模型选择规则

模型选择指在给定数据的情况下，从一组候选模型中选择最佳模型。对一组性能相似的模型，选择最简单的模型（奥卡姆剃刀原则）。

一个好的模型选择技术会在拟合性能和简单性之间取得平衡。复杂的模型能够更好拟合数据，但额外的参数可能没有任何实际价值。拟合性能通常使用似然比或类似方法来确定，然后进行卡方检验。复杂度则通过模型中的参数量来衡量。

最常用的标准有 Akaike information criterion (AIC) 和 Bayesian information criterion (BIC)。它们在 `ModelSelection` 中实现。BIC 和 AIC 的公式类似，但对参数量的惩罚不同，AIC 的惩罚为 $2k$，而 BIC 的惩罚为 $\log(n)\cdot k$。

AIC 和 BIC 基于不同的假设近似正确，不过都因假设的不切实际而受到批评。

当 false negative 比 false positive 更重要，AIC 更合适；反之则 BIC 更合适。

## 参考

- https://haifengl.github.io/validation.html