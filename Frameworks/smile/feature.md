# 特征工程

## 简介

特征的生成、选择和降维对机器学习算法最终的影响往往比算法选择本身更重要。

理解问题和业务室准备特征的关键。除了训练样本提供的属性外，我们还可以对属性进行修改或增强，以找到能够更好描述对象的特征。

推断函数的准确性在很大程度上取决于输入对象的表示方式。通常，输入对象会被转换为特征向量。由于维度灾难，特征数量不宜过多；但应该包含足够的信息以准确预测输出。除了特征向量，有些算法（如 SVM）可以通过设计 kernel 函数处理复杂的数据对象，如序列、tree 甚至 graph。

如果每个特征对输出做出独立贡献，那么基于线性函数的算法，如线性回归、逻辑回归、线性 SVM、朴素贝叶斯性能通常很好；如果特征之间存在复杂的相互作用，则非线性 SVM、决策树和神经网络等算法效果更佳。

如果输入特征包含冗余信息，如高度相关的特征，某些算法，如线性回归、路基回归和基于距离的方法会由于数值不稳定而表现不佳。这些问题通常可以通过正则化解决。

## 预处理

2025-10-23⭐
***

许多机器学习算法，如神经网络和采用高斯核的 SVM，需要对特征进行适当的缩放/标准化。例如，每个变量被缩放到 [0,1] 区间，或者均值为 0，标准差为 1。基于**距离函数**的方法对此尤为敏感。`smile.feature` 提供了几个用于预处理特征的类。这些类通常从**训练集中学习变换**，然后应用于新的特征向量。

### Scaler

`Scaler` 类将数值变量缩放到 [0,1] 范围。

### WinsorScaler

如果数据集包含 outliers，那么归一化会将数据缩放到一个非常小的区间。此时，应该采用 Winsorization 算法：大于指定上限的值替换为上限，低于下限的值替换为下限。范围通常以原始分布的百分位数表示，如 5th 和 95th 百分位。`WinsorScaler` 实现了该算法。

### MaxAbsScaler

`MaxAbsScaler` 采用每个特征绝对值的最大值对特征进行缩放，使得每个特征的最大绝对值为 1。它不会移动数据或使数据居中，因此不会破坏数据分布。

```java
import smile.base.mlp.Layer;
var pendigits = Read.csv("data/classification/pendigits.txt", 
                         CSVFormat.DEFAULT.withDelimiter('\t'));
var df = pendigits.drop(16);
var y = pendigits.column(16).toIntArray();
var scaler = WinsorScaler.fit(df, 0.01, 0.99);
var x = scaler.apply(df).toArray();

CrossValidation.classification(10, x, y, (x, y) -> {
    var model = new smile.classification.MLP(Layer.input(16),
                Layer.sigmoid(50),
                Layer.mle(10, OutputFunction.SIGMOID)
        );

    for (int epoch = 0; epoch < 10; epoch++) {
        for (int i : MathEx.permutate(x.length)) {
            model.update(x[i], y[i]);
        }
    }

    return model;
});
```

### Standardizer

`Standardizer` 将数值特征转换为均值为 0，方差为 1。标准化假设数据服从高斯分布，当有 outliers 时该方法不够 robust。一个 robust 替代方案是减去中位数，再除以 IQR，`RobustStandardizer` 实现了该方法：

```java
var zip = Read.csv("data/usps/zip.train", CSVFormat.DEFAULT.withDelimiter(' '));
var df = zip.drop(0);
var y = zip.column(0).toIntArray();

var scaler = Standardizer.fit(df);
var x = scaler.apply(df).toArray();

var model = new smile.classification.MLP(Layer.input(256),
            Layer.sigmoid(768),
            Layer.sigmoid(192),
            Layer.sigmoid(30),
            Layer.mle(10, OutputFunction.SIGMOID)
    );

model.setLearningRate(TimeFunction.constant(0.1));
model.setMomentum(TimeFunction.constant(0.0));

for (int epoch = 0; epoch < 10; epoch++) {
    for (int i : MathEx.permutate(x.length)) {
        model.update(x[i], y[i]);
    }
}
```

### Normalizer

`Normalizer` 将**样本**转换为单位范数。该类无状态，无需从数据中学习变换参数。

每个包含至少一个非 0 特征的样本（即每一行）都会独立于其它样本进行缩放，使其范数（L1 或 L2）等于 1。将输入缩放到单位范数是文本分类或文本聚类等问题的常见操作。

### 其它方法

`smile.math.MathEx` 类也提供了几个类似用途的函数，例如用于矩阵的 `standardize()`, `normalize()` 和 `scale()`。

虽然有些算法（如决策树）能够直接处理 nominal 变量，但其它算法通常需要将 nominal 变量转换为多个 binary 变量，以表示特征的存在或不存在。`OneHotEncoder` 类使用 one-of-K 方法对分类特征进行编码。

在 `smile.feature` 中还有其它特征生成类。例如，`DateFeature` 生成 `Date` 对象的属性。`Bag` 是词袋 feature 的通用实现，可以用于 `String` 以外的通用对象。

## 参考

- https://haifengl.github.io/feature.html