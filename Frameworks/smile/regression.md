# 回归

## 简介

与分类问题不同，回归分析输出连续值。Smile 的回归算法位于 `smile.regression`  package，所有算法都实现 `Regression` 接口，该接口包含一个 `predict` 方法，用于将模型应用于实例。所有算法都可以通过构造函数进行模型训练；同时每个算法都有一个对应的 `Trainer` 类，用于保存模型超参数并应用于多个训练数据集。

## 普通最小二乘法

在线性回归中，因变量为自变量的线性组合。残差是模型预测值与真实值之间的差值。普通最小二乘法获取使残差平方和（Sum of Squared Residues, SSE）最小的参数。

当自变量不存在共线性时，普通最小二乘（Ordinary Least Squares, OLS）是一致的；当误差是同方差且串行不相关时，OLS 线性无偏估计最优。在这些条件下，当误差具有有限方差时，OLS 方法提供最小方差均值的无偏估计。

```java
public class OLS {
    public static LinearModel fit(Formula formula,
                                  DataFrame data,
                                  Options options);
}
```

可以在多种不同的框架中构建线性回归模型，获得的公式和结果都相同。框架的选择取决于数据性质和需要执行的推理任务。

如果实现误差呈正态分布，则最小二乘对应最大似然准则，也可以作为矩估计方法的推导。

示例：

```java
DataFrame planes = Read.arff("2dplanes.arff");
LinearModel model = OLS.fit(Formula.lhs("y"), planes);
System.out.println(model);
```

```
Linear Model:

Residuals:
       Min          1Q      Median          3Q         Max
   -8.5260     -1.6514     -0.0049      1.6755      7.8116

Coefficients:
                  Estimate Std. Error    t value   Pr(>|t|)
Intercept          -0.0148     0.0118    -1.2503     0.2112 
x1                  2.9730     0.0118   251.7998     0.0000 ***
x2                  1.5344     0.0145   105.8468     0.0000 ***
x3                  1.0357     0.0144    71.7815     0.0000 ***
x4                  0.5281     0.0145    36.4827     0.0000 ***
x5                  1.4766     0.0144   102.2472     0.0000 ***
x6                  1.0044     0.0144    69.5380     0.0000 ***
x7                  0.5238     0.0145    36.1696     0.0000 ***
x8                 -0.0011     0.0145    -0.0750     0.9402 
x9                  0.0024     0.0145     0.1649     0.8690 
x10                -0.0278     0.0145    -1.9239     0.0544 .
---------------------------------------------------------------------
Significance codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

Residual standard error: 2.3838 on 40757 degrees of freedom
Multiple R-squared: 0.7056,    Adjusted R-squared: 0.7055
F-statistic: 9766.9504 on 11 and 40757 DF,  p-value: 0.000
```

在回归模型应用于新数据：

```java
double out = model.predict(planes.get(0));
System.out.println(out);
```

```
5.073347388202894
```

评估拟合效果的常用指标包括 R-squared, 残差分析和假设检验。

## 参考

- https://haifengl.github.io/regression.html