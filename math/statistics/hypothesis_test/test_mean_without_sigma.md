# 单样本均值检验（σ 未知）

- [单样本均值检验（σ 未知）](#单样本均值检验σ-未知)
  - [t 分布中的临界值](#t-分布中的临界值)
  - [均值 μ 的 t-Test](#均值-μ-的-t-test)
  - [单总体均值 t-test 流程：拒绝域](#单总体均值-t-test-流程拒绝域)
  - [t-test 流程：p-Value](#t-test-流程p-value)

2024-06-07
@author Jiawei Mao
***

## t 分布中的临界值

在现实生活中，总体标准差大多时候是未知的，因此 Z-test 应用有限。

当**总体为正态分布**或**样本量不小于 30** 时，仍然可以检验总体均值 $\mu$。只是换成 t-test。

**计算 t 分布临界值**

1. 执行显著性水平 $\alpha$；
2. 确定自由度 $\mathbf{d.f.} = n-1$；
3. 查表，根据自由度确定临界值。

如下图所示：

![image-20240506100348060](./images/image-20240506100348060.png)

**例 1：** 查找左边检验的临界值

假设 $\alpha=0.05$，$n=21$，计算左边检验临界值 $t_0$。

**解：** 自由度为

$$
\text{d.f.}=n-1=21-1=20
$$

```java
TDistribution distribution = new TDistribution(20);
double v = distribution.inverseCumulativeProbability(0.05);
assertEquals(v, -1.725, 1e-3);
```

如下图所示：

<img src="./images/image-20240506103127467.png" alt="image-20240506103127467" style="zoom: 50%;" />

**例 2：** 查找右边检验的临界值

假设 $\alpha=0.01$，$n=17$，计算右边检验的临界值 $t_0$。

```java
double criticalValue = TTestUtils.getRightCriticalValue(16, 0.01);
assertEquals(criticalValue, 2.583, 1e-3);
```

<img src="./images/image-20240506103545439.png" alt="image-20240506103545439" style="zoom:50%;" />

**例 3：** 双边检验的临界值。

假设 $\alpha=0.10$，$n=26$。

```java
DoubleDoublePair criticalValue = TTestUtils.getCriticalValue(25, 0.1);
assertEquals(criticalValue.leftDouble(), -1.708, 1e-3);
assertEquals(criticalValue.rightDouble(), 1.708, 1e-3);
```

## 均值 μ 的 t-Test

$\sigma$ 未知对均值 $\mu$ 的检验，可以使用 t 抽样分布。标准化检验统计量的形式为：

$$
t=\frac{样本均值-假设均值}{标准差}
$$

由于 $\sigma$ 未知，所以用**样本标准差** $s$ 替代 $\sigma$ 计算标准化检验统计量。

均值 $\mu$ 的 t 检验即对总体均值的统计检验。检验统计量为样本均值 $\overline{x}$。标准化检验统计量为：
$$
t=\frac{\overline{x}-\mu}{s/\sqrt{n}}
$$
需满足的前提条件：

1. 样本随机
2. 总体为正态分布，或样本量 $n\ge 30$

自由度 $df=n-1$。

## 单总体均值 t-test 流程：拒绝域

1. 验证前提条件：$\sigma$ 未知，样本随机，总体为正态分布或 $n\ge 30$；
2. 声明假设：$H_0$ 和 $H_a$；
3. 指定显著性水平 $\alpha$；
4. 确定自由度：$df=n-1$；
5. 计算临界值；
6. 确定拒绝域；
7. 计算标准化检验统计量：$t=\frac{\overline{x}-\mu}{s/\sqrt{n}}$；
8. 下结论：拒绝或无法拒绝 null 假设；
9. 解释结论。

**例 4** 使用拒绝域策略执行 -test

一位二手车商说，过去 12 个月售出的二手车平均价格至少为 21000 美元。你怀疑这个说法不正确，随机选择过去 12 个月售出的 14 辆二手车，平均价格为 19189 美元，标准差为 2950 美元。那么，是否有足够证据表明在 $\alpha=0.05$ 时拒绝二手车商的说法？假设总体为正态分布。

1. $\sigma$ 未知，样本随机，总体为正态分布，可以用 t-test；
2. 要拒绝的假设：$H_0\ge 21000$，备择假设 $H_a<21000$；
3. 显著性水平 $\alpha=0.05$；
4. 自由度：$df=n-1=13$;
5. 显然，这是左边检验，计算临界值：

```java
double criticalValue = TTestUtils.getLeftCriticalValue(13, 0.05);
System.out.println(criticalValue); // -1.7709333959890659
```

6. 所以，拒绝域为 $t<-1.771$；
7. 计算检验统计量

```jade
double statistic = TTestUtils.getStatistic(19189, 21000, 2950 * 2950, 14);
assertEquals(statistic, -2.297, 1e-3);
```

8. 下结论

由于 t 落在拒绝域内，因此拒绝 null 假设。

9. 解释

在 5% 显著性水平下，有足够证据来拒绝 $H_0$，即拒绝过去 12 个月售出的二手车的平均价格至少为 21000 美元的说法。

> 用假设检验得出的结论可能犯错，拒绝 $H_0$ 可能犯 I 类错误。

## t-test 流程：p-Value

对单个均值的 t-test，也可以用 p-value。

**例 5** 基于 p-value 的 t-test

一个车站管理处称，人们平均等待时间不到 14 分钟。随机抽取 10 个人，平均等待时间 13 分钟，标准差 3.5 分钟。在$\alpha=0.10$ 时，检验该车站管理处的说法。假设总体为正态分布。

1. 验证前提条件：$\sigma$  未知，样本随机，总体为正态分布；
2. 假设：$H_0$：$\mu \ge 14$ 分钟, $H_a$: $\mu < 14$ 分钟；
3. 显著性水平 $\alpha=0.10$；
4. 自由度：$df=9$；
5. 根据假设可判断右边检验，计算 p-value

```java
double pValue = TTestUtils.getOneSampleOneTailedPValue(13, 14, 3.5 * 3.5, 10);
System.out.println(pValue); // 0.19489940273037554
```

6. 下结论：因为 p-value 大于 $\alpha$，所以无法拒绝 null 假设；
7. 解释

在 10% 置信水平下，没有足够证据支持管理处平均时间少于 14 分钟的说法。

