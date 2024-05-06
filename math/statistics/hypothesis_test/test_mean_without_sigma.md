# 单样本均值检验（σ 未知）

## t 分布中的临界值

在现实生活中，总体标准差大多时候是未知的，因此 Z-test 应用有限。

当**总体为正态分布**或**样本量不小于 30** 时，仍然可以检验总体均值 $\mu$。为此，需要用到 t-分布。

### 计算 t 分布临界值

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

由于 $\sigma$ 未知，所以用样本标准差 $s$ 计算标准化检验统计量。


