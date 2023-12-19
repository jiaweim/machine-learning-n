# 随机化数据

- [随机化数据](#随机化数据)
  - [简介](#简介)

2023-12-12, 11:26
****

## 简介

由于学习算法可能受数据顺序影响，因此对数据进行随机化（也称为洗牌）是缓解此问题的常用方法。特别是重复随机化，如在交叉验证期间，有助于生成更真实的统计数据。

Weka 提供了两种随机化数据集的方式：

- 使用 `weka.core.Instances` 对象的 `randomize(Random)` 方法。该方法需要 `java.util.Random` 示例。
- 使用 `Randomize` 过滤器

机器学习试验一个非常重要的方面，就是可重复。相同设置的多次运行，必须产生完全相同的结果。在这种情况下仍然可能随机化。随机数生成器永远不会返回一个完全随机的数字序列，而是伪随机数。为了实现可重复的随机序列，使用 seed 生成器。相同的 seed 总会得到相同的序列。

所以，不要使用 `java.util.Random` 的默认构造函数，而使用 `Random(long)` 构造函数，指定 seed 值。

为了获得更依赖于数据集的随机随机化，可以使用 `weka.core.Instances` 的 `getRandomNumberGenerator(int)` 方法。该方法返回一个 `java.util.Random` 对象，其 seed 为提供的 seed 和从 Instances 中随机选择的 `weka.core.Instance `的 hashcode 的加和。

所以随机化数据可以分为两步：

1. 获得随机数生成器

```java
Instances dataSet = source.getDataSet();

Random random = data.getRandomNumberGenerator(1);
```

2. 随机化数据

```java
dataSet.randomize(random);
```

或者合并为一步：

```java
dataset.randomize(new Random(1));
```
