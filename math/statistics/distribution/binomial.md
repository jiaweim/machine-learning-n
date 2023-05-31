# 二项分布

- [二项分布](#二项分布)
  - [基础](#基础)
  - [二项分布属性](#二项分布属性)
  - [二项分布应用](#二项分布应用)
  - [R 函数](#r-函数)
    - [概率密度计算](#概率密度计算)
    - [分布函数计算](#分布函数计算)
    - [分位数计算](#分位数计算)
    - [随机数](#随机数)
  - [图形示例](#图形示例)
  - [参考](#参考)

2020-05-12, 16:22
@jiaweiM
***

## 基础

二项式概率分布，简称二项分布，用于处理结果只有两种的情况，如硬币是证明还是反面。

满足如下条件的事件符合二项分布：

1. 试验次数固定
2. 试验之间互相独立，即一个试验的结果不会影响其它试验的概率
3. 试验结果只有两种情况，如成功或失败
4. 所有试验成功的概率相同

设事件 A 在一次试验中发生的概率为 p。现在把这试验独立重复 n 次，则 A 发生 i 次的概率为：

$$p_i=\binom{n}{i}p^i(1-p)^{n-i}, i=0,1,...,n$$

二项分布记为 $B(n,p)$。X 服从二项部分记为 $X\text{\textasciitilde}B(n,p)$。

## 二项分布属性

期望值 $E(x)=np$

标准差 $\sigma(x)^2=np(1-p)$

## 二项分布应用

二项分布有两个重要条件：

1. 各次试验的条件稳定，保证时间的概率 p 在各次试验中保持不变。
2. 各次试验的独立性。

现实生活中有许多现象不同程度地符合这两条件。

例如，某厂每天生产 n 个产品，若原材料质量、机器设备、工人操作水平等在一段时间内大体保持稳定，且每次产品合格与否与其它产品无显著关联，则每日的废品数 X 就大体上服从二项分布。

## R 函数

`dbinom` 计算概率密度，`pbinom` 计算分布函数，`qbinom` 给出分位数，`rbinom` 生成随机数。

### 概率密度计算

概率密度函数计算，由于二项分布是离散部分，所以也称为概率质量函数：

```r
dbinom(x, size, prob, log = FALSE)
```

每次试验成功的概率为 `prob`，计算 `size` 次试验成功 `x` 次的概率。log，如果为TRUE，将概率值 p 转换为 log(p) 返回。

如果 x 不是整数，返回 0，并抛出异常。

- 例1

如果小明发球命中率为 60%，如果他罚球12次，那么他投中10的概率是：

```r
> p <- dbinom(x = 10, size = 12, prob = .6)
> print(p)
[1] 0.06385228
```

- 例2

波妞投掷一枚均匀的硬币20次，那么7次正面朝上的概率是多少？

```r
> p <- dbinom(x = 7, size = 20, prob = .5)
> print(p)
[1] 0.07392883
```

### 分布函数计算

```r
pbinom(q, size, prob, lower.tail = TRUE, log.p = FALSE)
```

`pbinom` 用于计算累积密度函数（cumulative density function, CDF），`prob` 为单次试验成功概率，`size` 为试验次数，`q` 为成功次数。

通俗地讲，`pbinom` 计算 `q` 值左侧二项分布区域的面积。如果你想计算 `q` 值右侧的面积，设置 `lower.tail=FALSE`即可。

- 例1

波妞投掷均匀硬币 5 次，硬币正面朝上次数超过2次的概率？

```r
p <- pbinom(2, size = 5, prob = .5, lower.tail = FALSE)
print(p)
```

即正面朝上至少2两次的概率为 0.5。

- 例2

假设波妞打保龄球得分的概率为 30%，如果他投球 10 次，那么她得分不超过4的概率为多少？

```r
> p <- pbinom(4, size = 10, prob = .3)
> print(p)
[1] 0.8497317
```

### 分位数计算

分位数定义为使得分布函数 $F(x)\geq p$的最小 x 值。为累积密度函数的逆运算，对概率 p，试验次数 `size`以及每次成功概率 `prob`，计算累积概率不小于 `p` 的最小 x 值。

```r
qbinom(p, size, prob, lower.tail = TRUE, log.p = FALSE)
```

- 例1

随机试验成功概率为 0.4，试验 10 次，那么10分位对应值（即累积概率 10% 处的随机变量值）：

```r
> p <- qbinom(.1, size = 10, prob = .4)
> print(p)
[1] 2
```

- 例2

随机试验30次，每次成功概率为 0.25，则40分位数为：

```r
> n <- qbinom(.4, size = 30, prob = .25)
> print(n)
[1] 7
```

### 随机数

函数 `rbinom` 根据试验次数 `size` 以及每次试验成功的概率 `prob` 生成 n 个满足该二项分布的随机数，以长度为 n 的向量形式返回。

```r
rbinom(n, size, prob)
```

例如：

```r
> # 100 次试验，每次试验成功概率为 0.3，生成 10 个数，对应成功次数
> results <- rbinom(10, size = 100, prob = .3)
> print(results)
 [1] 36 26 26 18 30 33 30 36 28 26
```

> 这里的随机数，表示在指定二项分布下，成功的次数。

如下所示，进行该实验的次数越多，最后均值越接近 100*0.3=30。

```r
> results <- rbinom(10, size = 100, prob = .3)
> print(results)
 [1] 29 35 30 30 27 29 30 31 31 33
> print(mean(results))
[1] 30.5
>
> results <- rbinom(100, size = 100, prob = .3)
> print(mean(results))
[1] 30.78
>
> results <- rbinom(10000, size = 100, prob = .3)
> print(mean(results))
[1] 29.9757
```

## 图形示例

- 概率分布

下面我们看一下试验10次，每次试验成功概率为 0.3 的概率分布情况：

```r
library(dplyr)
library(ggplot2)

data.frame(heads = 0:10,
           prob = dbinom(x = 0:10, size = 10, prob = 0.3)) %>%
  mutate(Heads = ifelse(heads == 2, "2", "Other")) %>%
  ggplot(aes(x = factor(heads), y = prob, fill = Heads)) +
  geom_col() +
  geom_text(
    aes(label = round(prob, 2), y = prob + 0.01),
    position = position_dodge(0.9),
    size = 3,
    vjust = 0
  ) +
  labs(title = "Probability of X=2 successed.",
       subtitle = "b(10, 0.3)",
       x = "Successes (x)",
       y = "probability"
  )
```

![plot](2020-08-31-09-14-20.png)

- 累积概率分布

```r
library(dplyr)
library(ggplot2)

tibble(heads = 0:10,
       pmf = dbinom(x = 0:10, size = 10, prob = 0.3),
       cdf = pbinom(q = 0:10, size = 10, prob = 0.3)
) %>%
  mutate(Heads = ifelse(heads <= 4, "<=5", ">5")) %>%
  ggplot(aes(x = factor(heads), y = cdf, fill = Heads)) +
  geom_col() +
  geom_text(
    aes(label = round(cdf, 2), y = cdf + 0.01),
    position = position_dodge(0.9),
    size = 3,
    vjust = 0
  ) +
  labs(title = "Probability of X <= 5 successes.",
       subtitle = "b(10, 0.3)",
       x = "Successes (x)",
       y = "Probability"
  )
```

![bar](2020-08-31-09-26-26.png)

## 参考

- 概率论与数理统计-陈希孺
- [statology](https://www.statology.org/dbinom-pbinom-qbinom-rbinom-in-r/)
- [RPubs](https://rpubs.com/mpfoley73/458411)
