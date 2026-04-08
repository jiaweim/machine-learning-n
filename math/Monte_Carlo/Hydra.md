# Hydra

## 简介

Hydra 是一个开源的 MCMC 计算库。

MCMC 是一种对可表示为分布的函数进行数值积分的方法，其优势在于，它能够从各类分布进行模拟，而无需对密度函数做严格的归一化处理。这使其成为贝叶斯统计模型中不可获取的工具，因为在该领域，对后验密度进行归一化往往不现实，甚至无法实现。

经过一段初始预热（burn-in）后，构造合理的 MCMC 采样器会从指定概率分布 $\pi$ 中生成一组（非独立）样本序列：$X_0,X_1,...,X_N$。利用这些样本，可通过样本均值 $\hat{E(g)}=\frac{1}{N}\sum_{t=0}^N g(X_t)$ 来估计任意函数 $g$ 在分布 $\Pi$ 下的期望。尽管大多数 MCMC 算法本身非常简洁，但 MCMC 算法软件包依然很少。

## 创建 Metropolis-Hastings Samplers

Hydra 提供了一套最常用的 MCMC 技术。包括：

- Metropolis-Hastings 算法
- Metropolis sampler
- Gibbs sampler
- multi-state Adaptive Metropolis Sampling (Gilks & Roberts, 1996)

下面重点介绍 Metropolis-Hastings 方法的实现，其他方法均可视为它的特例。

Metropolis–Hastings 算法非常简洁。给定与统计模型对应的目标分布 Π、初始点 X0 以及提议分布 Q(Xt)，采样器的每一次迭代包含四个步骤：

1. 基于提议分布 Q(Xt) 生成一个候选状态 Y，该分布可能依赖于当前状态 Xt

$$
Y\leftarrow Q(X_t)
$$

2. 计算 Metropolis–Hastings 接受概率

$$
\begin{aligned}
    \alpha(X_t,Y)&=\min\{1,\frac{\pi(Y)q(Y\rightarrow X_t)}{\pi(X_t)q(X+t\rightarrow Y)}\}\\
    &=\min{1,\frac{p(Y)q(Y\rightarrow X_t)}{P(X_t)q(X_t\rightarrow Y)}}
\end{aligned}
$$

其中，π 是对应目标分布 Π 的概率密度，p(x)∝π(x) 为**未归一化密度**，而q(Y→Xt)=q(Y∣Xt) 是在提议分布 Q(Xt) 下 Y 的条件密度。

3. 以概率 $\alpha(X_t,Y)$ 接受提议点 $Y$，并设置

$$
X_{t+1}\leftarrow Y
$$

拒绝提议点，则设置：
$$
X_{t+1}\leftarrow X_t
$$

4. 增加时间

$$
t\leftarrow t+1
$$

只要提议分布 Q 满足特定条件，该算法生成的序列 X 就会收敛为一组来自目标分布 Π 的**非独立样本**。

`CustomMetropolisHastingsSampler` 类实现了 Metropolis-Hastings 采样器的核心逻辑，它使用由用户指定的目标分布（模型）、初始状态以及提议分布来完成采样。实现这一功能的前提是，代表目标分布与提议分布的对象必须提供特定的方法。这些方法分别由 `UnnormalizedDensity`（未归一化密度）接口与 `GeneralProposal`（通用提议分布）接口所定义。只要初始状态与用户指定的目标分布和提议分布相兼容，对其便不做任何额外限制。

为了灵活报告 MCMC 采样器的运行进度，`CustomMetropolisHastingsSampler` 会维护一个用户自定义对象列表，在每次迭代的接受步骤完成时对这些对象进行通知。当选择详细报告模式时，这些 “监听器” 会接收到一个对象，其中包含每次 MCMC 迭代的大量详细信息。

### UnnormalizedDensity 用于目标分布

目标分布需实现 `UnnormalizedDensity` 接口，该接口定义了两个方法：

```java
double unnormalizedPDF(Object state);

double logUnnormalizedPDF(Object state);
```

这些方法用于计算以传入参数形式给定状态下，模型的（对数）未归一化密度。

### GeneralProposal 用于提议分布

提议分布实现 **GeneralProposal** 接口，该接口包含 4 个方法：

```java
double conditionalPDF(Object stat, Object conditions);

double logConditionalPDF(Object state, Object conditions);

double transitionProbability(Object from, Object to);

double logTransitionProbability(Object from, Object to);
```

`conditionalPDF` 与 `logConditionalPDF` 方法用于计算：在当前状态为 `current` 的条件下，生成下一个对象 `next` 的概率。后两个方法执行相同的计算，只是交换了参数顺序。

> [!NOTE]
>
> The transitionProbability and logTransitionProbability are depreciated and will not be required in a future release of the software.

### MCMCListener 接口用于监听

在每次 MCMC 迭代完成时接收通知的对象，需要实现 **MCMCListener** 接口，该接口定义了一个方法：

```java
void notify(MCMCEvent event);
```

`notify` 方法的参数是一个包含本次 MCMC 迭代相关信息的对象。该对象可以通过多种方式使用这些信息，例如将当前状态保存到文件、在图表中展示，以及计算累积统计量等。

当禁用详细报告功能时，传递给 `notify` 方法的对象为 `GenericChainStepEvent`。该对象仅包含一个字段：

```java
public MCMCState current;
```

该字段包含采样器的当前状态 Xt。

当启用详细报告功能时，传递给 `notify` 方法的对象为 `DetailedChainStepEvent`，该对象额外包含以下字段：

```java
public MCMCState proposed;
public MCMCState last;
public double lastProb;
public double proposedProb;
public double forwardProb;
public double reverseProb;
public double probAccept;
public double acceptRand;
public boolean accepted;
public double acceptRate;
```

这些字段提供了关于 MCMC 迭代的大量信息，有助于调试以及评估不同提议分布的性能。每个字段的含义如表 1 所示。

## 示例

Hydra 提供的类可直接用于编译后的 Java 程序。我们将通过一个实例进行说明：使用 Hydra 为食管癌中遗传物质缺失的**二项–贝塔二项混合模型**构建两种不同的采样器。首先展示如何在 Java 中实现对应于该二项–贝塔二项混合模型的**未归一化密度函数**，使其能够与 Hydra 配合使用。利用该模型，我们先构建一个单变量逐次更新的 Metropolis 采样器，再构建一个正态核耦合器（Normal Kernel Coupler）。

Metropolis-Hastings 采样器包含四个由用户指定的组成部分：

1. 目标分布（模型）
2. 初始状态
3. 提议分布
4. 均匀随机数生成器

Hydra 库提供了可靠的随机数生成器和一系列标准**提议分布**，用户只需自行构建目标分布与初始状态即可。

### 创建目标分布

要创建一个表示目标分布（模型）的对象，用户需要实现 `UnnormalizedDensity` 接口。针对我们示例中的问题，我们希望为该贝叶斯分层模型实现一个对应的类。
$$
X_i\sim \eta Binomial(N_i,\pi_1)+(1-\eta)Beta-Binomial(N_i,\pi_2,\omega_2)\\
\eta \sim Unif(0,1)\\
\pi_1\sim Unif(0,1)\\
\pi_2\sim Unif(0,1)\\
\omega_2\sim Unif(0,1/2)
$$
`Binomial_BetaBinomial_SimpleLikelihood` 给出了实现该模型密度函数的 Java 类。我们将重点介绍使该类能够用作目标分布的编程细节。

首先，需要实现 `UnnormalizedDensity` 接口：

```java
public class Binomial_BetaBinomial_SimpleLikelihood implements UnnormalizedDensity
```

现在，我们的类必须提供 `unnormalizedPDF` 和 `logUnnormalizedPDF` 这两个方法。Metropolis-Hastings 采样器会使用这些方法来计算提议状态的接受概率。





## 参考

- https://sourceforge.net/projects/hydra-mcmc/