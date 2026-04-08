# Adaptive MCMC

## 简介

本文研究利用自适应 MCMC 算法在运行过程中自动调整马尔科夫链参数。包括：

- Adaptive Metropolis (AM) 多元算法
- 非共轭分层模型的 Metropolis-within-Gibbs 算法
- 区域调整 Metropolis 算法
- 对数尺度调整算法

模拟结果表明，即便在高维情形，自适应算法的性能也远优于非自适应算法。

Metropolis-Hastings 等 MCMC 算法在统计推断中应用极为广泛，用于从复杂高维分布中抽样。对 proposal 方差等参数的调优对实现高效混合至关重要，但这一过程也很难。

自适应 MCMC 算法试图解决这一问题：在 MCMC 算法运行过程中，自动 “学习” 更优的参数值。本文列举了多个此类算法，其中包含一些高维场景下的算法。可以看到，自适应 MCMC 能够在极少需要用户干预的情况下，高效寻找到优良参数。在本文语境下，“优良” 将依据马尔可夫链混合效果的合适度量标准来定义，例如目标泛函的积分自相关系数。

众所周知，自适应 MCMC 算法不是总能保持目标分布π(⋅)的平稳性。不过，若在再生时进行自适应调整，或是满足关于自适应过程的多种技术条件，算法就能够收敛。

Roberts 和 Rosenthal（2005）证明了自适应 MCMC 算法在易于应用条件下的遍历性，且不要求自适应参数自身收敛。为精确表述其结论，假设算法使用核 $P_{Γ_n}$ 将 $X_n$ 更新为 $X_{n+1}$，其中每个固定核 $Pγ$ 均具有平稳分布 π(⋅)，但 $Γ_n$ 为随机索引，依据算法之前的输出结果从集合 Y 中迭代选取。用 $\lVert\cdots\rVert$ 表示全变差距离，$X$ 表示状态空间，并记：$Mϵ(x,γ)=\inf\{n≥1:∥P_γ^n(x,⋅)−π(⋅)∥≤ϵ\}$ 为从状态 x∈X 出发时，转移核 Pγ 的收敛时间。那么，Roberts 和 Rosenthal（2005）的定理 13，结合其推论 8、推论 9 以及定理 23，可保证：$\lim_{n→∞}∥L(Xn)−π(⋅)∥=0$（即渐近收敛），同时对所有有界可测函数 g:X→R，有 $\lim_{n→∞}\frac{1}{n}\sum_{i=1}^ng(X_i)=π(g)$，仅需满足**递减自适应条件**即可:
$$

$$


## Adaptive Metropolis-Within-Gibbs

自适应

## 参考

- Roberts, G. O.; Rosenthal, J. S. Examples of Adaptive MCMC. *Journal of Computational and Graphical Statistics* **2009**, *18* (2), 349–367. https://doi.org/10.1198/jcgs.2009.06134.
- https://www.sumsar.net/best_online/
- https://sourceforge.net/projects/hydra-mcmc/
- https://github.com/endymecy/MCMC-sampling/blob/master/src/main/java/sample/MetropolisHastings.java
- https://github.com/cyrilchim/Adaptive-Gibbs-Sampler
- https://blackjax-devs.github.io/blackjax/examples/howto_metropolis_within_gibbs.html
- https://towardsdatascience.com/bayesian-statistics-metropolis-hastings-from-scratch-in-python-c3b10cc4382d/
- https://every-algorithm.github.io/2024/06/03/metropolishastings_algorithm.html
- https://www.statlect.com/fundamentals-of-statistics/Metropolis-Hastings-algorithm
- https://gregorygundersen.com/blog/2019/11/02/metropolis-hastings/
- https://jessekelighine.com/metropolis-hastings-algorithm/#fn1
- [统计计算-李东风](https://www.math.pku.edu.cn/teachers/lidf/docs/statcomp/html/_statcompbook/index.html)



