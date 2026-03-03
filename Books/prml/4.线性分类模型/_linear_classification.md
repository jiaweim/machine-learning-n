# 线性分类模型

## 简介

分类的目标是将输入向量 $\mathbf{x}$ 分配到 $K$ 个离散类别 $C_k$ 之一，其中 $k=1,\cdots,K$。最常见的情况是不同类别互不相交，因此每个输入只分配到一个类。输入空间由此被划分为**决策区域**（decision region），区域的边界称为**决策边界**（decision boundary）或决策面（decision surface）。下面讨论线性分类模型，线性分类模型的 decision-surface 是输入向量 $\mathbf{x}$ 的线性函数，因此由 $D$ 维输入空间的 $D-1$ 维超平面定义。如果数据集的类别能够通过线性 decision-surface 精确分离，则称该数据集是线性可分的。

对回归问题，target-variable $t$ 是实数向量。对分类问题，有多种方法可以使用 target-variable 来表示 class-labels。对概率模型：

- 在二分类问题中，最方便的方法是二元表示，target-variable $t\in \{0,1\}$，其中 $t=1$ 表示 class $C_1$，$t=0$ 表示 class $C_2$。我们可以将 $t$ 值解释为 class 是 $C_1$ 的概率，其值在 0 到 1 之间。
- 对 $K>2$ 多类别情况，则使用 one-hot-encoding 很方便，此时 $t$ 是长度为 $K$ 的向量，如果类别为 $C_j$，那么 $t$ 的所有元素 $t_k$ 除了 $t_j=1$，其它都是 0。

例如，如果有 $K=5$ 个 classes，那么 class-2 对应的 target-vector 为：
$$
\mathbf{t}=(0,1,0,0,0)^T \tag{1}
$$
此时，我们还是可以将 $t_k$ 的值解释为 class 是 $C_k$ 的概率。对非概率模型，target-variable 还是其它合适的选择。

对分类问题有三种不同的分类方法：

1. 最简单的方法是构建一个判别函数（discriminant function），直接将每个向量 $\mathbf{x}$ 分配到特定 class。
2. 更有效的方法是在推理阶段对条件概率分布 $p(C_k|\mathbf{x})$ 建模，然后使用该分布做出最优决策。将推理和决策分离有许多好处，如 1.5.4 节所述。

确定条件概率 $p(C_k|\mathbf{x})$ 有两种不同的方法：一种是直接建模，例如，将它们表示为参数模型，然后使用训练集优化参数；另一种是采用生成方法，即对由 $p(\mathbf{x}|C_k)$ 给出的 class 条件概率以及各个 classes 的先验概率 $p(C_k)$ 进行建模，然后使用贝叶斯定理计算所需的后验概率。
$$
p(C_k|\mathbf{x})=\frac{p(\mathbf{x}|C_k)p(C_k)}{p(\mathbf{x})} \tag{2}
$$
在本章将通过示例演示这三种方法。

在第三章讨论的线性回归模型中，
