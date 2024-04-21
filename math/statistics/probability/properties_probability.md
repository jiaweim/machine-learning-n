# 概率的属性

统计学涉及数据的收集和分析。统计学对结果不确定的试验进行研究，这类试验称为**随机试验**（random experiments）。随机试验的具体结果在试验前虽然未知，但所有可能结果的集合是已知的。

所有可能结果的集合用 $S$ 表示，称为**样本空间**。给定样本空间 S，设 A 是 S 中结果集合的子集，即 $A\subset S$，那么 A 称为**事件**（event）。当进行随机试验，并且试验结果在 A 中，就称事件 A 发生了。

在学习概率时，集合（set） 和事件（event）可以互换使用。下面是一些集合术语：

- $\emptyset$ 称为空集
- $A\subset B$ 称 A 是 B 的子集（subset）
- $A\cup B$ 是 A 和 B 的并集（union）
- $A\cap B$ 是 A 和 B 的交集（intersection）
- $A'$ 称为 A 的补集（即 $A'$ 包含 $S$ 中所有不属于 $A$ 的元素）

统计学中与事件相关的一些术语：

- $A_1,A_2,...,A_k$ 为**互斥事件**（mutually exclusive event），表明 $A_i\cap A_j=\emptyset$，$i\ne j$；即 $A_1,A_2,...,A_k$ 为不相交集合；
- $A_1,A_2,...,A_k$ 为**穷举事件**（exhaustive event），表明 $A_1\cup A_2 \cup \cdots \cup A_k=S$。

因此，如果 $A_1,A_2,...,A_k$ 为互斥穷举事件，表明 $A_i\cap A_j=\emptyset, i\ne j$，且 $A_1\cup A_2\cup\cdots\cup A_k=S$。

集合操作满足多个属性。例如，如果 A, B 和 C 是 S 的子集，那么：

- 交换律（commutative law）

$$A\cup B = B\cup A$$
$$A\cap B=B\cap A$$

- 结合律（associative law）

$$(A\cup B)\cup C=A\cup(B\cup C)$$
$$(A\cap B)\cap C=A\cap(B\cap C)$$

- 分配率（distributive law）

$$A\cap (B\cup C)=(A\cap B)\cup(A\cap C)$$
- De Morgan's law

$$(A\cup B)'=A'\cap B'$$
$$(A\cap B)'=A'\cup B'$$
可以用韦恩图来证明 De Morgan 定理。如下图所示，

![[Pasted image 20230605141101.png|400]]

如 (a) 所示，$A\cup B$ 为横线区域，那么 $(A\cup B)'$ 为竖线区域。在 (b) 中，$A'$ 用横线表示，$B'$ 用竖线表示，那么 $A'\cap B'$  为交叉区域。显然，(b) 中的交叉区域和（a）的竖线区域相同。

我们感兴趣的是事件 A 发生的概率。设重复试验 n 次，计算事件 A 在这 n 次试验中发生的次数，该次数称为事件 A 的频率，用 $N(A)$ 表示。