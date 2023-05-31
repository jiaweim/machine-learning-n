# 矩阵代数

2023-05-31
****
## 矩阵的定义

**$p\times q$ 矩阵**：指 p 行 q 列的矩阵，也会用 $(a_{ij}):p\times q$ 这种简写形式

$$\mathbf{A}=
\begin{bmatrix} a_{11} & a_{12} & \dots & a_{1q} \\ a_{21} & a_{22} & \dots & a_{2q} \\ \vdots & \vdots & & \vdots \\ a_{p1} & a_{p2} & \dots & a_{pq} \end{bmatrix}=\left(a_{ij}\right):p\times q
$$

**p 维列向量：** 当矩阵的列数为 1，得到列向量

$$\mathbf{a}=\begin{pmatrix}
a_1 \\
a_2 \\
\vdots \\
a_p
\end{pmatrix}$$
**q 维行向量：** 当矩阵的行数为 1，得到行向量，用 a 的转置符号 $a'$ 或 $a^T$ 表示

$$\mathbf{a'(=a^T)}=(a_1,a_2,...,a_q)$$
**向量 a 的长度：**

$$\lVert \mathbf{a}\rVert=\sqrt{\mathbf{a'a}}=\sqrt{a^2_1+a^2_2+\cdots+a^2_p}$$

**单位向量：** 如果向量长度为 1，就称其为单位向量

$$\lVert \mathbf{a}\rVert=1$$

### 向量的几何意义

- 以 $p=2$ 为例

$$\mathbf{a}=\begin{pmatrix}
a_1\\
a_2
\end{pmatrix}$$
向量在几何上可以有两种理解：

- 看作坐标点
- 看作带有方向和长度的量

![[Pasted image 20230531092303.png|250]]
> 坐标点
![[Pasted image 20230531092323.png|250]]
> 带有方向和长度的量

### 一些矩阵概念

- **零矩阵**：所有元素为 0

$$\mathbf{0}=\mathbf{0_{pq}}=(0):p\times q$$
- **p 阶方阵**：行数和列数相同

$$\mathbf{A}=\begin{bmatrix} a_{11} & a_{12} & \dots & a_{1p} \\ a_{21} & a_{22} & \dots & a_{2p} \\ \vdots & \vdots & & \vdots \\ a_{p1} & a_{p2} & \dots & a_{pp} \end{bmatrix}$$
- **对角线元素**：p 阶方阵对角线上的元素

$$a_{11},a_{22},...,a_{pp}$$
- **非对角线元素**：p 阶方阵非对角线元素

$$a_{ij}(i\ne j)$$
- **上三角矩阵**：对角线左下角部分都为 0 的方阵

$$\mathbf{A}=\begin{bmatrix} a_{11} & a_{12} & \dots & a_{1p} \\ 0 & a_{22} & \dots & a_{2p} \\ \vdots & \vdots & & \vdots \\ 0 & 0 & \dots & a_{pp} \end{bmatrix}$$
- **下三角矩阵**：对角线右上角部分都为 0 的方阵

$$\mathbf{A}=
\begin{bmatrix}
a_{11} & 0 & \dots & 0 \\ 
a_{21} & a_{22} & \dots & 0 \\ 
\vdots & \vdots & & \vdots \\ 
a_{p1} & a_{p2} & \dots & a_{pq} 
\end{bmatrix}$$
- **对角矩阵**：除了对角线，其它元素都是 0 的方阵

$$\mathbf{A}=\begin{pmatrix}
a_{11} & 0 & \dots & 0 \\
0 & a_{22} & \dots & 0 \\
\vdots & \vdots && \vdots \\
0 & 0 & \dots & a_{pp}
\end{pmatrix}=diag(a_{11},a_{22},...,a_{pp})$$

- **单位矩阵**：对角线元素都为 1 的对角矩阵

$$\mathbf{I}=\mathbf{I_p}=\begin{pmatrix}
1 & 0 & \dots & 0 \\
0 & 1 & \dots & 0 \\
\vdots & \vdots && \vdots \\
0 & 0 & \dots & 1
\end{pmatrix}$$
- **A 的转置**：将 A 和行和列互换一下称为 A 的转置，记为 A' 或 $A^T$，转置后从 $p\times q$ 变为 $q\times p$

$$\mathbf{A'}=\mathbf{A^T}=\begin{pmatrix}
a_{11} & a_{21} & \dots a_{p1} \\
a_{12} & a_{22} & \dots a_{p2} \\
\vdots & \vdots & & \vdots \\
a_{1q} & a_{2q} & \dots & a_{pq}
\end{pmatrix}$$
- **对称矩阵**：A 为方阵，满足 $A'=A$；显然，$a_{ij}=a_{ji}$。例如：

$$\begin{pmatrix}
1 & 3 \\
3 & 4
\end{pmatrix}$$
$$\begin{pmatrix}
9 & 0 & 0 \\
0 & 7 & 0 \\
0 & 0 & 1 
\end{pmatrix}$$
$$\begin{pmatrix}
2 & 3 & 5 \\
3 & 0 & 1 \\
5 & 1 & 1 
\end{pmatrix}$$
## 矩阵的运算

- **矩阵和**：若 $A=(a_{ij}):p\times q$, $B=(b_{ij}):p\times q$，即 A 和 B 阶数相同，则 A 和 B 的和定义为
$$\mathbf{A+B}=(a_{ij}+b_{ij}):p\times q$$


- **常数和矩阵乘积**：常数 c 与 A 的积定义为

$$cA=(ca_{ij}):p\times q$$

- **矩阵和矩阵乘积**：若 $A=(a_{ij}):p\times q$，$B=(b_{ij}):q\times r$，即 A 的列数必须与 B 的行数相同，则 A 与 B 的积定义为

$$AB=\begin{bmatrix}
\vdots & \vdots & \dots & \vdots \\
a_{i1} & a_{i2} & \dots & a_{iq} \\
\vdots & \vdots & \dots & \vdots
\end{bmatrix}
\begin{bmatrix}
\dots & b_{1j} & \dots \\
\dots & b_{2j} & \dots \\
\vdots & \vdots & \vdots \\
\dots & v_{qj} & \dots 
\end{bmatrix}=\left(\sum^{q}_{k=1}a_{ik}b_{kj}\right):p\times r$$
$\left(\sum^{q}_{k=1}a_{ik}b_{kj}\right)$ 是第 i 行第 j 列的元素值。

### 矩阵的运算规律

1. $\boldsymbol{(A+B)'=A'+B'}$

根据元素加和易证。

2. $(AB)'=B'A'$

3. $A(B_1+B_2)=AB_1+AB_2$

易证。

4. $A(\sum^{k}_{i=1}B_i)=\sum^{k}_{i=1}AB_i$

规律 3 的推广。

5. $c(A+B)=cA+cB$

易证。

### 矩阵的分块

- **分块矩阵定义**：设 $A=(a_{ij}):p\times q$，将它分成四块，表示成

![[Pasted image 20230531103140.png|200]]
如果 A 的阶数很高，在分块之后，有可能计算更方便，或者有利于数学推导。在统计学中，矩阵分块之后，不同块可能有不同的统计含义。

- **分块矩阵相加**：若 A 和 B 有相同的分块，则

$$A+B=\begin{bmatrix}
A_{11}+B_{11} & A_{12}+B_{12} \\
A_{21}+B_{21} & A_{22}+B_{22}
\end{bmatrix}$$
- **分块矩阵乘积**：若 C 为 $q\times r$ 矩阵，分成

$$C=\begin{bmatrix}
C_{11} & C_{12} \\
C_{21} & C_{22}
\end{bmatrix}$$
则有（要求 A 分块的列数与 C 分块的行数相等）：

![[Pasted image 20230531104050.png|300]]
$=\begin{pmatrix}A_{11}C_{11}+A_{12}C_{21} & A_{11}C_{12}+A_{12}C_{22} \\ A_{21}C_{11}+A_{22}C_{21} & A_{21}C_{12}+A_{22}C_{22} \end{pmatrix}$

里面都是矩阵相乘，不是元素相乘，所以前后位置不能互换。


## 正交矩阵

## 矩阵的行列式、逆和秩

## 矩阵的特征值、特征向量和迹

## 正定矩阵、非负定矩阵


