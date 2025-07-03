# LM 方法及其实现

## 简介

最小二乘问题：
$$
f(x)=\frac{1}{2}\sum_{j=1}^m r_j^2(x) \tag{1}
$$
存在一种廉价的方法来计算目标函数 $f$ 的 Hessian 矩阵的近似值。通过省略 Hessian 矩阵中通常可以忽略不计的一部分，由此得到的近似值。

### 目标函数推导

将 （1）式所有残差合并为一个残差向量，简化目标函数 $f$：
$$
r(x)\coloneqq (r_1(x),r_2(x),\cdots,r_m(x))^T
$$

因此，$f$ 可以记为：

$$
f(x)=\frac{1}{2}\sum_{j=1}^m r_j^2(x)=\frac{1}{2}r(x)^Tr(x)=\frac{1}{2}\lVert r(x)\rVert^2
$$

残差向量 $r$ 的雅克比矩阵是一个 $m\times n$ 矩阵，形式如下：
$$
J(x)\begin{bmatrix}
    \triangledown r_1(x)^T\\
    \triangledown r_2(x)^T\\
    \vdots\\
    \triangledown r_m(x)^T\\
\end{bmatrix}
$$

其中，缩写 $\triangledown r_j(x)$ 表示 $r_j(x)$ 对 $j\in \{1,\cdots,m\}$ 的梯度。这里 $\triangledown r_j(x)$ 为列向量。

此外，可以直接计算目标函数 $f$ 的梯度和 Hessian：

- 梯度可以利用链式法则获得
 
$$
\triangledown f(x)=\sum_{j=1}^m r_j(x)\cdot \triangledown r_j(x)=J(x)^T r(x)
$$



## 雅可比矩阵

**单变量变换**

单变量求积分：

$$
\int_a^b f(x)dx
$$

有时可以采用变量变换技巧：

$$
\begin{aligned}
    x&=h(y)\\
    \frac{dx}{dy}&=h'(y)\\
    dx&=h'(y)dy
\end{aligned}
$$

因此：

$$
\int_a^b f(x)dx=\int_c^d f(h(y))h'(y)dy
$$

其中：

$$
\begin{aligned}
a&=h(c)\\
b&=h(d)\\
dx&=Jdy
\end{aligned}
$$

此时，雅可比矩阵 $J$ 是 $dx$ 与 $dy$ 的比值关系。

**双变量变换**

求重积分时：

$$
\int\int_Rf(x,y)dxdy
$$

也可以采用变量变换技巧：

$$
\begin{aligned}
(x,y)&\rightarrow (u,v)\\
x&=h(u,v)\\
y&=k(u,v)
\end{aligned}
$$

因此：

$$
\int\int_R f(x,y)dxdy=\int\int_Q f(h(u,v),k(u,v))\cdot \lvert \frac{\partial(x,y)}{\partial (u,v)}\rvert dudv
$$

其中：

$$
\lvert\frac{\partial(x,y)}{\partial(u,v)}\rvert=\begin{bmatrix}
    \frac{\partial x}{\partial u}& \frac{\partial x}{\partial v}\\
    \frac{\partial y}{\partial u}& \frac{\partial y}{\partial v}
\end{bmatrix}
$$

$dxdy$ 可以看作一个很小的面积，$dudv$ 也是如此，

### Jacobian 示例

考虑如下积分：

$$
\int_0^2 x\cos(x^2)dx
$$

为了计算该积分，可以使用 $u$ 替换：

$$
u=x^2
$$

替换后，区间 $[0,2]$ 变为 $[0,4]$。可以看到区间被拉伸了，而且这种拉伸不均匀。所以我们还需要计算 $du$：

$$
\frac{du}{dx}=2x
$$




## 参考

- 《The Levenberg-Marquardt Method and Its Implementation in Python》Master Thesis by Marius Kaltenbach at University of Konstanz, 康斯坦茨大学
- https://math.libretexts.org/Bookshelves/Calculus/Supplemental_Modules_(Calculus)/Vector_Calculus/3%3A_Multiple_Integrals/3.8%3A_Jacobians
- https://www.khanacademy.org/math/multivariable-calculus/multivariable-derivatives/jacobian/v/the-jacobian-matrix