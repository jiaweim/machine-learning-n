# 多元数据的描述与展示

## 一元随机变量回顾

一元随机变量：

- 数值特征描述
- 可视化

### 数值特征

随机变量由它的概率分布函数确定，如何用数值来代表这个分布，或者这个随机变量的特征。一般关心两个特征：中心位置和离散程度。

1. 中心位置：总体均值

$$\mu=E(Y)=\int yf(y)dy$$
2. 离散程度：总体方差

$$\sigma^2=var(Y)=E(Y-\mu)^2$$
总体标准差：

$$\sigma=\sqrt{\sigma^2}$$
标准差与原变量单位一致。

样本与总体稍微不同，对随机样本 $\{y_1,...,y_n\}$：

**样本均值**：

$$\overline{y}=\frac{1}{n}\sum^{n}_{i=1}y_i$$
样本均值可用来估计总体均值。

**样本方差：**

$$s^2=\frac{1}{n-1}\sum^{n}_{i=1}(y_i-\overline{y})^2$$
```ad-note
样本方差分母采用 $n-1$，使样本方差为总体方差的无偏估计。
```

**样本标准差：**

$$s=\sqrt{s^2}$$
样本标准差与样本单位相同，还可以刻画样本的离散程度。

### 二元随机变量

二元随机变量 $(X,Y)$：

**总体协方差：**

$$\sigma_{XY}=cov(X,Y)=E[(X-\mu_X)(Y-\mu_Y)]=E(XY)-\mu_X\mu_Y$$
协方差可正可负。

**总体相关系数：**

$$\rho_{XY}=corr(X,Y)=\frac{\sigma_{XY}}{\sigma_X\sigma_Y}$$
对 **二元随机样本**： $\{(x_1,y_1),...,(x_n,y_n)\}$

**样本协方差：**

$$s_{xy}=\frac{1}{n-1}\sum^{n}_{i=1}(x_i-\overline{x})(y_i-\overline{y})$$
**样本相关系数：**

$$r_{xy}=\frac{S_{xy}}{S_xS_y}$$
### 协方差与独立性

- $\sigma_{XY}=0 \Leftrightarrow$  X 和 Y 是不相关/**线性**独立的。
- 如果 X  和 Y 服从二元正态分布，那么：

$$\sigma_{XY}=0 \Leftrightarrow X 和 Y 是独立的$$


```ad-note
协方差刻画两个变量之间的线性关系。
```

### 可视化

**例**：由20名男生构成的样本所提供的身高（单位：英寸）和体重（单位：磅）数据如下：

![[Pasted image 20230530115233.png|500]]

**一维散点图**

![[Pasted image 20230530115305.png|350]]

一维散点图能大致看出 X 和 Y 的相关关系。

**二维散点图**

![[Pasted image 20230530115503.png|350]]
二维散点图比一维散点图更直观。

## 随机向量的描述与展示

### 多元数据的数值特征及可视化

经典多元数据：鸢尾花数据集

![[Pasted image 20230530115915.png|350]]

Iris data 就是鸢尾花数据，是由著名统计学家 R. A. Fisher 在 1936 年提出来研究多元数据分析的经典数据。数据包含50个样本，来自3个纸鸢花物种，数据还包括花萼（sepal）和花瓣（petal）的长度和宽度。

#### 多元数据的矩阵表示

设有 n 个样本点，每个样本点包含 p 个变量的观测值，则数据集可表示为 $n\times p$ 矩阵：

$$Y=\begin{bmatrix}
y_{11} & \dots & y_{1j} & \dots & y_{1p}\\
\vdots && \vdots && \vdots\\
y_{i1} & \dots & y_{ij} & \dots & y_{ip} \\
\vdots && \vdots && \vdots \\
y_{n1} & \dots & y_{nj} & \dots & y_{np}
\end{bmatrix}=\begin{bmatrix}
y_1' \\
\vdots \\
y_i' \\
\vdots \\
y_n'
\end{bmatrix}$$
其中 $y_i'=(y_{i1},...,y_{ip})'$ 由 Y 的第 i 行构成，表示第 i 个样本。

#### 均值向量（Mean Vector）

对随机向量 $y=(Y_1,...,Y_p)'$，总体均值为：

$$E(y)=(E(Y_1),...,E(Y_p))'=(\mu_1,...,\mu_p)'=\mathbb{\mu}$$
对随机样本 $\{y_1,...,y_n\}$，样本均值：

$$\overline{y}=\frac{1}{n}\sum^{n}_{i=1}y_i=(\overline{y}_1,...,\overline{y}_p)'$$
其中 $\overline{y}_j=\frac{1}{n}\sum^{n}_{i=1}y_{ij}$，$E(\overline{y})=\mu$。

即对每个分量求一个均值，$\overline{y}_j$ 就是第 j 个分量的均值。

如三维向量数据的均值示意图：

![[Pasted image 20230530135543.png|500]]

均值向量的 R 实现：

```r
> round(colMeans(iris[,1:4]),2) # 保留了两位有效数字
Sepal.Length Sepal.Width Petal.Length Petal.Width
        5.84        3.06         3.76        1.20
```

#### 协方差矩阵（Covariance matrix）

描述多元数据的离散程度，以及多元数据变量与变量之间的相关性。

对随机向量 $y=(Y_1,...,Y_p)'$，$p\times p$ **总体协方差矩阵**定义为：

$$\sum=Cov(Y)=E[(y-\mu)(y-\mu)']=\begin{bmatrix}
\sigma_{11} & \sigma_{12} & \dots & \sigma_{1p} \\
\sigma_{21} & \sigma_{22} & \dots & \sigma_{2p} \\
\vdots & \vdots && \vdots \\
\sigma_{p1} & \sigma_{p2} & \dots & \sigma_{pp}
\end{bmatrix}$$

$\sigma_{jk}$ 为 $Y_j$ 和 $Y_k$ 之间的协方差。
$\sigma_{jj}=\sigma^2_j$ 为 $Y_j$ 的方差。

对随机样本 $\{y_1,...,y_n\}$，$p\times p$ **样本协方差矩阵**定义为：

$$S=\frac{1}{n-1}\sum^{n}_{i=1}(y_i-\overline{y})(y_i-\overline{y})'=\begin{bmatrix}
s_{11} & s_{12} & \dots & s_{1p}\\
s_{21} & s_{22} & \dots & s_{2p}\\
\vdots & \vdots &&\vdots \\
s_{p1} & s_{p2} & \dots & s_{pp}
\end{bmatrix}$$
其中，$s_{jk}=\frac{1}{n-1}\sum^{n}_{i=1}(y_{ij}-\overline{y}_j)(y_{ij}-\overline{y}_k)$，
$s_{jj}=s^2_j=\frac{1}{n-1}\sum^{n}_{i=1}(y_{ij}-\overline{y}_j)^2$。

#### 协方差矩阵的性质

- $\sum$ 和 $S$ 是对称的，因为 $\sigma_{jk}=\sigma_{kj}$，$s_{jk}=s_{kj}$
- $S$ 是 $\sum$ 的无偏估计，即 $E(S)=\sum$
- $\overline{s}$ 的协方差矩阵是 $COV(\overline{y})=\sum/n$

这都与一维的情况类似，在一维的基础上进行了扩展。

#### 相关系数矩阵

**总体相关系数矩阵（Correlation matrix）**

$$P=(\rho_{jk})=\begin{bmatrix}
1 & \rho_{12} & \dots & \rho_{1p} \\
\rho_{21} & 1 & \dots & \rho_{2p} \\
\vdots & \vdots & & \vdots \\
\rho_{p1} & \rho_{p2} & \dots & 1
\end{bmatrix}$$

说明：

- 相关系数是协方差的标准化。
- 对每一个协方差，$\rho_{jk}=\sigma_{jk}/(\sigma_j\sigma_k)$ 为 $Y_j$ 和 $Y_k$ 之间的总体相关系数。
- 对角线元素为 1.

**样本相关系数矩阵**

对随机样本 $\{y_1,...,y_n\}$ ：

$$R=(r_{jk})=\begin{bmatrix}
1 & r_{12} & \dots & r_{1p} \\
r_{21} & 1 & \dots & r_{2p} \\
\vdots & \vdots && \vdots \\
r_{p1} & r_{p2} & \dots & 1
\end{bmatrix}$$
其中，$r_{jk}=S_{jk}/\sqrt{S_{jj}S_{kk}}=S_{jk}/(S_jS_k)$ 是第 j 和第 k 个变量之间的样本相关系数。

样本协方差矩阵和相关系数矩阵的 R 实现：

```r
S<-round(cov(iris[,1:4]),2) #keep 2 decimals of the covariance matrix
R<-round(cor(iris[,1:4]),2) #keep 2 decimals of the correlation matrix
```

#### 随机向量可视化

$n\times p$ 多元数据矩阵 Y 可以通过两种散点图来表示：

- 两两散点图矩阵（scatterplot matrix）
- 三维散点图

**鸢尾花数据集四个变量之间的两两散点图**

```r
pairs(iris[,1:4],main="Scatterplot Matrix for Fisher's Iris Data")
```

![[Pasted image 20230530145046.png|500]]
用颜色显示种类信息：

![[Pasted image 20230530150710.png|500]]

**三维散点图**

```r
> cloud(Sepal.Length~Petal.Length*Petal.Width, data = iris,
+ groups = Species, screen = list(z = 20, x = -70, y=2),
+ key = list(title = "Iris Data", x = 0.05, y = 1, corner = c(0,1),
+ border = TRUE, points = Rows(trellis.par.get("superpose.symbol"), 1:3),
+ text = list(levels(iris$Species))))
```

![[Pasted image 20230530152619.png|500]]

### 协方差矩阵的用途

协方差矩阵的用途：

1. 刻画数据整体离散型
2. 定义统计距离

#### 刻画数据整体离散型

协方差矩阵是个矩阵，不是数，不便于刻画数据整体离散型。线性代数里的行列式 $|S|$。

如果 $|S|$ 很小，由可能数据波动比较小，也有可能是存在共线性线性。故将协方差矩阵的行列式 $|S|$ 称为**广义方差**（generalized variance）。

矩阵的迹，也叫 trace，$tr(S)$ 刻画了各变量波动程度的总和，但忽略了变量间的相关性，故称为**总方差**（total variance）。trace 就是矩阵对角线求和，协方差矩阵的对角线就是每个分量变量的方差，所以称其为总方差。

#### 定义统计距离

回顾：在一元中，如何定义 $y_1$ 和 $y_2$ 两点之间的距离？

- 欧氏距离：$|y_1-y_2|$
- 标准化/统计距离：$|y_1-y_2|/S_y$

在**多元情形**：对两个 $p$ 维向量 $y_1=(y_{11},...,y_{1p})'$ 和 $y_2=(y_{21},...,y_{2p})'$，其欧式距离定义为：

$$\lVert y_1-y_2\rVert=\sqrt{(y_1-y_2)'(y_1-y_2)}=\sqrt{\sum^{p}_{j=1}(y_{1j}-y_{2j})^2}$$
**欧式距离**只考虑分量之间的距离，没有考虑：

- 不同变量变化的尺度不同
- 变量之间的相关性

要体现出尺度问题，又能体现变量之间的相关性，引入了**统计距离/马氏距离**（Mahalanobis distance）。

类似于一元情形 $|y_1-y_2|/S_y$，定义 $\overline{y}_1$ 和 $\overline{y}_2$ 之间的统计距离/马氏距离为：

$$d=\sqrt{(y_1-y_2)'S^{-1}(y_1-y_2)}$$
这样，方差更大的变量贡献更小的权重，两个高度相关的变量的贡献小于两个相关性较低的变量。其中 $S^{-1}$ 是 S 的逆。

#### 统计距离 vs 欧式距离

统计距离其实是两个经过 “标准化” 的向量 $S^{-1/2}y_1$ 和 $S^{-1/2}y_2$ 之间的欧式距离：

$$\begin{align}
\lVert \sqrt{S^{-1/2}}y_1-{\sqrt{S^{-1/2}}}y_2 \rVert &= \lVert \sqrt{S^{-1/2}}(y_1-y_2) \rVert \\
&=\sqrt{(y_1-y_2)'(S^{-1/2})'S^{-1/2}(y_1-y_2)}\\
&=\sqrt{(y_1-y_2)'S^{-1}(y_1-y_2)}
\end{align}$$
其中 $S^{-1}$ 表示 S 的逆。

可以验证 $Cov(S^{-1/2}Y_1)=I_p$ ，即这个协方差矩阵为单位矩阵。$S^{-1/2}Y_1$ 操作等价于对 $Y_1$ 进行旋转和伸缩变换。

以鸢尾花数据集为例，这里只用前 6 行数据。

**欧氏距离**

```r
#pairwise Euclidean distance of the first 6 rows
L2<-dist(iris[1:6,1:4])   
L2
```

![[Pasted image 20230530165552.png|400]]

**马氏距离**

```r
#pairwise statistical distance of the first 6 rows
library(expm)
S.inv.sqrt<-sqrtm(solve(S))                     #obtain S^{-1/2}
Y.tran<-as.matrix(iris[1:6,1:4])%*%S.inv.sqrt   #transform the original data
d<-dist(Y.tran)           #Euclidean distance of the transformed data matrix
d
```

## 随机向量的变换

- 随机向量的分割
- 变量的线性组合

### 随机向量的分割

将 p 维样本分割为两部分：

$$y=(Y_1,...,Y_p)'=\begin{pmatrix}
y^{1} \\
y^{2}
\end{pmatrix}$$
其中 $y^{(1)}=(Y_1,...,Y_q)'$，$y^{(2)}=(Y_{q+1},...,Y_p)'$。

#### 均值分割

此时，y 的**总体均值向量**可以分割为：

$$\mu=\begin{pmatrix}
\mu^{(1)}\\
\mu^{(2)}
\end{pmatrix}$$
其中，$\mu^{(1)}=E(y^{(1)})=(\mu_1,...,\mu_q)'$，$\mu^{(2)}=E(y^{(2)})=(\mu_{q+1},...,\mu_p)'$。

样本均值向量 $\overline{y}$ 可以用同样的方式进行分割：

#### 协方差矩阵分割

y 的总体方差矩阵可以分割为：

$$\sum=\begin{pmatrix}
(\sum_{11})_{q\times q} & (\sum_{12})_{q\times(p-q)}\\
(\sum_{21})_{(p-q)\times q} & (\sum_{22})_{(p-q)\times (p-q)}
\end{pmatrix}_{p\times p}$$
其中：

- $\sum_{11}$ 是 $y^{(1)}$ 的协方差矩阵
- $\sum_{22}$ 是 $y^{(2)}$ 的协方差矩阵
- $\sum_{12}=\sum_{21}'$ 是包含所有 $y^{(1)}$ 和 $y^{(2)}$ 元素两两之间协方差的矩阵，通常记作 $COV(y^{(1)},y^{(2)})$。

样本协方差矩阵可以做类似的分隔。

依然以鸢尾花数据集为例：

![[Pasted image 20230530175056.png|450]]
![[Pasted image 20230530175203.png|450]]

对相关系数进行类似的分割：

![[Pasted image 20230530175329.png|450]]

### 随机向量的线性组合

