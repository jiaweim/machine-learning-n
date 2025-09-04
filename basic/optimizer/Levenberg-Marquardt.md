# Levenberg-Marquardt

2025-09-03
@author Jiawei Mao
***
## 简介

在数学和计算领域，莱文博格-马夸尔特（Levenberg-Marquardt Algorithm, LMA）用于解决**非线性最小二乘问题**，又称为阻尼最小二乘法（Damped Least-Squares, DLS）。这类最小化问题在最小二乘曲线拟合中很常见，即将参数化数学模型拟合到一组数据点，通过最小化一个目标函数，该目标函数表示为模型函数与一组数据点之间的误差平方和。如果模型的系数是线性的，那么最小二乘目标函数是系数的二次函数。该目标函数可以通过线性方程组一步求解。如果拟合函数的系数不是线性的，则最小二乘问题需要迭代求解。此类算法通过对模型系数值进行一系列精心更新，来降低模型函数与数据点之间的误差平方和。

LMA 算法结合了两种数值最小化算法：高斯-牛顿算法（Gauss-Newton Algorithm, GNA）和梯度下降（Gradient-Descent），即保证了收敛速度，又提高了数值稳定性。在梯度下降中，通过更新最陡下降方向的系数来降低误差平方和；在高斯-牛顿法中，假设最小二乘函数的系数为局部二次函数，并求解二次函数的最小值以最小化误差平方和：

- 当系数远离其最优值时，LMA 的行为与梯度下降类似
- 当系数接近最优值时，LMA 的行为更类似高斯-牛顿法

LMA 比 GNA 更稳健，在许多情况下，即使起始值距离最终最小值很远，它也能找到解。对行为良好的函数和合理的起始参数，LMA 通常比 GNA 慢。

LMA 算法最初由 Kenneth Levenberg 于 1944 年在法兰克福陆军兵工厂工作时发表。1963 年，杜邦公司的统计学家 Donald Marquardt 重新发现该算法，之后， Girard, Wynne, Morrison 也分别独立发现了该算法。

## 问题

Levenberg-Marquardt 算法主要应用于最小二乘曲线拟合问题：给定一组 $m$ 对自变量和因变量对 $(x_i,y_i)$，求出模型曲线 $f(x,\beta)$ 的参数 $\beta$，使用残差平方和 $S(\beta)$ 最小化：

$$
\hat{\beta}\in \argmin _{\beta}S(\beta)\equiv \argmin _{\beta}\sum_{i=1}^m[y_i-f(x_i,\beta)]^2
$$

- **高斯-牛顿法**：通过线性化残差（泰勒展开）近似目标函数，迭代更新参数，但当雅可比矩阵接近奇异时可能不收敛
- **梯度下降法**：沿负梯度方向更新参数，稳定性好但收敛慢

LM 方法引入阻尼因子 $\lambda$ 融合两者：

- 当 $\lambda$ 较小时，接近高斯-牛顿法（收敛快）
- 当 $\lambda$ 较大时，接近梯度下降法（更稳定）

## 迭代公式

每次迭代，参数更新 $\Delta x$ 通过求解以下线性方程组得到：
$$
(J^TJ+\lambda I)\Delta x=J^Tr
$$
其中：

- $J$ 是残差向量 $r=(r_1,\cdots,r_m)^T$ 的雅可比矩阵（$m\times n$，第 $i$ 行第 $j$ 列为 $\partial r_i/\partial x_j$）
- I 是单位矩阵
- $\lambda$ 是阻尼 因子（初始值通常设置为较小的正数，如 $10^{-3}$）

**阻尼因子跳转策略**

迭代后计算残差 $F(x+\Delta x)$，并与原残差 $F(x)$ 比较：

- 若 $F(x+\Delta x)<F(x)$，说明更新有效，接受 $\Delta x$，并减小 $\lambda$ (如 $\lambda=\lambda/10$)
- 若更新无效，则拒绝 $\Delta x$，并增大 $\lambda$ (如 $\lambda=\lambda\times 10$)，重新计算 $\Delta x$

**收敛条件**

当满足以下任一条件时停止迭代：

- 残差平方和 $F(x)$ 小于预设阈值，如 $10^{-6}$
- 参数更新量 $\lVert\Delta x\rVert$ 小于阈值
- 达到最大迭代次数

## Java 实现

矩阵乘法、转置、Cholesky 分解等功能

```java
public class MatrixUtils {

    // 矩阵乘法：A(m×n) * B(n×p) = C(m×p)
    public static double[][] multiply(double[][] A, double[][] B) {
        int m = A.length;
        int n = B.length;
        int p = B[0].length;
        double[][] C = new double[m][p];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < p; j++) {
                for (int k = 0; k < n; k++) {
                    C[i][j] += A[i][k] * B[k][j];
                }
            }
        }
        return C;
    }

    // 矩阵转置：A(m×n) → A^T(n×m)
    public static double[][] transpose(double[][] A) {
        int m = A.length;
        int n = A[0].length;
        double[][] A_T = new double[n][m];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                A_T[j][i] = A[i][j];
            }
        }
        return A_T;
    }

    // 向量点积：v · w
    public static double dotProduct(double[] v, double[] w) {
        double sum = 0;
        for (int i = 0; i < v.length; i++) {
            sum += v[i] * w[i];
        }
        return sum;
    }

    // Cholesky分解：A = L * L^T（A是正定矩阵）
    public static double[][] choleskyDecomposition(double[][] A) {
        int n = A.length;
        double[][] L = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j <= i; j++) {
                double sum = 0;
                if (j == i) {
                    for (int k = 0; k < j; k++) {
                        sum += L[j][k] * L[j][k];
                    }
                    L[j][j] = Math.sqrt(A[j][j] - sum);
                } else {
                    for (int k = 0; k < j; k++) {
                        sum += L[i][k] * L[j][k];
                    }
                    L[i][j] = (A[i][j] - sum) / L[j][j];
                }
            }
        }
        return L;
    }

    // 解下三角方程组 L * y = b
    public static double[] solveLowerTriangular(double[][] L, double[] b) {
        int n = L.length;
        double[] y = new double[n];
        for (int i = 0; i < n; i++) {
            double sum = 0;
            for (int k = 0; k < i; k++) {
                sum += L[i][k] * y[k];
            }
            y[i] = (b[i] - sum) / L[i][i];
        }
        return y;
    }

    // 解上三角方程组 L^T * x = y
    public static double[] solveUpperTriangular(double[][] L, double[] y) {
        int n = L.length;
        double[] x = new double[n];
        for (int i = n - 1; i >= 0; i--) {
            double sum = 0;
            for (int k = i + 1; k < n; k++) {
                sum += L[k][i] * x[k]; // L^T的(i,k) = L(k,i)
            }
            x[i] = (y[i] - sum) / L[i][i];
        }
        return x;
    }

    // 求解线性方程组 A * x = b（A是正定矩阵，用Cholesky分解）
    public static double[] solveLinearSystem(double[][] A, double[] b) {
        double[][] L = choleskyDecomposition(A);
        double[] y = solveLowerTriangular(L, b);
        return solveUpperTriangular(L, y);
    }

    // 打印矩阵（调试用）
    public static void printMatrix(double[][] A) {
        for (double[] row : A) {
            for (double val : row) {
                System.out.printf("%.4f ", val);
            }
            System.out.println();
        }
    }
}
```



## 工具

### smile

smile 提供的 `LevenbergMarquardt.fit` 方法需要四个参数：

1. `DifferentiableMultivariateFunction` 可微函数
2. x, 包含所有 x 值的 `double[]`
3. `y`, 包含所有 y 值的 `double[]`

```java
DifferentiableMultivariateFunction func = new DifferentiableMultivariateFunction() {
    @Override
    public double f(double[] x) {
        return 1 / (1 + x[0] * Math.pow(x[2], x[1]));
    }

    @Override
    public double g(double[] x, double[] g) {
        double pow = Math.pow(x[2], x[1]);
        double de = 1 + x[0] * pow;
        g[0] = -pow / (de * de);
        g[1] = -(x[0] * x[1] * Math.log(x[2]) * pow) / (de * de);
        return 1 / de;
    }
};
MathEx.setSeed(19650218); // to get repeatable results.
double[] x = new double[100];
double[] y = new double[100];
GaussianDistribution d = new GaussianDistribution(0.0, 1);
for (int i = 0; i < x.length; i++) {
    x[i] = (i + 1) * 0.05;
    y[i] = 1.0 / (1 + 1.2 * Math.pow(x[i], 1.8)) + d.rand() * 0.03;
}

double[] p = {0.5, 0.0};
LevenbergMarquardt lma = LevenbergMarquardt.fit(func, x, y, p);

assertEquals(0.0863, lma.sse, 1E-4); // The sum of squares due to error.
assertEquals(1.2260, lma.parameters[0], 1E-4);
assertEquals(1.8024, lma.parameters[1], 1E-4);
```



## 参考

- https://en.wikipedia.org/wiki/Levenberg%E2%80%93Marquardt_algorithm
- https://people.duke.edu/~hpgavin/lm.pdf
- https://heath.cs.illinois.edu/iem/optimization/index.php
