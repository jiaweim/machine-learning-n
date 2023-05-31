
# 简介

# 理论
对 $m$ 个数据点 $(x_1, y_1)$,$(x_2, y_2)$,...,$(x_m, y_m)$，模型函数 $y=f(x,\beta)$ 除了依赖于变量 $x$，还依赖于 $n$ 个参数 $\beta=(\beta_1,\beta_2,...,\beta_n)$，其中 $m\geq n$。目的是找到合适的参数向量 $\beta$ 使得函数能够和数据最好的拟合，对最小二乘，使得平方差最小：  
$S=\displaystyle\sum_{i-1}^mr_i^2$

其中差值 $r_i$ 定义为：  
$r_i=y_i-f(x_i,\beta)$



目标函数能够写成 m 个函数平方和的优化问题，就是最小二乘问题：

$minF(x)\stackrel{def}{=}\displaystyle\sum_{i=1}^mf_i^2(x)$

而当 $f_i(x)$ 是关于 x 的非线性函数，即为非线性优化问题，此时需要利用 Taylor 一阶展开近似 $f_i(x)$。

## $f_i(x)$ 的一阶展开


# 参考
- https://en.wikipedia.org/wiki/Non-linear_least_squares