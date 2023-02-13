# 损失函数

## 概述

损失函数是预测值与已知答案的差距。

主流 loss 有三种计算方法：

- MSE（Mean Squared Error）
- 自定义
- CE（Cross Entropy）

## MSE

平方损失函数（Quadratic Loss Function）经常用在预测标签 y 为实数值的任务中，定义为：

$$
L(y,f(x;\theta)) = \frac{1}{2} (y-f(x;\theta))^2
$$

平方损失函数一般不适用于分类问题。

Python 实现：

```py
def mean_squared_error(y, t):
    """
    均方差损失函数
    :param y: 输出值
    :param t: 监督数据，即正确值
    :return: 损失值
    """
    return 0.5 * np.sum((y - t) ** 2)
```

## binary cross-entropy

## 参考

- https://towardsdatascience.com/understanding-binary-cross-entropy-log-loss-a-visual-explanation-a3ac6025181a
- [为何出现验证损失小于训练损失？](https://pyimagesearch.com/2019/10/14/why-is-my-validation-loss-lower-than-my-training-loss/)