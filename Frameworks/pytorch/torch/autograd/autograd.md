# torch.autograd

## 简介

`torch.autograd` 提供任意标量值函数的自动微分的类和函数。

它只需要对现有代码进行少量更改：在声明 `Tensor` 时使用 `requires_grad=True` 关键字参数。目前只支持 float 类型 `Tensor`: half, float, double, bfloat16 和 complex 类型 `Tensor`: cfloat, cdouble。

- `backward`

计算指定张量相对 leaf 的梯度之和。

- `grad`

计算并返回输出相对于出入的梯度之和。

## forward 模式自动微分

> [!WARNING]
>
> 此 API 处于测试阶段。尽管函数签名基本不会变化，但正在改进其运算符覆盖范围。



## 参考

- https://pytorch.org/docs/stable/autograd.html