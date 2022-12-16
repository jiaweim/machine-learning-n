# Conv2d

## 简介

```python
class torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
```

2D 卷积层。

在最简单的情况，输入 $(N,C_{in},H,W)$ 和输出 $(N,C_{out},H_{out},W_{out})$ 的关系为：

$$out(N_i,C_{out_j})=bias(C_{out_j})$$



## 参考

- https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
