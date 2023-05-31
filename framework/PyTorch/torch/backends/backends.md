# torch.backends

## 简介

`torch.backends` 用于设置 PyTorch 支持的各种后端的行为。

PyTorch 支持的后端：

- `torch.backends.cuda`
- `torch.backends.cudnn`
- `torch.backends.mps`
- `torch.backends.mkl`
- `torch.backends.mkldnn`
- `torch.backends.openmp`
- `torch.backends.opt_einsum`
- `torch.backends.xeon`

## torch.backends.cudnn

- `torch.backends.cudnn.deterministic`

`bool` 值，`True` 时 cuDNN 只使用确定性卷积算法。参考 `torch.are_deterministic_algorithms_enabled()` 和 `torch.use_deterministic_algorithms()`。

## 参考

- https://pytorch.org/docs/stable/backends.html
