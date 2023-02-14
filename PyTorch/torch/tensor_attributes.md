# Tensor Attributes

- [Tensor Attributes](#tensor-attributes)
  - [torch.device](#torchdevice)
  - [torch.memory\_format](#torchmemory_format)
  - [参考](#参考)

***

## torch.device

```python
class torch.device
```

`torch.device` 用于表示分配 `torch.Tensor` 的设备。

`torch.device` 包含设备类型（`'cpu'` or `'cuda'`）

## torch.memory_format

```python
class torch.memory_format
```

`torch.memory_format` 表示 `torch.Tensor` 使用的内存格式。

可选值：

- `torch.contiguous_format`: Tensor is or will be allocated in dense non-overlapping memory. Strides represented by values in decreasing order.

torch.channels_last: Tensor is or will be allocated in dense non-overlapping memory. Strides represented by values in strides[0] > strides[2] > strides[3] > strides[1] == 1 aka NHWC order.

torch.channels_last_3d: Tensor is or will be allocated in dense non-overlapping memory. Strides represented by values in strides[0] > strides[2] > strides[3] > strides[4] > strides[1] == 1 aka NDHWC order.

torch.preserve_format: Used in functions like clone to preserve the memory format of the input tensor. If input tensor is allocated in dense non-overlapping memory, the output tensor strides will be copied from the input. Otherwise output strides will follow torch.contiguous_format

## 参考

- https://pytorch.org/docs/stable/tensor_attributes.html
