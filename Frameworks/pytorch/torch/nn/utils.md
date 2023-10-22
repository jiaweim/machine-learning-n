# torch.nn.utils

- [torch.nn.utils](#torchnnutils)
  - [rnn.PackedSequence](#rnnpackedsequence)
  - [rnn.pack\_padded\_sequence](#rnnpack_padded_sequence)
  - [rnn.pad\_packed\_sequence](#rnnpad_packed_sequence)
  - [rnn.pad\_sequence](#rnnpad_sequence)
  - [rnn.pack\_sequence](#rnnpack_sequence)
  - [参考](#参考)

***

## rnn.PackedSequence

```python
class torch.nn.utils.rnn.PackedSequence(
    data, 
    batch_sizes=None, 
    sorted_indices=None, 
    unsorted_indices=None)
```

保存打包序列数据。

所有 RNN 模块都支持打包序列作为输入。

> **NOTE**
> 不应该手动创建该类实例，而应该使用 `pack_padded_sequence()` 这样的函数来创建。
> 

**变量：**

- **data** (`Tensor`) – 包含打包序列的张量
- **batch_sizes** (`Tensor`) – integer 张量，包含每个时间步的 batch size
- **sorted_indices** (`Tensor`, optional) – integer 张量，Tensor of integers holding how this PackedSequence is constructed from sequences.
- **unsorted_indices** (`Tensor`, optional) – integer 张量，包含原始序列的顺序。

## rnn.pack_padded_sequence

```python
torch.nn.utils.rnn.pack_padded_sequence(
    input, 
    lengths, 
    batch_first=False, 
    enforce_sorted=True)
```

将填充序列打包。

`input` 的 size 为 `T x B x *`，其中 T 是最长序列长度（等于 `lengths[0]`），B 是 batch size，`*` 是余下维度。如果 `batch_first=True`，则要求输入 size 为 `B x T x *`。

对未排序的序列，使用 `enforce_sorted = False`。如果 `enforce_sorted=True`，则序列应该按长度降序排列，即 `input[:,0]` 应该最长，`input[:,B-1]` 最短。`enforce_sorted = True` 只对导出 ONNX 有用。

> **NOTE**
> 该函数接受至少有两个维度的任何输入。可以用它来打包标签，并用 RNN 的输出与标签计算损失。

**参数：**

- **input** (`Tensor`) – 填充过的变长序列 batch
- **lengths** (`Tensor` or `list(int)`) – list of sequence lengths of each batch element (must be on the CPU if provided as a tensor).

batch_first (bool, optional) – if True, the input is expected in B x T x * format.

enforce_sorted (bool, optional) – if True, the input is expected to contain sequences sorted by length in a decreasing order. If False, the input will get sorted unconditionally. Default: True.

**返回：**

- `PackedSequence ` 对象。

## rnn.pad_packed_sequence

```python
torch.nn.utils.rnn.pad_packed_sequence(
    sequence, 
    batch_first=False, 
    padding_value=0.0, 
    total_length=None)
```

填充打包的变长序列 batch。

它是 `pack_padded_sequence()` 的逆操作。

返回张量的 shape 为 `T x B x *`，其中 T 是最长序列长度，B 是 batch size。如果 `batch_first` 为 True，数据转换为 `B x T x *` 格式。

**示例：**

```python
>>> from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
>>> seq = torch.tensor([[1,2,0], [3,0,0], [4,5,6]])
>>> lens = [2, 1, 3]
>>> packed = pack_padded_sequence(seq, lens, batch_first=True, enforce_sorted=False)
>>> packed
PackedSequence(data=tensor([4, 1, 3, 5, 2, 6]), batch_sizes=tensor([3, 2, 1]),
               sorted_indices=tensor([2, 0, 1]), unsorted_indices=tensor([1, 2, 0]))
>>> seq_unpacked, lens_unpacked = pad_packed_sequence(packed, batch_first=True)
>>> seq_unpacked
tensor([[1, 2, 0],
        [3, 0, 0],
        [4, 5, 6]])
>>> lens_unpacked
tensor([2, 1, 3])
```

## rnn.pad_sequence

```python
torch.nn.utils.rnn.pad_sequence(
    sequences, 
    batch_first=False, 
    padding_value=0.0)
```

用 `padding_value` 填充变长张量列表。

`pad_sequence` 沿着一个新维度堆叠张量列表，并填充到相等长度。

假设 `B` 是 batch size，为 `sequences` 中序列个数。`T` 是最长序列的长度，`*` 表示余下维度。那么该函数返回 `T x B x *` 或 `B x T x *`。

**示例：**

```python
>>> from torch.nn.utils.rnn import pad_sequence
>>> a = torch.ones(25, 300)
>>> b = torch.ones(22, 300)
>>> c = torch.ones(15, 300)
>>> pad_sequence([a, b, c]).size()
torch.Size([25, 3, 300])
```

**参数：**

- **sequences** (`list[Tensor]`) – 变长序列 list
- **batch_first** (`bool`, optional) – True 时输出 `B x T x *`，False 输出 `T x B x *`。默认 False。
- **padding_value** (`float`, optional) – 填充值，默认 0.。

**返回：**

`batch_first=False` 时返回 shape 为 `T x B x *` 的张量，否则 shape 为 `B x T x *`。

## rnn.pack_sequence

```python
torch.nn.utils.rnn.pack_sequence(
    sequences, 
    enforce_sorted=True)
```

打包变长张量列表。

`sequences` 为张量列表，shape 为 `L x *`，其中 L 是序列长度，* 是余下维度。

对未排序的序列，使用 `enforce_sorted = False`。如果 `enforce_sorted = True`，则序列应该按照长度递减排序。`enforce_sorted = True` 仅用于导出 ONNX。

**示例：**

```python
>>> from torch.nn.utils.rnn import pack_sequence
>>> a = torch.tensor([1,2,3])
>>> b = torch.tensor([4,5])
>>> c = torch.tensor([6])
>>> pack_sequence([a, b, c])
PackedSequence(data=tensor([1, 4, 6, 2, 5, 3]), batch_sizes=tensor([3, 2, 1]), sorted_indices=None, unsorted_indices=None)
```

**参数：**

- **sequences** (`list[Tensor]`) – 序列 list，按长度降序排列
- **enforce_sorted** (`bool`, optional) – `True` 时检查输入序列是否按长度降序排列，`False` 则不检查。默认：`True`。

**返回：**

- `PackedSequence` 对象。

## 参考

- https://pytorch.org/docs/stable/nn.html#module-torch.nn.utils
