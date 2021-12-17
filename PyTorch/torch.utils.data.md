# torch.utils.data

- [torch.utils.data](#torchutilsdata)
  - [简介](#简介)
  - [数据集类型](#数据集类型)
    - [map 风格数据集](#map-风格数据集)
    - [Iterable 风格数据集](#iterable-风格数据集)
  - [数据加载顺序和 Sampler](#数据加载顺序和-sampler)
  - [批处理](#批处理)
    - [自动批处理](#自动批处理)
  - [API](#api)
    - [DataLoader](#dataloader)
    - [Dataset](#dataset)
  - [参考](#参考)

## 简介

PyTorch 数据加载工具的核心类是 `torch.utils.data.DataLoader`。它表示在数据集上的 Python 可迭代对象，支持如下功能：

- map 风格和 iterable 风格数据集；
- 自定义数据加载顺序；
- 自动 batching；
- 单进程和多进程加载数据；
- 自动内存锁定。

这些功能通过 `DataLoader` 构造函数参数配置：

```python
DataLoader(dataset, batch_size=1, shuffle=False, sampler=None,
           batch_sampler=None, num_workers=0, collate_fn=None,
           pin_memory=False, drop_last=False, timeout=0,
           worker_init_fn=None, *, prefetch_factor=2,
           persistent_workers=False)
```

## 数据集类型

`dataset` 是 `DataLoader` 构造函数中最重要的参数，它指示数据集对象。PyTorch 支持两种不同类型的数据集：

- map 风格
- iterable 风格

### map 风格数据集

map 风格的数据集实现了 `__getite__()` 和 `__len__()` 协议，表示从索引（可以是非整数）/键到数据样本的映射。

对这类数据，可以使用 `dataset[idx]` 从磁盘上的文件夹中读取第 `idx` 个图像及其对应的标签。

### Iterable 风格数据集

iterable 风格数据集是 `IterableDataset` 的子类，实现 `__iter__()` 协议，可以对数据集样本进行迭代。这种类型的数据集特别适用于随机读取操作非常昂贵甚至不太可能的情况。

例如，对这类数据集，当调用 `iter(dataset)` 时，返回从数据库、远程服务器甚至实时生成日志读取的数据流。

## 数据加载顺序和 Sampler

对 iterable 风格的数据集，顺序加载顺序完全由用户定义的 iterable 控制。这样更容易实现块读取和动态 batch size。

本节剩余部分主要讨论 map 样式数据集的情况。`torch.utils.data.Sampler` 类用于指定数据加载中使用的 索引/键 的序列。

## 批处理

`DataLoader` 支持通过参数 `batch_size`, `drop_last` 和 `batch_sampler` 将单个数据集样本整理成多个批次。

### 自动批处理

这是最常见的情况（默认值），

## API

### DataLoader

```py
torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, sampler=None, batch_sampler=None, num_workers=0, collate_fn=None, pin_memory=False, drop_last=False, timeout=0, worker_init_fn=None, multiprocessing_context=None, generator=None, *, prefetch_factor=2, persistent_workers=False)
```

数据加载类。组合 `Dataset` 和 `Sampler`，提供了对数据集的迭代功能。

`DataLoader` 支持映射和迭代样式数据集，支持单进程和多进程加载、自定义加载顺序以及可选的批处理和内存固定。

- `dataset`

要从中加载数据的数据集。

- `batch_size`

批量大小，即每个 batch 包含的样本数，默认为 1.

- `shuffle`

是否在每个 epoch 都打乱数据，默认为 `False`。

- `num_workers`

用来加载数据的子进程数。0 表示在主进程中加载数据，默认为 0.



### Dataset

```py
torch.utils.data.Dataset
```

数据集抽象类。所有提供键到样本数据映射的数据集都应该继承该类。需要实现：

- `__getitem__()`，提供通过键查询样本数据的操作；
- `__len__()`，选择性覆盖，许多 `Sampler` 实现通过它获取数据集大小。


## 参考

- https://pytorch.org/docs/stable/data.html
