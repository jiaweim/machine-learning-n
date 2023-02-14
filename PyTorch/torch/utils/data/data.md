# torch.utils.data

- [torch.utils.data](#torchutilsdata)
  - [简介](#简介)
  - [Dataset 类型](#dataset-类型)
    - [map 样式](#map-样式)
    - [iterable 样式](#iterable-样式)
  - [数据加载顺序和 `Sampler`](#数据加载顺序和-sampler)
  - [加载 Batched 和 Non-Batched 数据](#加载-batched-和-non-batched-数据)
    - [自动 batching (默认)](#自动-batching-默认)
    - [禁用自动 batching](#禁用自动-batching)
    - [collate\_fn](#collate_fn)
  - [单进程和多进程](#单进程和多进程)
    - [单进程加载数据（默认）](#单进程加载数据默认)
    - [多进程加载数据](#多进程加载数据)
      - [平台特异性行为](#平台特异性行为)
      - [多进程数据加载的随机性](#多进程数据加载的随机性)
  - [锁页内存（Memory Pinning）](#锁页内存memory-pinning)
  - [API](#api)
    - [torch.utils.data.DataLoader](#torchutilsdatadataloader)
    - [torch.utils.data.Dataset](#torchutilsdatadataset)
    - [torch.utils.data.IterableDataset](#torchutilsdataiterabledataset)
    - [torch.utils.data.TensorDataset](#torchutilsdatatensordataset)
    - [torch.utils.data.ConcatDataset](#torchutilsdataconcatdataset)
    - [torch.utils.data.ChainDataset](#torchutilsdatachaindataset)
    - [torch.utils.data.Subset](#torchutilsdatasubset)
    - [torch.utils.data.\_utils.collate.collate](#torchutilsdata_utilscollatecollate)
    - [torch.utils.data.default\_collate](#torchutilsdatadefault_collate)
    - [torch.utils.data.default\_convert](#torchutilsdatadefault_convert)
    - [torch.utils.data.get\_worker\_info](#torchutilsdataget_worker_info)
    - [torch.utils.data.random\_split](#torchutilsdatarandom_split)
    - [torch.utils.data.Sampler](#torchutilsdatasampler)
    - [torch.utils.data.SequentialSampler](#torchutilsdatasequentialsampler)
    - [torch.utils.data.RandomSampler](#torchutilsdatarandomsampler)
    - [torch.utils.data.SubsetRandomSampler](#torchutilsdatasubsetrandomsampler)
    - [torch.utils.data.WeightedRandomSampler](#torchutilsdataweightedrandomsampler)
    - [torch.utils.data.BatchSampler](#torchutilsdatabatchsampler)
    - [torch.utils.data.distributed.DistributedSampler](#torchutilsdatadistributeddistributedsampler)
  - [参考](#参考)

Last updated: 2023-01-31, 14:56
****

## 简介

PyTorch 加载数据的核心是 `torch.utils.data.DataLoader` 类。它表示数据集上的 Python 可迭代对象，支持：

- map 样式和 iterable 样式数据集；
- 自定义数据加载顺序；
- 自动 batching；
- 单进程和多进程加载数据；
- 自动锁业内存。

这些选项由 `DataLoader` 的构造函数参数配置，`DataLoader` 的签名：

```python
DataLoader(dataset, 
    batch_size=1, # 自动 batching
    shuffle=False, # 是否顺序采样
    sampler=None, # Sampler，一次采 1 个样
    batch_sampler=None, # Sampler，一次采多个样
    num_workers=0, 
    collate_fn=None,
    pin_memory=False, 
    drop_last=False, # batch
    timeout=0,
    worker_init_fn=None, 
    *, 
    prefetch_factor=2,
    persistent_workers=False
)
```

## Dataset 类型

`DataLoader` 构造函数的最重要参数是 `dataset`，指定数据集对象。PyTorch 支持两种类型的数据集：

- map 样式
- iterable 样式

### map 样式

map 数据集扩展 `Dataset`，实现 `__getitem__()` 和 `__len__()` 协议，表示从 index/key (可能为非整数)到数据样本的映射。

例如，使用 `dataset[idx]` 访问 map 数据集，可以从磁盘读取文件夹中第 `idx` 个图像及其标签。

### iterable 样式

iterable 数据集扩展 `IterableDataset` 类，实现 `__iter__()` 协议，表示可迭代数据集。

这类数据集适合随机读取无法实现或代价高的情况，batch size 取决于获取的数据。

例如，当对 iterable 数据集调用 `iter(dataset)`，可以返回从数据库、远程服务器甚至实时生成的日志读取的数据流。

> **NOTE:** 在多进程中使用 `IterableDataset`，每个工作进程复制相同的数据集对象，因此必须对副本进行不同的配置，以避免数据重复。

## 数据加载顺序和 `Sampler`

对 iterable 数据集，数据加载顺序完全取决于自定义的 iterable。因此很容易实现 chunk-reading 和动态 batch size (例如，每次 yield 一批样本)。

下面主要讨论 map 数据集。`torch.utils.data.Sampler` 类用于指定 index/key 的顺序，是数据集 index 上的可迭代对象。例如，在随机梯度下降（SGD）中，`Sampler` 可以随机排列 index 列表，然后每次 yield 一个 index（对 mini-batch SGD，每次 yield 小批量 index）。

使用 `DataLoader` 的 `shuffle` 参数设置采用方式（顺序采样还是随机采用）。另外，还可以使用 `sampler` 参数指定自定义 `Sampler` 对象，每次 yield 下一个 index/key。

自定义一次生成一批 index 的 `Sampler`，使用 `batch_sampler` 参数。也可以使用 `batch_size` 和 `drop_last` 参数启动自动 batching。

> `sampler` 和 `batch_sampler` 都与 iterable 数据集不兼容，因为 iterable 数据集没有 key 或 index。

## 加载 Batched 和 Non-Batched 数据

DataLoader 通过 `batch_size`, `drop_last`, `batch_sampler` 和 `collate_fn` 参数可以将提取的样本收集起来自动整理为 batch。

### 自动 batching (默认)

这是最常用的功能，即收集一批数据，将它们合并成包含 batch 维度的张量。

当 `batch_size` (默认 1) 不是 `None`，`DataLoader` yield batch 样本而非单个样本。`batch_size` 和 `drop_last` 用来指定 `DataLoader` 获取数据集 keys。对 map dataset，还可以指定 `batch_sampler`，从而每次 yield key list。

> **NOTE:** `batch_size` 和 `drop_last` 参数本质上用来从 `sampler` 构造 `batch_sampler`。对 map dataset，`sampler` 要么由用户提供，要么基于 `shuffle` 参数构建。对 iterable dataset，`sampler` 是一个虚拟的无效采样器。

> **NOTE:** 在多进程中使用 iterable 数据集，`drop_last` 参数会删除每个工作进程中数据集副本的最后一个不完整的 batch。 

在使用 `sampler` 的索引获取样本列表后，使用 `collate_fn` 参数指定的函数将样本列表整理成 batch。

对 map 数据集，`collate_fn` 的应用方式大致等同于：

```python
for indices in batch_sampler:
    yield collate_fn([dataset[i] for i in indices])
```

对 iterable 数据集，大致等同于：

```python
dataset_iter = iter(dataset)
for indices in batch_sampler:
    yield collate_fn([next(dataset_iter) for _ in indices])
```

可以自定义 `collate_fn` 函数，例如，将序列数据填充到 batch 最长的序列。

### 禁用自动 batching

在某些情况，用户可能希望手动处理 batching，或者简单地加载单个样本。例如，直接加载批量数据可能效率更高（例如，从数据库批量读取或读取连续的内存块），或者 batch size 依赖于数据，或者是专门处理单个样本的程序。在这些情况下，就不适合使用自动 batching，而是让 `DataLoader` 直接返回 `dataset` 对象的每个成员。

当 `batch_size` 和 `batch_sampler` 都为 `None` (`batch_sampler` 默认为 `None`)，将禁用自动 batching。从 `dataset` 获得的每个样本直接由 `collate_fn` 指定的函数进行处理。

禁用自动 batching 后，默认 `collate_fn` 只是将 NumPy 数组转换为 PyTorch 张量，其它保持不变。

此时从 map 数据集加载数据大致相当于：

```python
for index in sampler:
    yield collate_fn(dataset[index])
```

而从 iterable 数据集加载数据大致相等于：

```python
for data in iter(dataset):
    yield collate_fn(data)
```

### collate_fn

在启用或禁用自动 batching 时，`collate_fn` 的用法略有不同：

- **禁用**自动 batching 时，对每个样本单独调用 `collate_fn`，从 `DataLoader` 迭代器 yield 输出。此时，默认 `collate_fn` 只是将 NumPy 数组转换为 PyTorch 张量。
- **启用**自动 batching 时，每次对样本 list 调用 `collate_fn`。将输入样本列表整理为一个 batch 后从 `DataLoader` 迭代器 yield 输出。

下面详细介绍默认 `collate_fn`。

假设每个样本由一张 3 通道图像和一个类标签组成，即每个样本为一个 tuple `(image, class_index)`，默认 `collate_fn` 将样本列表（list of tuple）整理成包含批量图像张量和类标签张量的单个 tuple。总结一下就是，默认 `collate_fn` 功能：

- 总是在前面添加一个新维度作为 batch 维度
- 自动将 NumPy 数组和 Python 数值转换为 PyTorch 张量
- 保留数据结构，例如，如果样本是 dict，则输出也是 dict，key 保持不变，但是值变为 batched 张量。

可以自定义 `collate_fn` 来实现特殊的 batching，例如，沿着其它维度而非第一个维度进行整理，填充不同长度的序列，以及添加对自定义数据类型的支持。

如果遇到 `DataLoader` 的输出的维度或类型与预期不符，就可能是 `collate_fn` 问题。

## 单进程和多进程

`DataLoader` 默认使用单进程加载数据。

在 Python 中，全局解释器锁（Global Interpreter Lock, GIL）不允许跨线程执行 Python 代码。为了避免加载数据阻塞计算，PyToch 提供了一个简单参数来执行多进程加载数据，设置 `num_workers` 为正整数即可。

### 单进程加载数据（默认）

在该模式下，获取数据与 `DataLoader` 在同一个进程，因此，数据加载可能阻塞计算。不过，当用于进程间共享数据的资源（如共享内存）有限时，或者当整个数据集很小可以完全加载到内存时，推荐使用单进程。

另外，单进程出错信息更容易解读，有利于调试。

### 多进程加载数据

将 `num_workers` 设置为正整数，将启用指定数目的工作进程来加载数据。

> **WARNING**
> 迭代多次后，对父进程中的所有 Python 对象，在加载工作进程中访问占用的 CPU 内存与父进程中相同。当 `Dataset` 包含大量数据（如加载大量文件名列表），或使用许多 workers（总内存消耗为 `number of workers * size of parent process`）。最简单的解决方案，用不会重新计算的表示形式替代 Python 对象，如 Pandas, NumPy 或 PyArrow 对象。

在该模式下，每次创建 `DataLoader` 的迭代器（如调用 `enumerate(dataloader)`），就会创建 `num_workers` 个工作进程。此时，`dataset`, `collate_fn` 和 `worker_init_fn` 被传递给每个工作进程，用于初始化和提取数据，因此数据集访问及其内部 IO、transforms (包括 `collate_fn`) 都是在工作进程中执行。

`torch.utils.data.get_worker_info()` 返回工作进程的各种信息（包括 worker id, 数据集副本，初始化 seed 等），在主进程中返回 `None`。用户可以在数据集代码和 `worker_init_fn` 中使用这些信息来单独配置每个数据集副本，并确定代码是否在工作进程中执行。对分隔数据集非常有用。

对 map 数据集，在主进程中使用 `sampler` 生成 indices，然后发送到各个工作进程。因此，任何洗牌随机化操作都是在主进程中执行的，它通过指定 indices 来引导数据加载。

对 iterable 数据集，由于每个工作进程都持有 `dataset` 的副本，因此直接多进程加载通常会导致数据重复。使用 `torch.utils.data.get_worker_info()` 和 `worker_init_fn` 用户可以独立配置每个副本。出于同样的原因，在多进程加载中，对 iterable 数据集 `drop_last` 参数会删除每个工作进程中最后一个不完整的 batch。

迭代结束或迭代器被垃圾回收，工作线程会自动关闭。

> **WARNING**
> 在多进程加载中不建议返回 CUDA 张量，处理起来太复杂。建议使用自动锁业内存（即设置 `pin_memory=True`），启用到 CUDA GPU 的快速数据传输。

#### 平台特异性行为

由于工作进程依赖于 [multiprocessing](https://docs.python.org/3/library/multiprocessing.html#module-multiprocessing) Python 包，因此 Windows 和 Unix 上的行为有所不同：

- 在 Unix，`multiprocessing` 的默认启动方法为 `fork()`。使用 `fork()`，子进程通常可以通过克隆的地址空间直接访问 `dataset` 和 Python 参数函数；
- 在 Windows 或 MacOS，`multiprocessing` 的默认启动方法为 `spawn()`。`spawn()` 会启动另一个解释器来运行 main 脚本，然后内部工作函数通过 `pickle` 序列化接收 `dataset`, `collate_fn` 和其它参数。

这种单独序列化意味着在使用多进程加载数据时，应采取如下两个操作以确保与 Windows 兼容：

- 将 main 脚本的大部分代码放在 `if __name__ == '__main__':` 中，以确保启动工作进程时不会再次运行（更可能是报错）。可以将 dataset 和 `DataLoader` 实例的创建逻辑放在这里，从而不会在工作进程中再次执行；
- 将自定义的 `collate_fn`, `worker_init_fn` 和 `dataset` 声明为顶级定义，放在 `__main__` 外。以确保它们在工作进程可用，因为函数是通过引用 pickle，而非 bytecode。

#### 多进程数据加载的随机性

每个工作进程默认将 PyTorch seed 设置为 `base_seed + worker_id`，其中 `base_seed` 是主进程使用其 RNG 或指定 `generator` 生成的一个 long。然而，其它库的 seed 可能会在初始化工作进程时被复制，导致每个工作进程返回相同的随机数。

在 `worker_init_fn` 中可以使用 `torch.utils.data.get_worker_info().seed` 或 `torch.initial_seed()` 查看每个 worker 的 PyTorch seed，并在加载数据前使用它设置其它库的 seed。

## 锁页内存（Memory Pinning）

主机中的内存有两种存在方式，一是锁定内存（page-locked），二是非锁定内存，**锁定内存**存放的内容不与主机的虚拟内存进行交换（注：虚拟内存就是硬盘），而**非锁定内存**在主机内存不足时，会将数据存放到虚拟内存。

当数据在锁定内存，从主机到 GPU 复制的速度要快得多。设置 `DataLoader` 的 `pin_memory=True` 后，加载数据时会自动将获得的数据张量放在锁定内存，使得数据传输到 CUDA GPU 更快。

默认内存锁定逻辑只识别张量和包含张量的 map 和 iterable。对自定义类型的 batch（自定义 `collate_fn` 返回自定义 batch 类型），或 batch 的元素是自定义类型，锁定逻辑无法识别，从而不使用锁定内存。对自定义类型应该锁定内存，需要在自定义类型中定义 `pin_memory()` 方法。例如：

```python
class SimpleCustomBatch:
    def __init__(self, data):
        transposed_data = list(zip(*data))
        self.inp = torch.stack(transposed_data[0], 0)
        self.tgt = torch.stack(transposed_data[1], 0)

    # custom memory pinning method on custom type
    def pin_memory(self):
        self.inp = self.inp.pin_memory()
        self.tgt = self.tgt.pin_memory()
        return self

def collate_wrapper(batch):
    return SimpleCustomBatch(batch)

inps = torch.arange(10 * 5, dtype=torch.float32).view(10, 5)
tgts = torch.arange(10 * 5, dtype=torch.float32).view(10, 5)
dataset = TensorDataset(inps, tgts)

loader = DataLoader(dataset, batch_size=2, collate_fn=collate_wrapper,
                    pin_memory=True)

for batch_ndx, sample in enumerate(loader):
    print(sample.inp.is_pinned())
    print(sample.tgt.is_pinned())
```

## API

### torch.utils.data.DataLoader

```python

```

### torch.utils.data.Dataset

```python
class torch.utils.data.Dataset(*args, **kwds)
```

`Dataset` 抽象类。

所有 map 数据集应该继承该类。需要实现两个方法：

- `__getitem__()` 给定 key 返回对应样本；
- `__len__()` 返回数据量，可选。

> **NOTE**
> `DataLoader` 默认构造一个 index sampler 生成整数索引。要实现非整数 key 的 map 数据集，需要自定义 sampler。

### torch.utils.data.IterableDataset

### torch.utils.data.TensorDataset

### torch.utils.data.ConcatDataset

### torch.utils.data.ChainDataset

### torch.utils.data.Subset

### torch.utils.data._utils.collate.collate

### torch.utils.data.default_collate

### torch.utils.data.default_convert

### torch.utils.data.get_worker_info

### torch.utils.data.random_split

### torch.utils.data.Sampler

```python
class torch.utils.data.Sampler(data_source)
```

所有 `Sampler` 的基类。

每个 `Sampler` 子类必须实现两个方法：

- `__iter__()` 用来迭代数据集元素的索引
- `__len__()` 迭代器包含的元素个数

> **NOTE**
> `DataLoader` 不要求 `__len__()` 方法，但任何涉及 `DataLoader` 长度的计算都需要用到。

### torch.utils.data.SequentialSampler

```python
class torch.utils.data.SequentialSampler(data_source)
```

按顺序采样。

**参数：**

- **data_source** (`Dataset`) – 数据来源

### torch.utils.data.RandomSampler

```python
class torch.utils.data.RandomSampler(data_source, 
    replacement=False, 
    num_samples=None, 
    generator=None)
```

随机采样。如果 `replacement=False`，则从打乱的数据集中进行采样。如果 `replacement=True`，则可以指定抽样个数 `num_samples`.

**参数：**

- **data_source** (`Dataset`) – 数据集
- **replacement** (`bool`) – `True` 表示按需返回抽样，默认 `False`
- **num_samples** (`int`) – 抽样个数，默认 `len(dataset)`
- **generator** (`Generator`) – 抽样中使用的 Generator

### torch.utils.data.SubsetRandomSampler

```python
class torch.utils.data.SubsetRandomSampler(indices, generator=None)
```

从给定索引列表中随机抽样，不放回抽样。

**参数：**

- **indices** (`sequence`) – 索引序列
- **generator** (`Generator`) – 用于抽样的 Generator

### torch.utils.data.WeightedRandomSampler

### torch.utils.data.BatchSampler

### torch.utils.data.distributed.DistributedSampler



## 参考

- https://pytorch.org/docs/stable/data.html
