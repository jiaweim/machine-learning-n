# torch.utils.data

## 简介

## 类

### torch.utils.data.DataLoader

```py
torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, sampler=None, batch_sampler=None, num_workers=0, collate_fn=None, pin_memory=False, drop_last=False, timeout=0, worker_init_fn=None, multiprocessing_context=None, generator=None, *, prefetch_factor=2, persistent_workers=False)
```

数据加载类。组合了数据集和采样，提供了对数据集的迭代功能。

`DataLoader` 支持映射和迭代样式数据集，支持单进程和多进程加载、自定义加载顺序以及可选的批处理和内存固定。




|参数|类型|说明|
|---|---|---|
|dataset|`Dataset`|数据集|



### torch.utils.data.Dataset

```py
torch.utils.data.Dataset
```

数据集抽象类。所有提供键到样本数据映射的数据集都应该继承该类。需要实现：

- `__getitem__()`，提供通过键查询样本数据的操作；
- `__len__()`，选择性覆盖，许多 `Sampler` 实现通过它获取数据集大小。


