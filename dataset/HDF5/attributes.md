# Attributes

- [Attributes](#attributes)
  - [简介](#简介)
  - [API](#api)
  - [参考](#参考)

Last updated: 2022-10-16, 15:58
@author Jiawei Mao
****

## 简介

属性（Attributes）是使 HDF5 成为“自我描述”格式的关键部分。它们是直接连接到 `Group` 和 `Dataset` 对象的命名数据片段。这是 HDF5 存储元数据的标准方式。

每个 Group 或 Dataset 都有一个小的代理对象 `<obj>.attrs`。属性具有如下特征：

- 它们可以从任意标量或 NumPy 数组创建
- 每个 attribute 都应该很小（一般 < 64k）
- 没有部分 I/O (即切片)；必须读取整个 attribute

`.attrs` 代理对象类为 `AttributeManager`。该类支持 dict 样式接口。

attributes 默认以字母数字顺序迭代。但是，如果以 `track_order=True` 创建 group 或 dataset，则 HDF5 文件保留 attribute 的插入顺序，迭代顺序与其相同。后者与 Python 3.7+ dict 一致。

所有新创建的 group 和 dataset 的默认 `track_order` 可以通过 `h5.get_config().track_order` 全局指定。

## API

```python
class h5py.AttributeManager(parent)
```

`AttributeManager` 对象由 h5py 直接创建。应该通过 `group.attrs` 或 `dataset.attrs` 访问该类实例，而不是直接手动创建。

```python
__iter__()
```

返回一个迭代所有属性名的迭代器。

```python
__contains__(name)
```

确定是否包含指定属性名。

```python
__getitem__(name)
```

检索属性。

```python
__setitem__(name, val)
```

创建一个属性，覆盖已有同名属性。属性的 type 和 shape 由 h5py 自动确定。

```python
__delitem__(name)
```

删除指定名称属性。如果不存在抛出 `KeyError`。

```python
keys()
```

返回连接到该对象的所有属性的名称。
返回：set-like 对象。

```python
values()
```

所有绑定到该对象的所有属性的值。
返回：集合或 bag-like 对象。

```python
items()
```

获取对象所有属性的 `(name, value)` tuples。
返回：集合或 set-like 对象。

```python
get(name, default=None)
```

检索 `name`，如果没有该属性，返回 `default`。

```python
get_id(name)
```

返回命名属性的底层 `AttrID`。

```python
create(name, data, shape=None, dtype=None)
```

创建一个新的属性。覆盖已有属性。参数：

- `name` (String) – 新属性的名称
- `data` – 属性值，通过 `numpy.array(data)` 添加
- `shape` (Tuple) – 属性 shape。提供该参数会覆盖 `data.shape`，但是数据点总数必须一致。
- `dtype` (NumPy dtype) – 属性数据类型，提供该参数会覆盖 `data.dtype`。

```python
modify(name, value)
```

修改属性的值，同时保留其类型和 shape。和 `AttributeManager.__setitem__()` 不同，如果属性已存在，只修改其值。在与外部生成的文件（type 和 shape 不可更改）进行交互很有用。

如果属性不存在，则以默认 shape 和 type 创建新的属性。

参数：

- `name` (String) – 待修改属性名称
- `value` – 新的值，通过 `numpy.array(value)` 添加

## 参考
