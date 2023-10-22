# Graph

Last updated: 2022-09-13, 16:07
@author Jiawei Mao
****

## 简介

```python
tf.Graph()
```

TF 的计算表示为数据流的 graph。

`tf.function` 使用 Graph 来表示函数的计算。每个 Graph 包含一组 `tf.Operation` 对象（计算单元）和 `tf.Tensor` 对象（数据单元）。

## 直接使用 Graph (已弃用)

可以和 TF1 一样直接构造和使用 `tf.Graph`，而无需 `tf.function`，但是这种方式已弃用，推荐使用 `tf.function`。如果直接使用 graph，则还需要使用其它弃用的 TF1 类来执行 graph，如 `tf.compat.v1.Session`。

可以使用 `tf.Graph.as_default` 注册默认 graph，这样操作就不会立刻执行，而是添加到 graph。例如：

```python
g = tf.Graph()
with g.as_default():
  # Define operations and tensors in `g`.
  c = tf.constant(30.0)
  assert c.graph is g
```

`tf.compat.v1.get_default_graph()` 可用来获得默认 graph。

> **!IMPORTANT** 该类创建 graph 不是线程安全的。所有操作都应该从单个线程创建，或者必须提供同步支持。无额外说明，所有方法都不是线程安全的。

`Graph` 实例支持任意数量由名称识别的 "collections"。为了便于构建大型 graph，collections 可用于存储一组相关对象，例如 `tf.Variable` 使用集合 `tf.GraphKeys.GLOBAL_VARIABLES` 存储 graph 构建期间创建的所有变量。调用者可以通过指定新名称来定义其它集合。

## 属性

|属性|说明|
|---|---|
|building_function|Returns True if this graph represents a function.|
|collections|Returns the names of the collections known to this graph.|
|finalized|True if this graph has been finalized.|
|graph_def_versions|The GraphDef version information of this graph.For details on the meaning of each version, see GraphDef.|
|seed|The graph-level random seed of this graph.|
|version|Returns a version number that increases as ops are added to the graph.Note that this is unrelated to the tf.Graph.graph_def_versions.|

## 方法

### add_to_collection

```python
add_to_collection(
    name, value
)
```

将 `value` 存储在给定 `name` 的集合中。

注意，collection 不是 set，因此可以多次向集合添加同一个值。

|参数|说明|
|---|---|
|name|集合名称。GraphKeys 类包含许多标准集合名称|
|value|要添加到集合的值|


## 参考

- https://www.tensorflow.org/api_docs/python/tf/Graph
