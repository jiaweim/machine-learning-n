# tf.linspace

2021-12-28, 12:24

## 签名

```python
tf.linspace(
    start, stop, num, name=None, axis=0
)
```

沿轴 `axis`，从起点 `start` 开始生成一组间隔均匀的数值序列。

返回和 `start` 相同类型的 `Tensor`。

## 参数

- `start`

第一个值，`Tensor` 类型，支持类型：`bfloat16`, `float32`, `float64`。

- `stop`

最后一个值，`Tensor` 类型，支持类型和 `start` 一样。

- `num`

生成的数值个数，`Tensor` 类型，支持类型：`int32` 和 `int64`。

如果 `num > 1`，则数值之间的间隔为 $(stop-start)/(num-1)$，最后一个值刚好为 `stop`。

如果 `num <= 0`，抛出 `ValueError`。

- `name`

操作名称，可选。

- `axis`

操作针对的轴，尽在提供 N-D 张量时使用。

## 实例

例如：

```python
tf.linspace(10.0, 12.0, 3, name="linspace") => [ 10.0  11.0  12.0]
```

`start` 和 `stop` 可以是任意 size 的 tensor:

```python
> tf.linspace([0., 5.], [10., 40.], 5, axis=0)
<tf.Tensor: shape=(5, 2), dtype=float32, numpy=
array([[ 0.  ,  5.  ],
       [ 2.5 , 13.75],
       [ 5.  , 22.5 ],
       [ 7.5 , 31.25],
       [10.  , 40.  ]], dtype=float32)>
```

`axis` 指定生成值的轴，返回的张量与轴对应的维数为 `num`：

```python
> tf.linspace([0., 5.], [10., 40.], 5, axis=-1)
<tf.Tensor: shape=(2, 5), dtype=float32, numpy=
array([[ 0.  ,  2.5 ,  5.  ,  7.5 , 10.  ],
       [ 5.  , 13.75, 22.5 , 31.25, 40.  ]], dtype=float32)>
```

## 参考

- https://www.tensorflow.org/api_docs/python/tf/linspace
