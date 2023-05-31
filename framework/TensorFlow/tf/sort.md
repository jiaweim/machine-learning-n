# tf.sort

2022-02-23, 14:02
****

## 简介

张量排序。

```python
tf.sort(
    values, axis=-1, direction='ASCENDING', name=None
)
```

返回与 `values` 相同 dtype 和 shape 的张量，其元素沿指定 `axis` 排序。

## 参数

|参数|说明|
|---|---|
|values|1-D or higher numeric Tensor.|
|axis|The axis along which to sort. The default is -1, which sorts the last axis.|
|direction|The direction in which to sort the values ('ASCENDING' or 'DESCENDING').|
|name|Optional name for the operation.|

## 示例

```python
>>> a = [1, 10, 26.9, 2.8, 166.32, 62.3]
>>> tf.sort(a).numpy()
array([  1.  ,   2.8 ,  10.  ,  26.9 ,  62.3 , 166.32], dtype=float32)
```

降序排列：

```python
>>> tf.sort(a, direction='DESCENDING').numpy()
array([166.32,  62.3 ,  26.9 ,  10.  ,   2.8 ,   1.  ], dtype=float32)
```

对多维输入，可以通过 `axis` 参数控制排序沿哪个轴进行。默认 `axis=-1`，即对最里面的轴进行排序：

```python
>>> mat = [[3,2,1],
           [2,1,3],
           [1,3,2]]
>>> tf.sort(mat, axis=-1).numpy()
array([[1, 2, 3],
       [1, 2, 3],
       [1, 2, 3]])
>>> tf.sort(mat, axis=0).numpy()
array([[1, 1, 1],
       [2, 2, 2],
       [3, 3, 3]])
```

## 参考

- https://www.tensorflow.org/api_docs/python/tf/sort
