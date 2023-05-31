# tf.random.set_seed

2022-02-23, 10:47
****

## 简介

设置全局的随机 seed。

```python
tf.random.set_seed(
    seed
)
```

依赖于随机 seed 的操作实际上从两个种子派生而来：global 和 operation-level seeds。该方法设置全局 seed。

全局 seed 与操作级（operation-level）seed 的相互作用如下：

1. 如果既没有设置全局 seed，也没有设置操作 seed，则该操作使用随机 seed。
2. 如果设置了全局 seed，但是没设置操作 seed：TF 会选择一个操作 seed，与全局 seed 一起，以便获得唯一的随机序列。在相同版本的 TF中，此序列是确定的。但是，在不同版本中，该序列可能不同，如果你的代码依赖于特定的 seed 才能工作，则应该显式指定全局和操作级 seed。
3. 如果设置了操作 seed，但没设置全局 seed，则使用默认的全局 seed 与操作 seed 一起确定随机序列。
4. 如果同时指定全局 seed 和操作 seed，则两个 seed 一起确定随机序列。

下面用示例进行说明。

- 如果既没有指定全局 seed，也没有指定操作 seed，则每次调用随机操作，以及重新运行，都会得到不同的结果

```python
>>> print(tf.random.uniform([1])) # A1
tf.Tensor([0.5735663], shape=(1,), dtype=float32)
>>> print(tf.random.uniform([1])) # A2
tf.Tensor([0.8158809], shape=(1,), dtype=float32)
```

然后关闭程序重新运行，结果都不一样了：

```python
>>> print(tf.random.uniform([1])) # A3
tf.Tensor([0.533373], shape=(1,), dtype=float32)
>>> print(tf.random.uniform([1])) # A4
tf.Tensor([0.9002656], shape=(1,), dtype=float32)
```

- 如果只设置全局 seed，则每次调用随机操作获得不同结果，但每次重新运行程序时，结果都相同

```python
tf.random.set_seed(1234)
print(tf.random.uniform([1]))  # A1
print(tf.random.uniform([1]))  # A2
```

```sh
tf.Tensor([0.5380393], shape=(1,), dtype=float32)
tf.Tensor([0.3253647], shape=(1,), dtype=float32)
```

关闭程序，重新运行，依然得到 A1 和 A2。

这里，第二次调用 `tf.random.uniform` 获得 A2 而不是 A1，是因为第二次调用使用了不同的操作 seed。

`tf.function` 的作用类似于重新运行程序，当设置了全局 seed，但没有设置操作 seed 时，每个 `tf.function` 生成的随机数序列是相同的：

```python
tf.random.set_seed(1234)

@tf.function
def f():
  a = tf.random.uniform([1])
  b = tf.random.uniform([1])
  return a, b

@tf.function
def g():
  a = tf.random.uniform([1])
  b = tf.random.uniform([1])
  return a, b

print(f())  # prints '(A1, A2)'
print(g())  # prints '(A1, A2)'
```

- 如果只设置操作 seed，则每次调用随机操作都会得到不同结果，但每次重新运行程序都会得到相同结果（和只设置全局 seed 效果一样）

```python
print(tf.random.uniform([1], seed=1))  # generates 'A1'
print(tf.random.uniform([1], seed=1))  # generates 'A2'
```

关闭后，重新运行程序，输出结果有一样。

那这次设置了相同操作 seed，第二次调用 `tf.random.uniform` 得到 A2 而不是 A1 呢？这是因为对相同参数调用，TensorFlow 使用相同的 `tf.random.uniform` 内核（即内部表示），内核维护一个计数器，每次调用 `uniform` 计数器递增，从而生成不同的结果。

调用 `tf.random.set_seed` 会重置该计数器：

```python
tf.random.set_seed(1234)
print(tf.random.uniform([1], seed=1))  # generates 'A1'
print(tf.random.uniform([1], seed=1))  # generates 'A2'
tf.random.set_seed(1234)
print(tf.random.uniform([1], seed=1))  # generates 'A1'
print(tf.random.uniform([1], seed=1))  # generates 'A2'
```

当多个相同的随机操作被包装在 `tf.function` 中，因为操作不再共享同一个计数器，所以行为会发生变化：

```python
@tf.function
def foo():
  a = tf.random.uniform([1], seed=1) # a 和 b 计数器不同
  b = tf.random.uniform([1], seed=1)
  return a, b
print(foo())  # prints '(A1, A1)'
print(foo())  # prints '(A2, A2)'

@tf.function
def bar():
  a = tf.random.uniform([1])
  b = tf.random.uniform([1])
  return a, b
print(bar())  # prints '(A1, A2)'
print(bar())  # prints '(A3, A4)'
```

第二次调用 `foo` 返回 '(A2, A2)' 而不是 '(A1, A1)'，是因为 `tf.random.uniform` 维护了一个内部计数器。如果希望 `foo` 每次都返回 '(A1, A1)'，可以使用无状态的随机操作，如 `tf.random.stateless_uniform`。

## 参考

- https://www.tensorflow.org/api_docs/python/tf/random/set_seed
