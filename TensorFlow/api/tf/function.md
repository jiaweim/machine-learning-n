# tf.function

- [tf.function](#tffunction)
  - [简介](#简介)
  - [特点](#特点)
  - [tf.function 创建多态调用](#tffunction-创建多态调用)
  - [Retracing](#retracing)
  - [输入签名](#输入签名)
  - [变量只能创建一次](#变量只能创建一次)
  - [Python 操作每次 trace 只执行一次](#python-操作每次-trace-只执行一次)
  - [使用类型注释来提高性能](#使用类型注释来提高性能)
  - [参数](#参数)
  - [参考](#参考)

Last updated: 2022-10-11, 16:08
****

## 简介

```python
tf.function(
    func=None,
    input_signature=None,
    autograph=True,
    jit_compile=None,
    reduce_retracing=False,
    experimental_implements=None,
    experimental_autograph_options=None,
    experimental_relax_shapes=None,
    experimental_compile=None,
    experimental_follow_type_hints=None
) -> tf.types.experimental.GenericFunction
```

将函数 `func` 编译为可调用的 TF graph。

> **WARNING**
> `experimental_compile` 参数已弃用，改用 `jit_compile`。
> `experimental_relax_shapes` 参数已启用，改用 `reduce_retracing`。

`tf.function` 构造了一个 `tf.types.experimental.GenericFunction`，通过 trace `func` 中 TF 操作的编译过程创建 TF graph，然后执行该 graph。例如：

```python
>>> @tf.function
... def f(x, y):
...     return x ** 2 + y
>>> x = tf.constant([2, 3])
>>> y = tf.constant([3, -2])
>>> f(x, y)
<tf.Tensor: shape=(2,), dtype=int32, numpy=array([7, 7])>
```

trace 编译允许在特定条件下执行非 TF 操作。一般来说，只有 TF 操作能保证在调用 `GenericFunction` 时运行并生成新的结果。

## 特点

- `func` 可以使用与数据相关的 Python 控制流语句，包括 `if`, `for`, `while`, `break`, `continue` 和 `return`：

```python
>>> @tf.function
... def f(x):
...     if tf.reduce_sum(x) > 0:
...         return x * x
...     else:
...         return -x // 2
>>> f(tf.constant(-2))
<tf.Tensor: shape=(), dtype=int32, numpy=1>
```

- `func` 的闭包可以包括 `tf.Tensor` 和 `tf.Variable` 对象：

```python
>>> @tf.function
... def f():
...     return x ** 2 + y
>>> x = tf.constant([-2, -3])
>>> y = tf.Variable([3, -2])
>>> f()
<tf.Tensor: shape=(2,), dtype=int32, numpy=array([7, 7])>
```

- `func` 也可以使用带副作用的操作，如 `tf.print`, `tr.Variable` 等：

```python
>>> v = tf.Variable(1)
>>> @tf.function
... def f(x):
...     for i in tf.range(x):
...         v.assign_add(i)
>>> f(3)
>>> v
<tf.Variable 'Variable:0' shape=() dtype=int32, numpy=4>
```

> **!IMPORTANT** 任何 Python 副作用（appending to a list, `print` 输出等）只在 trace `func` 执行一次。要在 `tf.function` 中执行副作用，需要将其实现为 TF 操作：

```python
>>> l = []
>>> @tf.function
... def f(x):
...     for i in x:
...         l.append(i + 1)  # Caution! Will only happen once when tracing
>>> f(tf.constant([1, 2, 3]))
>>> l
[<tf.Tensor 'while/add:0' shape=() dtype=int32>]
```

应该使用TF 集合，如 `tf.TensorArray`：

```python
>>> @tf.function
... def f(x):
...     ta = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)
...     for i in range(len(x)):
...         ta = ta.write(i, x[i] + 1)
...     return ta.stack()
>>> f(tf.constant([1, 2, 3]))
<tf.Tensor: shape=(3,), dtype=int32, numpy=array([2, 3, 4])>
```

## tf.function 创建多态调用

`tf.types.experimental.GenericFunction` 内部可能包含多个 `tf.types.experimental.ConcreteFunction`，每个对应特定数据类型或 shape 的参数，这是因为 TF 对特定 shape、dtype 的 graph 可以执行更多优化。tf.function 将纯 Python 值视为不透明对象（可以视为编译时常量），并为遇到的每组 Python 参数创建一个单独的 `tf.Graph`。

执行 `GenericFunction` 将根据参数类型和值选择并执行合适的 `ConcreteFunction`。

使用 `GenericFunction.get_concrete_function` 可以获得单独的 `ConcreteFunction`。可以使用和 `func` 一样的参数调用，返回 `tf.types.experimental.ConcreteFunction`。`ConcreteFunction` 由单个 tf.Graph 支持：

```python
>>> @tf.function
... def f(x):
...     return x + 1
>>> isinstance(f.get_concrete_function(1).graph, tf.Graph)
True
```

可以像执行 `GenericFunction` 一样执行 `ConcreteFunction`，但是输入仅限于它们专用的类型。

## Retracing

当使用新的 TF 类型、shape 或新的 Python 值作为参数调用 `GenericFunction` `时，会动态构建（trace）ConcreteFunction`。当 `GenericFunction` 构建一个新的 trace 时，就称 `func` 被 retraced。对 `tf.function` 来说，retracing 是一个常见的性能问题，因为它可能比执行已 trace 的 graph 慢很多。所以应当尽量减少代码中的 retracing。

> **Caution:** 给 `tf.function` 传入 Python 标量或 list 参数通常会导致 retrac。为了避免该情况，应尽可能将数值参数转换为 Tensor 后再传入。

```python
>>> @tf.function
... def f(x):
...     return tf.abs(x)
>>> f1 = f.get_concrete_function(1)
>>> f2 = f.get_concrete_function(2)  # Slow - compiles new graph
>>> f1 is f2
False
>>> f1 = f.get_concrete_function(tf.constant(1))
>>> f2 = f.get_concrete_function(tf.constant(2))  # Fast - reuses f1
>>> f1 is f2
True
```

对 Python 数值参数，在参数只有少量不同值时可以使用，如超参数神经网络的层数。

## 输入签名

对 Tensor 参数，`GenericFunction` 为每个 unique 输入 shape 和数据类型创建一个新的 `ConcreteFunction`。下例创建了两个单独的 `ConcreteFunction`，针对不同的输入 shape：

```python
>>> @tf.function
... def f(x):
...     return x + 1
>>> vector = tf.constant([1.0, 1.0])
>>> matrix = tf.constant([[3.0]])
>>> f.get_concrete_function(vector) is f.get_concrete_function(matrix)
False
```

可以选择向 `tf.function` 提供输入签名来控制这个过程。输入签名使用 `tf.TensorSpec` 对象指定函数每个 Tensor 参数的 shape 和类型。可以使用更 general 的 shape，这样可以确保只创建一个 `ConcreteFunction`，将 `GenericFunction` 限制为指定的 shape 和类型。当 Tensor 具有动态 shape 时，这是一种有效限制 retracing 的方法。

```python
>>> @tf.function(
...     input_signature=[tf.TensorSpec(shape=None, dtype=tf.float32)])
... def f(x):
...     return x + 1
>>> vector = tf.constant([1.0, 1.0])
>>> matrix = tf.constant([[3.0]])
>>> f.get_concrete_function(vector) is f.get_concrete_function(matrix)
True
```

## 变量只能创建一次

`tf.function` 只允许在第一次调用时创建新的 `tf.Variable` 对象：

```python
>>> class MyModule(tf.Module):
...     def __init__(self):
...         self.v = None
... 
...     @tf.function
...     def __call__(self, x):
...         if self.v is None:
...             self.v = tf.Variable(tf.ones_like(x))
...         return self.v * x
```

通常建议在 `tf.function` 外创建 ·。在简单情况下，跨越 `tf.function` 边界的持久化状态可以使用纯函数实现，其中状态由作为参数传递的 `tf.Tensor` 表示，并以返回值返回。

对比下面的两种风格：

```python
>>> state = tf.Variable(1)
>>> @tf.function
... def f(x):
...     state.assign_add(x)
>>> f(tf.constant(2))  # Non-pure functional style
>>> state
tf.Variable 'Variable:0' shape=() dtype=int32, numpy=3>
```

```python
>>> state = tf.constant(1)
>>> @tf.function
... def f(state, x):
...     state += x
...     return state
>>> state = f(state, tf.constant(2))  # Pure functional style
>>> state
<tf.Tensor: shape=(), dtype=int32, numpy=3>
```

## Python 操作每次 trace 只执行一次

`func` 可能同时包含 TF 操作和纯 Python 操作。但是在执行函数时只执行 TF 操作。Python 操作只在 trace 时运行一次。如果 TF 操作依赖于 Python 操作的结果，这些操作会在 graph 中冻结。

```python
>>> @tf.function
... def f(a, b):
...     print('this runs at trace time; a is', a, 'and b is', b)
...     return b
>>> f(1, tf.constant(1))
this runs at trace time; a is 1 and b is Tensor("b:0", shape=(), dtype=int32)
<tf.Tensor: shape=(), dtype=int32, numpy=1>
```

```python
>>> f(1, tf.constant(2))
<tf.Tensor: shape=(), dtype=int32, numpy=2>
```

```python
>>> f(2, tf.constant(1))
this runs at trace time; a is 2 and b is Tensor("b:0", shape=(), dtype=int32)
<tf.Tensor: shape=(), dtype=int32, numpy=1>
```

```python
>>> f(2, tf.constant(2))
<tf.Tensor: shape=(), dtype=int32, numpy=2>
```

## 使用类型注释来提高性能

将 `experimental_follow_type_hints` 和类型注释一起使用，可以自动将 Python 值转换为 `tf.Tensor` 从而减少 retracing。

```python
>>> @tf.function(experimental_follow_type_hints=True)
... def f_with_hints(x: tf.Tensor):
...     print('Tracing')
...     return x
>>> @tf.function(experimental_follow_type_hints=False)
... def f_no_hints(x: tf.Tensor):
...     print('Tracing')
...     return x
>>> f_no_hints(1)
Tracing
>>> f_no_hints(2)
Tracing
>>> f_with_hints(1)
Tracing
>>> f_with_hints(2)
tf.Tensor: shape=(), dtype=int32, numpy=2>
```

## 参数

|参数|说明|
|---|---|
|func|要编译的函数。如果 `func` 为 `None`，`tf.function` 返回一个可以用单个参数 `func` 调用的装饰器。换句话说，`tf.function(input_signature=...)(func)` 等价于 `tf.function(func, input_signature=...)`。前者可用作装饰器|
|input_signature|可能嵌套的 tf.TensorSpec 对象序列，用来指定提供给该函数的 Tensor 的 shape 和 dtype。`None` 表示为每个推断的输入签名实例化一个单独的函数。如果指定了 input_signature，那么 `func` 的输入必须为 Tensor，且 `func` 不能接受 `**kwargs`|
|autograph|在 trace graph 前是否在 `func` 上应用 autograph。数据依赖的 Python 控制流语句需要设置 `autograph=True`|
|jit_compile|`True` 时使用 XLA 编译函数。XLA 执行编译器优化，尝试生成更有效的代码。该操作可能大大提供性能。`True` 时整个函数由 XLA 编译，或者抛出`errors.InvalidArgumentError`。`None`（默认）在 TPU 上运行时使用 XLA 编译函数，在其它设备上则使用常规函数执行路径。`False` 时不使用 XLA 编译执行函数。当直接在 TPU 上运行多设备功能（如两个 TPU 内核，或一个 TPU 内核一个 CPU 内核），将此值设置为 `False`。并非所有函数都可以使用 XLA 编译，详情参考 [XLA 文档](https://tensorflow.org/xla/known_issues)|
|reduce_retracing|`True` 时 tf.function 会尝试减少 retracing 次数，例如使用更多更通用的 shape。这可以通过定制用户对象相关的 `tf.types.experimental.TraceType` 来控制|
|experimental_implements|If provided, contains a name of a "known" function this implements. For example "mycompany.my_recurrent_cell". This is stored as an attribute in inference function, which can then be detected when processing serialized function. See standardizing composite ops for details. For an example of utilizing this attribute see this example The code above automatically detects and substitutes function that implements "embedded_matmul" and allows TFLite to substitute its own implementations. For instance, a tensorflow user can use this attribute to mark that their function also implements embedded_matmul (perhaps more efficiently!) by specifying it using this parameter: @tf.function(experimental_implements="embedded_matmul") This can either be specified as just the string name of the function or a NameAttrList corresponding to a list of key-value attributes associated with the function name. The name of the function will be in the 'name' field of the NameAttrList. To define a formal TF op for this function implements, try the experimental composite TF project.
|experimental_autograph_options|可选的 `tf.autograph.experimental.Feature` 值|
|experimental_relax_shapes|弃用，改用 `reduce_retracing`|
|experimental_compile|启用，改用 'jit_compile'|
|experimental_follow_type_hints|When True, the function may use type annotations from func to optimize the tracing performance. For example, arguments annotated with tf.Tensor will automatically be converted to a Tensor.|

## 参考

- https://tensorflow.google.cn/api_docs/python/tf/function
