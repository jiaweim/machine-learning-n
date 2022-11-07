# 使用 tf.function 提高性能

- [使用 tf.function 提高性能](#使用-tffunction-提高性能)
  - [简介](#简介)
  - [设置](#设置)
  - [基础](#基础)
    - [使用](#使用)
    - [Tracing](#tracing)
      - [什么是 tracing](#什么是-tracing)
      - [tracing 规则](#tracing-规则)
    - [控制 retracing](#控制-retracing)
      - [传递固定的 `input_signature` 给 `tf.function`](#传递固定的-input_signature-给-tffunction)
      - [使用未知维度来获得灵活性](#使用未知维度来获得灵活性)
      - [使用张量代替 Python 字面量](#使用张量代替-python-字面量)
      - [使用 trace 协议](#使用-trace-协议)
    - [获得 concrete function](#获得-concrete-function)
    - [获取 graphs](#获取-graphs)
    - [Debugging](#debugging)
  - [AutoGraph 转换](#autograph-转换)
    - [条件语句](#条件语句)
    - [循环](#循环)
      - [迭代 Python 数据](#迭代-python-数据)
  - [参考](#参考)

***

## 简介

TF 2 默认启用 eager 执行，eager 执行用户接口直观灵活，运行一次性操作更容易、快捷，但这可能牺牲**性能**和**可部署性**。

> eager 执行使用简单，但性能和可部署性较差

可以使用 `tf.function` 将 Python 函数转换为 graph。即 `tf.function` 是一个转换工具，将 Python 代码转换为不依赖于 Python 的数据流 graph。这有助于创建高性能的可移植模型，是使用 `SavedModel` 的基础。

下面介绍 `tf.function` 的概念和工作机制，从而可以更有效地使用它。

主要建议：

- 在 eager 模型下调试，然后使用 `@tf.function` 进行装饰；
- 不要依赖于 Python 副作用，如对象 mutation 或 list append 操作；
- `tf.function` 与 TF 操作运行时效果最好，NumPy 和 Python 调用被转换为**常量**。

## 设置

```python
import tensorflow as tf
```

定义一个辅助函数来演示可能遇到的各种错误类型：

```python
import traceback
import contextlib


# Some helper code to demonstrate the kinds of errors you might encounter.
@contextlib.contextmanager
def assert_raises(error_class):
    try:
        yield
    except error_class as e:
        print('Caught expected exception \n  {}:'.format(error_class))
        traceback.print_exc(limit=2)
    except Exception as e:
        raise e
    else:
        raise Exception('Expected {} to be raised but no error was raised!'.format(
            error_class))
```

## 基础

### 使用

定义的 `Function`（如使用 `@tf.function` 装饰器）就像 TF 核心操作，可以 eager 执行，也可以计算梯度等。

```python
@tf.function  # tf.function 装饰器将 `add` 转换为 `Function`.
def add(a, b):
    return a + b

add(tf.ones([2, 2]), tf.ones([2, 2]))  #  [[2., 2.], [2., 2.]]
```

```txt
<tf.Tensor: shape=(2, 2), dtype=float32, numpy=
array([[2., 2.],
       [2., 2.]], dtype=float32)>
```

```python
v = tf.Variable(1.0)
with tf.GradientTape() as tape:
    result = add(v, 1.0)
tape.gradient(result, v)
```

```txt
<tf.Tensor: shape=(), dtype=float32, numpy=1.0>
```

`Function` 可以嵌套使用：

```python
@tf.function
def dense_layer(x, w, b):
    return add(tf.matmul(x, w), b)


dense_layer(tf.ones([3, 2]), tf.ones([2, 2]), tf.ones([2]))
```

```txt
<tf.Tensor: shape=(3, 2), dtype=float32, numpy=
array([[3., 3.],
       [3., 3.],
       [3., 3.]], dtype=float32)>
```

`Function` 可以比 eager 代码快很多，特别是对包含许多小操作的 graph。但是对包含少量昂贵操作的 graph （如卷积），性能提升可能不大。

```python
import timeit

conv_layer = tf.keras.layers.Conv2D(100, 3)


@tf.function
def conv_fn(image):
    return conv_layer(image)


image = tf.zeros([1, 200, 200, 100])
# Warm up
conv_layer(image);
conv_fn(image)
print("Eager conv:", timeit.timeit(lambda: conv_layer(image), number=10))
print("Function conv:", timeit.timeit(lambda: conv_fn(image), number=10))
print("Note how there's not much difference in performance for convolutions")
```

```txt
Eager conv: 0.00861140000006344
Function conv: 0.009842699999921933
Note how there's not much difference in performance for convolutions
```

### Tracing

下面介绍 `Function` 的底层是如何工作的，包括一些实现细节。了解 tracing 发生的原因和时间，就能更有效地使用 `tf.function`。

#### 什么是 tracing

`Function` 在 TF Graph 中运行程序。然而，TF eager 程序中的部分内容无法表示为 `tf.Graph`。例如，Python 支持多态，但 `tf.Graph` 要求其输入为指定的数据类型和 shape。还有一些副作用操作，如读取命令行参数、抛出错误，或者使用更复杂的 Python 对象，这些任务都不能在 `tf.Graph` 中运行。

`Function` 通过将代码分为两个阶段来弥补这一差距：

1. 第一阶段，称为 **tracing**，`Function` 创建一个新的 `tf.Graph`。Python 代码正常运行，但是 TF 操作（如两个 Tensor 相加）被延迟，它们被 `tf.Graph` 捕获，而不运行。
2. 第二阶段，运行包含第一阶段中延迟的所有内容的 `tf.Graph`。这个阶段比 tracing 快得多。

根据其输入，`Function` 被调用时并不总运行第一阶段，具体可参考下面的 [tracing 规则](#tracing-规则)。跳过第一阶段，只执行第二阶段是 TensorFlow 提高性能的关键。

当 `Function` 确定要 trace 时，tracing 后会立刻运行第二阶段，所以调用 `Function` 会同时创建和运行 `tf.Graph`。稍后会介绍如何使用 `get_concrete_function` 只运行 tracing 阶段。

当将不同类型的参数传入 `Function`，会同时运行两个阶段：

```python
@tf.function
def double(a):
    print("Tracing with", a)
    return a + a


print(double(tf.constant(1)))
print()
print(double(tf.constant(1.1)))
print()
print(double(tf.constant("a")))
print()
```

```txt
Tracing with Tensor("a:0", shape=(), dtype=int32)
tf.Tensor(2, shape=(), dtype=int32)

Tracing with Tensor("a:0", shape=(), dtype=float32)
tf.Tensor(2.2, shape=(), dtype=float32)

Tracing with Tensor("a:0", shape=(), dtype=string)
tf.Tensor(b'aa', shape=(), dtype=string)
```

如果以相同类型参数重复调用 `Function`，TF 会跳过 tracing 阶段并重用之前 trace 的 graph，因为反正生成 graph 是相同的。

```python
# This doesn't print 'Tracing with ...'
print(double(tf.constant("b")))
```

```txt
tf.Tensor(b'bb', shape=(), dtype=string)
```

可以使用 `pretty_printed_concrete_signatures()` 查看所有可用的 traces：

```python
print(double.pretty_printed_concrete_signatures())
```

```txt
double(a)
  Args:
    a: int32 Tensor, shape=()
  Returns:
    int32 Tensor, shape=()

double(a)
  Args:
    a: float32 Tensor, shape=()
  Returns:
    float32 Tensor, shape=()

double(a)
  Args:
    a: string Tensor, shape=()
  Returns:
    string Tensor, shape=()
```

到目前为止，我们已看到 tf.function 在 TF graph tracing 的逻辑上创建了一个缓存的动态调度层，具体来说：

- `tf.Graph` 是对 TF 计算的一个原始的、与语言无关的、可移植的表示；
- 一个 `ConcreteFunction` wrap 一个 `tf.Graph`；
- `Function` 管理 `ConcreteFunction` 缓存，根据输入选择正确缓存的 `ConcreteFunction`；
- `tf.function` wrap 一个 Python 函数，返回 `Function` 对象；
- Tracing 创建一个 `tf.Graph`，并将其 wrap 到一个 `ConcreteFunction` `中，ConcreteFunction` 也称为 trace。

#### tracing 规则

调用 `Function` 时，它使用每个参数的 `tf.types.experimental.TraceType`与已有的 `ConcreteFunction` 进行匹配。如果找到匹配的 `ConcreteFunction`，则将调用分派给它；如果未匹配到任何 `ConcreteFunction`，则 trace 一个新的 `ConcreteFunction`。

如果找到多个匹配项，则选择最 specific 的签名。匹配通过 [subtyping](https://en.wikipedia.org/wiki/Subtyping) 完成，和 C++ 或 Java 的函数调用一样。例如，`TensorShape([1, 2])` 是 `TensorShape([None, None])` 的子类型，因此使用 `TensorShape([1, 2])` 调用 tf.function 可以被分配到使用 `TensorShape([None, None])` 生成的 `ConcreteFunction`，但如果已有一个和 `TensorShape([1, None])` 对应的 `ConcreteFunction`，则优先使用它，因为它更 specific。

`TraceType` 按如下规则从输入参数确定：

- 对 `Tensor`，类型由 `dtype` 和 `shape` 参数化；ranked shape 是 unranked shape 的子类型；固定维度是未知维度的子类型。
- 对 `Variable`，类型类似于 `Tensor`，但还包含变量的唯一资源 ID，该 ID 对正确连接控件依赖项是必需的。
- 对 Python primitive 值，类型对应值本身。例如，值 `3` 的 `TraceType` 是 `LiteralTraceType<3>`，而不是 `int`。
- 对 Python 有序容器，如 `list`, `tuple` 等，类型由包含的元素类型参数化；例如，`[1, 2]` 的类型为 `ListTraceType<LiteralTraceType<1>, LiteralTraceType<2>>`，`[2, 1]` 的类型为 `ListTraceType<LiteralTraceType<2>, LiteralTraceType<1>>`，两者不同。
- 对 `dict` 这样的 Python 映射，类型是相同 key 到值类型的映射。例如，`{1: 2, 3: 4}` 的类型为 `appingTraceType<<KeyValue<1, LiteralTraceType<2>>>, <KeyValue<3, LiteralTraceType<4>>>>`，然而和有序容器不同的是，对 dict `{1: 2, 3: 4}` 和 `{3: 4, 1: 2}` 的类型相同。
- 实现 `__tf_tracing_type__` 方法的 Python 对象的类型是该方法返回的类型。
- 其它 Python 对象的类型是通用的 `TraceType`，它使用对象的 equality 和 hashing 来匹配。（注意：它依赖于对象的 [weakref](https://docs.python.org/3/library/weakref.html)，因此对象在作用域中且未删除时才能工作）

> **Note:** `TraceType` 基于 `Function` 的输入参数，因此仅仅改变全局变量或自由变量不会创建新的 trace。

### 控制 retracing

Retracing，即 `Function` 创建多个 trace 的行为，用于确保 TF 为每组输入生成正确的 graph。但是 trace 是一项昂贵的操作。如果 `Function` 为每个调用都 retrace 一个新的 graph，你会发现代码的执行速度比没有使用 `tf.function` 要慢很多。

可以使用以下技术控制 tracing 行为。

#### 传递固定的 `input_signature` 给 `tf.function`

```python
@tf.function(input_signature=(tf.TensorSpec(shape=[None], dtype=tf.int32), ))
def next_collatz(x):
    print("Tracing with", x)
    return tf.where(x % 2 == 0, x // 2, 3 * x + 1)


print(next_collatz(tf.constant([1, 2])))
# You specified a 1-D tensor in the input signature, so this should fail.
with assert_raises(ValueError):
    next_collatz(tf.constant([[1, 2], [3, 4]]))

# You specified an int32 dtype in the input signature, so this should fail.
with assert_raises(ValueError):
    next_collatz(tf.constant([1.0, 2.0]))
```

```txt
Tracing with Tensor("x:0", shape=(None,), dtype=int32)
tf.Tensor([4 1], shape=(2,), dtype=int32)
Caught expected exception 
  <class 'ValueError'>:
Caught expected exception 
  <class 'ValueError'>:
Traceback (most recent call last):
  File "C:\Users\happy\AppData\Local\Temp\ipykernel_15120\1159894030.py", line 9, in assert_raises
    yield
  File "C:\Users\happy\AppData\Local\Temp\ipykernel_15120\4031220677.py", line 10, in <module>
    next_collatz(tf.constant([[1, 2], [3, 4]]))
ValueError: Python inputs incompatible with input_signature:
  inputs: (
    tf.Tensor(
[[1 2]
 [3 4]], shape=(2, 2), dtype=int32))
  input_signature: (
    TensorSpec(shape=(None,), dtype=tf.int32, name=None)).
Traceback (most recent call last):
  File "C:\Users\happy\AppData\Local\Temp\ipykernel_15120\1159894030.py", line 9, in assert_raises
    yield
  File "C:\Users\happy\AppData\Local\Temp\ipykernel_15120\4031220677.py", line 14, in <module>
    next_collatz(tf.constant([1.0, 2.0]))
ValueError: Python inputs incompatible with input_signature:
  inputs: (
    tf.Tensor([1. 2.], shape=(2,), dtype=float32))
  input_signature: (
    TensorSpec(shape=(None,), dtype=tf.int32, name=None)).
```

#### 使用未知维度来获得灵活性

由于 TF 根据张量的 shape 来匹配它们，因此使用 `None` 维度作为通配符允许 `Function` 对不同 size 的输入复用 trace。比如长度不同的序列，不同 batch 的图片大小不同，都会出现输入大小不一的情况。

```python
@tf.function(input_signature=(tf.TensorSpec(shape=[None], dtype=tf.int32), ))
def g(x):
    print('Tracing with', x)
    return x


# No retrace!
print(g(tf.constant([1, 2, 3])))
print(g(tf.constant([1, 2, 3, 4, 5])))
```

```txt
Tracing with Tensor("x:0", shape=(None,), dtype=int32)
tf.Tensor([1 2 3], shape=(3,), dtype=int32)
tf.Tensor([1 2 3 4 5], shape=(5,), dtype=int32)
```

#### 使用张量代替 Python 字面量

通常，Python 实参用于设置超参数和 graph 结构，如 `num_layers=10`、`training=True` 或 `nonlinearity='relu'`。因此，如果 Python 参数发生变化，那么必须 retrace graph 也是合理的。

但是，Python 参数也可能用来控制 graph 构造之外的事。此时，Python 值的更改可能触发不必要的 retracing。以下面的训练循环为例，其中 AutoGraph 会动态展开。虽然有多个 traces，但是生成的 graph 实际上都是一样的，因此 retracing 是不必要的。

```python
def train_one_step():
    pass


@tf.function
def train(num_steps):
    print("Tracing with num_steps = ", num_steps)
    tf.print("Executing with num_steps = ", num_steps)
    for _ in tf.range(num_steps):
        train_one_step()


print("Retracing occurs for different Python arguments.")
train(num_steps=10)
train(num_steps=20)

print()
print("Traces are reused for Tensor arguments.")
train(num_steps=tf.constant(10))
train(num_steps=tf.constant(20))
```

```txt
Retracing occurs for different Python arguments.
Tracing with num_steps =  10
Executing with num_steps =  10
Tracing with num_steps =  20
Executing with num_steps =  20

Traces are reused for Tensor arguments.
Tracing with num_steps =  Tensor("num_steps:0", shape=(), dtype=int32)
Executing with num_steps =  10
Executing with num_steps =  20
```

如果需要强制 retracing，可以创建新的 `Function`。单独的 `Function` 对象不会共享 traces。

```python
def f():
    print('Tracing!')
    tf.print('Executing')


tf.function(f)()
tf.function(f)()
```

```txt
Tracing!
Executing
Tracing!
Executing
```

#### 使用 trace 协议

如果可能，应该将 Python 类型转换为 `tf.experimental.ExtensionType`。另外，`ExtensionType` 的 `TraceType` 是与其关联的 `tf.TypeSpec`。因此，如果需要，可以通过覆盖默认的 `tf.TypeSpec` 来控制 `ExtensionType` 的 trace 协议。详情可参考[扩展类型指南](https://tensorflow.google.cn/guide/extension_type)。

否则，为了直接控制 `Function` 对特定 Python 类型何时 retrace，可以自己实现 trace 协议。

```python
@tf.function
def get_mixed_flavor(fruit_a, fruit_b):
    return fruit_a.flavor + fruit_b.flavor


class Fruit:
    flavor = tf.constant([0, 0])


class Apple(Fruit):
    flavor = tf.constant([1, 2])


class Mango(Fruit):
    flavor = tf.constant([3, 4])


# As described in the above rules, a generic TraceType for `Apple` and `Mango`
# is generated (and a corresponding ConcreteFunction is traced) but it fails to
# match the second function call since the first pair of Apple() and Mango()
# have gone out out of scope by then and deleted.
get_mixed_flavor(Apple(), Mango())  # Traces a new concrete function
get_mixed_flavor(Apple(), Mango())  # Traces a new concrete function again

# However, each subclass of the `Fruit` class has a fixed flavor, and you
# can reuse an existing traced concrete function if it was the same
# subclass. Avoiding such unnecessary tracing of concrete functions
# can have significant performance benefits.


class FruitTraceType(tf.types.experimental.TraceType):

    def __init__(self, fruit_type):
        self.fruit_type = fruit_type

    def is_subtype_of(self, other):
        return (type(other) is FruitTraceType
                and self.fruit_type is other.fruit_type)

    def most_specific_common_supertype(self, others):
        return self if all(self == other for other in others) else None

    def __eq__(self, other):
        return type(
            other) is FruitTraceType and self.fruit_type == other.fruit_type

    def __hash__(self):
        return hash(self.fruit_type)


class FruitWithTraceType:

    def __tf_tracing_type__(self, context):
        return FruitTraceType(type(self))


class AppleWithTraceType(FruitWithTraceType):
    flavor = tf.constant([1, 2])


class MangoWithTraceType(FruitWithTraceType):
    flavor = tf.constant([3, 4])


# Now if you try calling it again:
get_mixed_flavor(AppleWithTraceType(),
                 MangoWithTraceType())  # Traces a new concrete function
get_mixed_flavor(AppleWithTraceType(),
                 MangoWithTraceType())  # Re-uses the traced concrete function
```

```txt
<tf.Tensor: shape=(2,), dtype=int32, numpy=array([4, 6])>
```

### 获得 concrete function

每 trace 一个函数，就会创建一个新的 concrete function。使用 `get_concrete_function` 可以直接获得 concrete function。

```python
print("Obtaining concrete trace")
double_strings = double.get_concrete_function(tf.constant("a"))
print("Executing traced function")
print(double_strings(tf.constant("a")))
print(double_strings(a=tf.constant("b")))
```

```txt
Obtaining concrete trace
Executing traced function
tf.Tensor(b'aa', shape=(), dtype=string)
tf.Tensor(b'bb', shape=(), dtype=string)
```

```python
# You can also call get_concrete_function on an InputSpec
double_strings_from_inputspec = double.get_concrete_function(
    tf.TensorSpec(shape=[], dtype=tf.string))
print(double_strings_from_inputspec(tf.constant("c")))
```

```txt
tf.Tensor(b'cc', shape=(), dtype=string)
```

print `ConcreteFunction` 会显示其输入参数（包括类型）和输出类型。

```python
print(double_strings)
```

```txt
ConcreteFunction double(a)
  Args:
    a: string Tensor, shape=()
  Returns:
    string Tensor, shape=()
```

也可以直接查看函数签名：

```python
print(double_strings.structured_input_signature)
print(double_strings.structured_outputs)
```

```txt
((TensorSpec(shape=(), dtype=tf.string, name='a'),), {})
Tensor("Identity:0", shape=(), dtype=string)
```

使用不兼容类型调用 concrete trace 会抛出错误：

```python
with assert_raises(tf.errors.InvalidArgumentError):
    double_strings(tf.constant(1))
```

```txt
Caught expected exception 
  <class 'tensorflow.python.framework.errors_impl.InvalidArgumentError'>:
Traceback (most recent call last):
  File "C:\Users\happy\AppData\Local\Temp\ipykernel_15120\1159894030.py", line 9, in assert_raises
    yield
  File "C:\Users\happy\AppData\Local\Temp\ipykernel_15120\2585775733.py", line 2, in <module>
    double_strings(tf.constant(1))
tensorflow.python.framework.errors_impl.InvalidArgumentError: cannot compute __inference_double_141 as input #0(zero-based) was expected to be a string tensor but is a int32 tensor [Op:__inference_double_141]
```

可以发现，Python 参数在 `ConcreteFunction` 的输入签名中做了特殊处理。在 TF2.3 之前，Python 参数直接从 `ConcreteFunction` 的签名中删除，从 TF2.3 开始，Python 参数保留在签名中，但在 trace 过程中限制为设置的值。

```python
@tf.function
def pow(a, b):
    return a**b


square = pow.get_concrete_function(a=tf.TensorSpec(None, tf.float32), b=2)
print(square)
```

```txt
ConcreteFunction pow(a, b=2)
  Args:
    a: float32 Tensor, shape=<unknown>
  Returns:
    float32 Tensor, shape=<unknown>
```

```python
assert square(tf.constant(10.0)) == 100

with assert_raises(TypeError):
    square(tf.constant(10.0), b=3)
```

```txt
Caught expected exception 
  <class 'TypeError'>:
Traceback (most recent call last):
  File "d:\conda3\envs\tf\lib\site-packages\tensorflow\python\eager\function.py", line 1617, in _call_impl
    return self._call_with_flat_signature(args, kwargs,
  File "d:\conda3\envs\tf\lib\site-packages\tensorflow\python\eager\function.py", line 1662, in _call_with_flat_signature
    raise TypeError(f"{self._flat_signature_summary()} got unexpected "
TypeError: pow(a) got unexpected keyword arguments: b.

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "C:\Users\happy\AppData\Local\Temp\ipykernel_15120\1159894030.py", line 9, in assert_raises
    yield
  File "C:\Users\happy\AppData\Local\Temp\ipykernel_15120\2991996595.py", line 4, in <module>
    square(tf.constant(10.0), b=3)
TypeError: ConcreteFunction pow(a, b) was constructed with int value 2 in b, but was called with int value 3.
```

### 获取 graphs

每个 `ConcreteFunction` 都是对一个 tf.Graph 的包装。虽然通常不需要检索实际的 `tf.Graph`，但是你可以从任何 `ConcreteFunction` 中轻松获取它。

```python
graph = double_strings.graph
for node in graph.as_graph_def().node:
    print(f'{node.input} -> {node.name}')
```

```txt
[] -> a
['a', 'a'] -> add
['add'] -> Identity
```

### Debugging

通常在 eager 模式下调试代码比在 `tf.function` 中更容易。在使用 `tf.function` 注释函数前，应该确保代码在 eager 模式下运行没有错误。为了帮助调试，可以使用 `tf.config.run_functions_eagerly(True)` 来全局开启或禁用 `tf.function`。

当只在 tf.function 中出现问题时，建议：

- Python `print` 调用只在 tracing 时运行，这有助于确定函数何时 (re)traced。
- `tf.print` 每次都会执行，可以帮助在执行期间跟踪中间值。
- `tf.debugging.enable_check_numerics` 是一种查找 NaN 和 Inf 创建位置的简单方法。
- `pdb` (Python 调试器) 可以帮助理解 tracing 期间发生了什么。（注意：`pdb` 会跳入 AutoGraph 转换的源代码中）

## AutoGraph 转换

AutoGraph 是 `tf.function` 默认使用的库，用来将 Python eager 代码转换为兼容 graph 的 TF 操作。包括 `if`, `for`, `while` 等控制流。

像 tf.cond 和 tf.while_loop 这样的 TF 操作可以继续工作，但是在用 Python 编写代码时，Python 控制流更容易编写和理解。

```python
# A simple loop


@tf.function
def f(x):
    while tf.reduce_sum(x) > 1:
        tf.print(x)
        x = tf.tanh(x)
    return x


f(tf.random.uniform([5]))
```

```txt
[0.281349301 0.302678704 0.061591506 0.982579 0.350352168]
[0.274153411 0.293762058 0.0615137368 0.754180193 0.336687833]
[0.26748535 0.285593718 0.061436262 0.637636244 0.32451725]
[0.261283368 0.278074205 0.0613590814 0.563287914 0.313585699]
[0.255495489 0.27112174 0.0612821914 0.510412872 0.303695619]
[0.250077486 0.264668286 0.0612055846 0.470266789 0.294690937]
[0.244991496 0.25865671 0.0611292645 0.438414872 0.286446601]
[0.240204811 0.253038675 0.0610532276 0.412329614 0.278860956]
[0.235689193 0.247772902 0.0609774776 0.390448898 0.271850526]
[0.231419861 0.242823988 0.0609020069 0.371747166 0.265345901]
[0.227375224 0.23816134 0.0608268119 0.355518967 0.259288877]
[0.223536208 0.233758301 0.0607519 0.34126091 0.253630251]
[0.219885901 0.229591593 0.0606772639 0.328602582 0.248328075]
[0.216409296 0.225640744 0.0606029034 0.317264557 0.243346363]
[0.213093013 0.221887738 0.0605288148 0.307031393 0.238654]
[0.2099251 0.21831654 0.0604549944 0.297734022 0.234223977]
[0.206894785 0.214912921 0.060381446 0.289237559 0.230032682]
<tf.Tensor: shape=(5,), dtype=float32, numpy=
array([0.20399237, 0.21166414, 0.06030817, 0.2814329 , 0.22605935],
      dtype=float32)>
```

如果有兴趣，可以查看 autograph 生成的代码：

```python
print(tf.autograph.to_code(f.python_function))
```

```txt
def tf__f(x):
    with ag__.FunctionScope('f', 'fscope', ag__.ConversionOptions(recursive=True, user_requested=True, optional_features=(), internal_convert_user_code=True)) as fscope:
        do_return = False
        retval_ = ag__.UndefinedReturnValue()

        def get_state():
            return (x,)

        def set_state(vars_):
            nonlocal x
            (x,) = vars_

        def loop_body():
            nonlocal x
            ag__.converted_call(ag__.ld(tf).print, (ag__.ld(x),), None, fscope)
            x = ag__.converted_call(ag__.ld(tf).tanh, (ag__.ld(x),), None, fscope)

        def loop_test():
            return ag__.converted_call(ag__.ld(tf).reduce_sum, (ag__.ld(x),), None, fscope) > 1
        ag__.while_stmt(loop_test, loop_body, get_state, set_state, ('x',), {})
        try:
            do_return = True
            retval_ = ag__.ld(x)
        except:
            do_return = False
            raise
        return fscope.ret(retval_, do_return)
```

### 条件语句

AutoGraph 将一些 `if <condition>` 语句转换为等价的 `tf.cond`。`<condition>` 是 Tensor 时进行此替换，否则 `if` 语句将作为 Python 条件执行。

Python 条件语句在 tracing 期间执行，因此条件语句的一个分支将被添加到 graph。如果没有 AutoGraph，对数据依赖的控制流，该 traced graph 无法采用其它分支。

`tf.cond` trace 并将条件语句的两个分支添加到 graph 中，在执行时动态选择一个分支。tracing 可能产生意想不到的副作用，更多信息可参考 [AutoGraph tracing effects](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/autograph/g3doc/reference/control_flow.md#effects-of-the-tracing-process)。

```python
@tf.function
def fizzbuzz(n):
    for i in tf.range(1, n + 1):
        print('Tracing for loop')
        if i % 15 == 0:
            print('Tracing fizzbuzz branch')
            tf.print('fizzbuzz')
        elif i % 3 == 0:
            print('Tracing fizz branch')
            tf.print('fizz')
        elif i % 5 == 0:
            print('Tracing buzz branch')
            tf.print('buzz')
        else:
            print('Tracing default branch')
            tf.print(i)


fizzbuzz(tf.constant(5))
fizzbuzz(tf.constant(20))
```

```txt
Tracing for loop
Tracing fizzbuzz branch
Tracing fizz branch
Tracing buzz branch
Tracing default branch
1
2
fizz
4
buzz
1
2
fizz
4
buzz
fizz
7
8
fizz
buzz
11
fizz
13
14
fizzbuzz
16
17
fizz
19
buzz
```

### 循环

AutoGraph 将一些 `for` 和 `while` 语句转换为等效的 TF 循环操作，如 `tf.while_loop`。如果没有转换，它们将作为 Python 循环执行。

这种替换在以下情况下执行：

- `for x in y`：如果 `y` 是 Tensor，转换为 `tf.while_loop`。当 y 是 tf.data.Dataset 时，将生成 `tf.data.Dataset` 操作的组合。
- `while <condition>`：如果 `<condition>` 是 Tensor，转换为 `tf.while_loop`。

Python 循环在 tracing 期间执行，为循环的每次迭代添加额外操作到 `tf.Graph`。

TF 循环 trace 循环体，并动态选择在执行时运行多少次迭代。循环体只在生成的 `tf.Graph` 中出现一次。

#### 迭代 Python 数据

一个常见的问题是在 `tf.function` 中迭代 Python/NumPy 数据。这类循环会在 trace 期间执行，并在循环的每次迭代中将模型的一个副本添加到 `tf.Graph` 中。

如果要将整个训练循环包装到 `tf.function` 中，最安全的方法是将数据包装为 `tf.data.Dataset`，这样 AutoGraph 会动态展开训练循环。

```python
def measure_graph_size(f, *args):
    g = f.get_concrete_function(*args).graph
    print("{}({}) contains {} nodes in its graph".format(
        f.__name__, ', '.join(map(str, args)), len(g.as_graph_def().node)))


@tf.function
def train(dataset):
    loss = tf.constant(0)
    for x, y in dataset:
        loss += tf.abs(y - x)  # Some dummy computation.
    return loss


small_data = [(1, 1)] * 3
big_data = [(1, 1)] * 10
measure_graph_size(train, small_data)
measure_graph_size(train, big_data)

measure_graph_size(
    train,
    tf.data.Dataset.from_generator(lambda: small_data, (tf.int32, tf.int32)))
measure_graph_size(
    train,
    tf.data.Dataset.from_generator(lambda: big_data, (tf.int32, tf.int32)))
```

```txt
train([(1, 1), (1, 1), (1, 1)]) contains 11 nodes in its graph
train([(1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1)]) contains 32 nodes in its graph
train(<FlatMapDataset element_spec=(TensorSpec(shape=<unknown>, dtype=tf.int32, name=None), TensorSpec(shape=<unknown>, dtype=tf.int32, name=None))>) contains 6 nodes in its graph
train(<FlatMapDataset element_spec=(TensorSpec(shape=<unknown>, dtype=tf.int32, name=None), TensorSpec(shape=<unknown>, dtype=tf.int32, name=None))>) contains 6 nodes in its graph
```



## 参考

- https://tensorflow.google.cn/guide/function
