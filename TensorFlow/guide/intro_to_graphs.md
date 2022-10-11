# Graph 和 tf.function 简介

- [Graph 和 tf.function 简介](#graph-和-tffunction-简介)
  - [速览](#速览)
  - [简介](#简介)
    - [什么是 graph](#什么是-graph)
    - [graph 的优点](#graph-的优点)
  - [配置](#配置)
  - [graph 的使用](#graph-的使用)
    - [将 Python 函数转换为 graph](#将-python-函数转换为-graph)
    - [多态性：一个 `Function`，多个 graphs](#多态性一个-function多个-graphs)
  - [tf.function 使用](#tffunction-使用)
    - [graph 执行和即时执行](#graph-执行和即时执行)
    - [non-strict 执行](#non-strict-执行)
    - [tf.function 最佳实践](#tffunction-最佳实践)
  - [提速效果](#提速效果)
    - [性能权衡](#性能权衡)
  - [Function tracing](#function-tracing)
  - [参考](#参考)

Last updated: 2022-10-11, 12:39
@author Jiawei Mao
****

## 速览

Graph 优点：

- 可移植性好
- 性能更好

## 简介

下面介绍 TensorFlow 和 Keras 的工作原理。包括如何使用 `tf.function` 从 eager 执行切换到 graph 执行。

### 什么是 graph

在前面的三个指南中，TF 都是 eager 执行，即 TF 操作由 Python 逐个执行的，执行结果也返回到 Python.

eager 执行有其独特优势，但是 graph 执行的可移植性更好、性能更佳。Graph 执行将张量计算作为 TF 图（TensorFlow graph）执行，下面将 `tf.Graph` 简称为 "graph"。

graph 是一个数据结构，包含一组基本计算单元 `tf.Operation` 和一组基本数据单元 `tf.Tensor`。这些对象在 [tf.Graph](https://tensorflow.google.cn/api_docs/python/tf/Graph) context 中定义。由于 graph 是数据结构，因此即使没有原始 Python 代码，也可以保存、运行和恢复 graph。

下图是在 TensorBoard 可视化的包含两层神经网络的 TF graph：

![](images/2021-12-24-11-00-37.png)

### graph 的优点

graph 很灵活，可以在没有 Python 的环境中使用，如手机端、嵌入式设备以及后端服务器等。TensorFlow 在 Python 中保存的模型格式 [SavedModel](https://tensorflow.google.cn/guide/saved_model) 也是 graph 格式。

graph 容易优化，允许编译器执行如下转换：

- 通过折叠常量节点静态推断张量的值（常量折叠）；
- 拆分计算的独立子部分，进行多线程计算；
- 通过消除常见子表达式简化算术运算。

优化系统 Grappler 用于执行这些优化。

简而言之，graph 可以让 TensorFlow 运行更快、并行以及在多个设备上高效运行。

但是，为了方便起见，大家仍然希望用 Python 来定义机器学习模型，然后在需要时自动构建 graph。

## 配置

```python
import tensorflow as tf
import timeit
from datetime import datetime
```

## graph 的使用

使用 [tf.function](https://tensorflow.google.cn/api_docs/python/tf/function) 创建并运行 graph。可以直接调用 `tf.function`，也可以将其作为 decorator 使用。`tf.function` 以常规函数为输入，返回 `Function`。`Function` 是可调用 Python 对象，负责将 Python 函数转换为 TF graph。例如：

```python
# 定义 Python 函数
def a_regular_function(x, y, b):
    x = tf.matmul(x, y)
    x = x + b
    return x


# `a_function_that_uses_a_graph` 是一个 TF `Function`.
a_function_that_uses_a_graph = tf.function(a_regular_function)

# 创建张量
x1 = tf.constant([[1.0, 2.0]])
y1 = tf.constant([[2.0], [3.0]])
b1 = tf.constant(4.0)

orig_value = a_regular_function(x1, y1, b1).numpy()
# 调用 `Function` 和调用 Python 函数一样
tf_function_value = a_function_that_uses_a_graph(x1, y1, b1).numpy()
assert orig_value == tf_function_value
```

从外观看，`Function` 就像使用 TensorFlow 操作编写的普通函数。然而在底层非常不同，一个 `Function` 的一个 API 后面封装了好几个 `tf.Graph`。这是 `Function` 能够利用 graph 优点的原因。

`tf.function` 应用于当前函数，及函数内部调用的其它函数：

```python
def inner_function(x, y, b):
  x = tf.matmul(x, y)
  x = x + b
  return x

# 使用装饰器标识 `outer_function` 为 `Function`.
@tf.function
def outer_function(x):
  y = tf.constant([[2.0], [3.0]])
  b = tf.constant(4.0)

  return inner_function(x, y, b)

# 该调用会创建一个 graph，该 graph 包括 `inner_function` 和 `outer_function`
outer_function(tf.constant([[1.0, 2.0]])).numpy()
```

```txt
array([[12.]], dtype=float32)
```

### 将 Python 函数转换为 graph

用 TensorFlow 编写的函数都同时包含 TF 内置操作和 Python 逻辑，如 `if` 语句、循环、`break`、`return`、`continue` 等。TF 操作很容易被 `tf.Graph` 捕获，但 Python 逻辑则需要额外步骤才能转换为 graph。`tf.function` 使用 [tf.autograph](https://tensorflow.google.cn/api_docs/python/tf/autograph) 库将 Python 代码转换为生成 graph 的代码。

```python
def simple_relu(x):
  if tf.greater(x, 0):
    return x
  else:
    return 0

# `tf_simple_relu` 是封装 `simple_relu` 的 TensorFlow `Function`
tf_simple_relu = tf.function(simple_relu)

print("First branch, with graph:", tf_simple_relu(tf.constant(1)).numpy())
print("Second branch, with graph:", tf_simple_relu(tf.constant(-1)).numpy())
```

```txt
First branch, with graph: 1
Second branch, with graph: 0
```

虽然一般不需要直接查看 graph，但可以检查输出，以检查确切的结果。下面的输出很复杂，不用仔细看：

```python
# This is the graph-generating output of AutoGraph.
print(tf.autograph.to_code(simple_relu))
```

```txt
def tf__simple_relu(x):
    with ag__.FunctionScope('simple_relu', 'fscope', ag__.ConversionOptions(recursive=True, user_requested=True, optional_features=(), internal_convert_user_code=True)) as fscope:
        do_return = False
        retval_ = ag__.UndefinedReturnValue()

        def get_state():
            return (do_return, retval_)

        def set_state(vars_):
            nonlocal do_return, retval_
            (do_return, retval_) = vars_

        def if_body():
            nonlocal do_return, retval_
            try:
                do_return = True
                retval_ = ag__.ld(x)
            except:
                do_return = False
                raise

        def else_body():
            nonlocal do_return, retval_
            try:
                do_return = True
                retval_ = 0
            except:
                do_return = False
                raise
        ag__.if_stmt(ag__.converted_call(ag__.ld(tf).greater, (ag__.ld(x), 0), None, fscope), if_body, else_body, get_state, set_state, ('do_return', 'retval_'), 2)
        return fscope.ret(retval_, do_return)
```

```python
# This is the graph itself.
print(tf_simple_relu.get_concrete_function(tf.constant(1)).graph.as_graph_def())
```

```txt
node {
  name: "x"
  op: "Placeholder"
  attr {
    key: "_user_specified_name"
    value {
      s: "x"
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "shape"
    value {
      shape {
      }
    }
  }
}
node {
  name: "Greater/y"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 0
      }
    }
  }
}
node {
  name: "Greater"
  op: "Greater"
  input: "x"
  input: "Greater/y"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "cond"
  op: "StatelessIf"
  input: "Greater"
  input: "x"
  attr {
    key: "Tcond"
    value {
      type: DT_BOOL
    }
  }
  attr {
    key: "Tin"
    value {
      list {
        type: DT_INT32
      }
    }
  }
  attr {
    key: "Tout"
    value {
      list {
        type: DT_BOOL
        type: DT_INT32
      }
    }
  }
  attr {
    key: "_lower_using_switch_merge"
    value {
      b: true
    }
  }
  attr {
    key: "_read_only_resource_inputs"
    value {
      list {
      }
    }
  }
  attr {
    key: "else_branch"
    value {
      func {
        name: "cond_false_23"
      }
    }
  }
  attr {
    key: "output_shapes"
    value {
      list {
        shape {
        }
        shape {
        }
      }
    }
  }
  attr {
    key: "then_branch"
    value {
      func {
        name: "cond_true_22"
      }
    }
  }
}
node {
  name: "cond/Identity"
  op: "Identity"
  input: "cond"
  attr {
    key: "T"
    value {
      type: DT_BOOL
    }
  }
}
node {
  name: "cond/Identity_1"
  op: "Identity"
  input: "cond:1"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "Identity"
  op: "Identity"
  input: "cond/Identity_1"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
}
library {
  function {
    signature {
      name: "cond_false_23"
      input_arg {
        name: "cond_placeholder"
        type: DT_INT32
      }
      output_arg {
        name: "cond_identity"
        type: DT_BOOL
      }
      output_arg {
        name: "cond_identity_1"
        type: DT_INT32
      }
    }
    node_def {
      name: "cond/Const"
      op: "Const"
      attr {
        key: "dtype"
        value {
          type: DT_BOOL
        }
      }
      attr {
        key: "value"
        value {
          tensor {
            dtype: DT_BOOL
            tensor_shape {
            }
            bool_val: true
          }
        }
      }
    }
    node_def {
      name: "cond/Const_1"
      op: "Const"
      attr {
        key: "dtype"
        value {
          type: DT_BOOL
        }
      }
      attr {
        key: "value"
        value {
          tensor {
            dtype: DT_BOOL
            tensor_shape {
            }
            bool_val: true
          }
        }
      }
    }
    node_def {
      name: "cond/Const_2"
      op: "Const"
      attr {
        key: "dtype"
        value {
          type: DT_INT32
        }
      }
      attr {
        key: "value"
        value {
          tensor {
            dtype: DT_INT32
            tensor_shape {
            }
            int_val: 0
          }
        }
      }
    }
    node_def {
      name: "cond/Const_3"
      op: "Const"
      attr {
        key: "dtype"
        value {
          type: DT_BOOL
        }
      }
      attr {
        key: "value"
        value {
          tensor {
            dtype: DT_BOOL
            tensor_shape {
            }
            bool_val: true
          }
        }
      }
    }
    node_def {
      name: "cond/Identity"
      op: "Identity"
      input: "cond/Const_3:output:0"
      attr {
        key: "T"
        value {
          type: DT_BOOL
        }
      }
    }
    node_def {
      name: "cond/Const_4"
      op: "Const"
      attr {
        key: "dtype"
        value {
          type: DT_INT32
        }
      }
      attr {
        key: "value"
        value {
          tensor {
            dtype: DT_INT32
            tensor_shape {
            }
            int_val: 0
          }
        }
      }
    }
    node_def {
      name: "cond/Identity_1"
      op: "Identity"
      input: "cond/Const_4:output:0"
      attr {
        key: "T"
        value {
          type: DT_INT32
        }
      }
    }
    ret {
      key: "cond_identity"
      value: "cond/Identity:output:0"
    }
    ret {
      key: "cond_identity_1"
      value: "cond/Identity_1:output:0"
    }
    attr {
      key: "_construction_context"
      value {
        s: "kEagerRuntime"
      }
    }
    arg_attr {
      key: 0
      value {
        attr {
          key: "_output_shapes"
          value {
            list {
              shape {
              }
            }
          }
        }
      }
    }
  }
  function {
    signature {
      name: "cond_true_22"
      input_arg {
        name: "cond_identity_1_x"
        type: DT_INT32
      }
      output_arg {
        name: "cond_identity"
        type: DT_BOOL
      }
      output_arg {
        name: "cond_identity_1"
        type: DT_INT32
      }
    }
    node_def {
      name: "cond/Const"
      op: "Const"
      attr {
        key: "dtype"
        value {
          type: DT_BOOL
        }
      }
      attr {
        key: "value"
        value {
          tensor {
            dtype: DT_BOOL
            tensor_shape {
            }
            bool_val: true
          }
        }
      }
    }
    node_def {
      name: "cond/Identity"
      op: "Identity"
      input: "cond/Const:output:0"
      attr {
        key: "T"
        value {
          type: DT_BOOL
        }
      }
    }
    node_def {
      name: "cond/Identity_1"
      op: "Identity"
      input: "cond_identity_1_x"
      attr {
        key: "T"
        value {
          type: DT_INT32
        }
      }
    }
    ret {
      key: "cond_identity"
      value: "cond/Identity:output:0"
    }
    ret {
      key: "cond_identity_1"
      value: "cond/Identity_1:output:0"
    }
    attr {
      key: "_construction_context"
      value {
        s: "kEagerRuntime"
      }
    }
    arg_attr {
      key: 0
      value {
        attr {
          key: "_output_shapes"
          value {
            list {
              shape {
              }
            }
          }
        }
      }
    }
  }
}
versions {
  producer: 1205
  min_consumer: 12
}
```

大多时候 `tf.function` 都可以直接使用，不需要考虑太多。

### 多态性：一个 `Function`，多个 graphs

`tf.Graph` 用于特定类型输入，如特定 `dtype` 的张量，或具有相同 [id()](https://docs.python.org/3/library/functions.html#id%5D) 的对象。

每次用新的 `dtype` 或 shape 调用 `Function`时，`Function` 都会根据参数创建一个新的 `tf.Graph`。`tf.Graph` 输入 `dtype` 和 shape 称为输入签名（input signature），或简称签名（**signature**）。

`Function` 将和签名对应的 `tf.Graph` 存储在 `ConcreteFunction`。`ConcreteFunction` 是对 `tf.Graph` 的封装。

```python
@tf.function
def my_relu(x):
    return tf.maximum(0., x)


# `my_relu` creates new graphs as it observes more signatures.
print(my_relu(tf.constant(5.5)))
print(my_relu([1, -1]))
print(my_relu(tf.constant([3., -3.])))
```

```txt
tf.Tensor(5.5, shape=(), dtype=float32)
tf.Tensor([1. 0.], shape=(2,), dtype=float32)
tf.Tensor([3. 0.], shape=(2,), dtype=float32)
```

如果已经使用相同签名调用过 `Function`，则 `Function` 不会创建新 `tf.Graph`：

```python
# These two calls do *not* create new graphs.
print(my_relu(tf.constant(-2.5)))  # Signature matches `tf.constant(5.5)`.
print(my_relu(tf.constant([-1., 1.])))  # Signature matches `tf.constant([3., -3.])`.
```

```txt
tf.Tensor(0.0, shape=(), dtype=float32)
tf.Tensor([0. 1.], shape=(2,), dtype=float32)
```

因为由多个 graph 支持，所以 `Function` 是多态的。这使得 `Function` 比单个 `tf.Graph` 支持更多的输入类型，同时也可以优化每个 `tf.Graph` 以获得更好的性能。

```python
# There are three `ConcreteFunction`s (one for each graph) in `my_relu`.
# The `ConcreteFunction` also knows the return type and shape!
print(my_relu.pretty_printed_concrete_signatures())
```

```txt
my_relu(x)
  Args:
    x: float32 Tensor, shape=()
  Returns:
    float32 Tensor, shape=()

my_relu(x=[1, -1])
  Returns:
    float32 Tensor, shape=(2,)

my_relu(x)
  Args:
    x: float32 Tensor, shape=(2,)
  Returns:
    float32 Tensor, shape=(2,)
```

## tf.function 使用

目前已介绍如何将 `tf.function` 作为函数或装饰器使用将 Python 函数转换为 graph。但在实践中，正确使用 `tf.function` 也不容易。下面介绍如何使用 `tf.function`。

### graph 执行和即时执行

`Function` 中代码可以即时执行，也可以作为 graph 执行。`Function` 默认以 graph 执行：

```python
@tf.function
def get_MSE(y_true, y_pred):
    sq_diff = tf.pow(y_true - y_pred, 2)
    return tf.reduce_mean(sq_diff)
```

```python
y_true = tf.random.uniform([5], maxval=10, dtype=tf.int32)
y_pred = tf.random.uniform([5], maxval=10, dtype=tf.int32)
print(y_true)
print(y_pred)
```

```txt
tf.Tensor([2 8 7 7 7], shape=(5,), dtype=int32)
tf.Tensor([6 9 6 7 8], shape=(5,), dtype=int32)
```

```python
get_MSE(y_true, y_pred)
```

```txt
<tf.Tensor: shape=(), dtype=int32, numpy=3>
```

要验证 `Function` 的 graph 是否与等效的 Python 函数执行相同的计算，可以设置 `tf.config.run_functions_eagerly(True)` 来即时执行。该设置关闭 `Function` 创建和运行 graph 的功能，作为常规模型执行代码。

```python
tf.config.run_functions_eagerly(True)
```

```python
get_MSE(y_true, y_pred)
```

```txt
<tf.Tensor: shape=(), dtype=int32, numpy=3>
```

```python
# Don't forget to set it back when you are done.
tf.config.run_functions_eagerly(False)
```

然而，`Function` 的 graph 执行和 eager 执行的行为可能不同。Python 的 `print` 函数在两种模式下就不同。比如在函数中插入一个 `print` 语句，然后重复调用：

```python
@tf.function
def get_MSE(y_true, y_pred):
    print("Calculating MSE!")
    sq_diff = tf.pow(y_true - y_pred, 2)
    return tf.reduce_mean(sq_diff)
```

反复调用，查看输出：

```python
error = get_MSE(y_true, y_pred)
error = get_MSE(y_true, y_pred)
error = get_MSE(y_true, y_pred)
```

```txt
Calculating MSE!
```

可以看到，虽然调用了三次 `get_MSE`，但是只输出了一次。

原因是，`Function` 通过 "tracing" 过程先运行一次原始代码以创建 graph，`print` 语句在此时执行。tracing 将 TF 操作捕获到 graph，但不会捕获 `print`。随后三次调用都运行次 graph，不再运行原始 Python 代码。

关闭 graph 执行来比较一下：

```python
# Now, globally set everything to run eagerly to force eager execution.
tf.config.run_functions_eagerly(True)
```

```python
# Observe what is printed below.
error = get_MSE(y_true, y_pred)
error = get_MSE(y_true, y_pred)
error = get_MSE(y_true, y_pred)
```

```txt
Calculating MSE!
Calculating MSE!
Calculating MSE!
```

```python
tf.config.run_functions_eagerly(False)
```

`print` 是 Python 的一个副作用，在将函数转换为 `Function` 时，还有其它需要注意的项。更多细节可以参考 [tf.function 性能](https://tensorflow.google.cn/guide/function)。

> **Note** 如果想在 eager 执行和 graph 执行中都能打印值，可以使用 `tf.print`。

### non-strict 执行

graph 执行只执行能生成可观察效果(observable effect)的操作，包括：

- 函数返回值；
- 众所周知的一些副作用操作，如：
  - IO 操作，如 `tf.print`
  - 调试操作，如 `tf.debugging` 中的 assert 函数
  - `tf.Variable` 的变种

这种行为通常被称为非严格执行（non-strict），不同于 eager 执行，eager 执行会执行所有操作，不管是否需要。

特别是，runtime error check 不算 observable effect，因此被跳过也不会引发任何运行时错误。

在下例中，graph 执行跳过了“不必要的” `tf.gather` 操作，因此不会引发 `InvalidArgumentError`。所以在 graph 执行中不要依赖抛出错误操作：

```python
def unused_return_eager(x):
    # Get index 1 will fail when `len(x) == 1`
    tf.gather(x, [1])  # unused
    return x


try:
    print(unused_return_eager(tf.constant([0.0])))
except tf.errors.InvalidArgumentError as e:
    # All operations are run during eager execution so an error is raised.
    print(f'{type(e).__name__}: {e}')
```

```txt
tf.Tensor([0.], shape=(1,), dtype=float32)
```

```python
@tf.function
def unused_return_graph(x):
    tf.gather(x, [1])  # unused
    return x


# Only needed operations are run during graph execution. The error is not raised.
print(unused_return_graph(tf.constant([0.0])))
```

```txt
tf.Tensor([0.], shape=(1,), dtype=float32)
```

### tf.function 最佳实践

习惯 `Function` 的行为可能需要一点时间。为了快速上手，可以尝试 `@tf.function` 装饰器，在 eager 和 graph 执行之间来回切换，查看效果。

设计 `tf.function` 可能是编写 graph 兼容的 TF 程序的最佳选择。建议：

- 在早期经常使用 `tf.config.run_functions_eagerly` 在 eager 和 graph 执行之间进行切换，查看是否有差别；
- 在 Python 函数外创建 `tf.Variable`，在函数内部修改。对其它使用 `tf.Variable` 的对象也是如此，如 `keras.layers`, `keras.Model` 以及 `tf.optimizers`；
- 编写的函数要避免依赖于外部 Python 变量，`tf.Variable` 和 keras 对象除外；
- 编写函数尽量以张量或其它 TF 类型作为输入，对其它类型要小心使用；
- 在 `tf.function` 中包含尽可能多的计算，以最大化性能增益。例如，使用 `tf.function` 装饰整个训练步骤或整个训练循环。

## 提速效果

`tf.function` 通常能提高代码性能，但是提高的程度取决于运行的计算类型。可以通过如下方式测试性能差异：

```python
x = tf.random.uniform(shape=[10, 10], minval=-1, maxval=2, dtype=tf.dtypes.int32)

def power(x, y):
    result = tf.eye(10, dtype=tf.dtypes.int32)
    for _ in range(y):
        result = tf.matmul(x, result)
    return result
```

```python
print("Eager execution:", timeit.timeit(lambda: power(x, 100), number=1000))
```

```txt
Eager execution: 2.5637862179974036
```

```python
power_as_graph = tf.function(power)
print("Graph execution:", timeit.timeit(lambda: power_as_graph(x, 100), number=1000))
```

```txt
Graph execution: 0.6316222000004927
```

`tf.function` 一般用于加快训练循环，在 [使用 keras 从头编写训练循环](https://tensorflow.google.cn/guide/keras/writing_a_training_loop_from_scratch) 中有更详细的说明。

> **Note:** 还可以尝试 `tf.function(jit_compile=True)` 来获得更显著的性能提升，特别是在代码中大量使用 TF 控制流和许多小型张量时。

### 性能权衡

graph 可以加快代码，但是创建 graph 也有开销。对有些函数，创建 graph 的开销可能比执行 graph 的还大。不过这种投资通常会随着后续执行的性能提升而弥补过来，需要注意的是，大型模型训练的前几个步骤可能因为 tracing 而变慢。

无论模型多大，都要避免频繁 tracing。[tf.function](https://tensorflow.google.cn/guide/function) 指南讨论了如何通过设置输入规范和使用张量参数来避免 retracing。如果发现性能异常差，最好检查一下是否出现了 retracing。

## Function tracing

要确定 `Function` 何时进行 tracing，可以在代码中添加 `print` 语句。`Function` 只在 tracing 时才会执行 `print` 语句。

```python
@tf.function
def a_function_with_python_side_effect(x):
    print("Tracing!")  # An eager-only side effect.
    return x * x + tf.constant(2)


# This is traced the first time.
print(a_function_with_python_side_effect(tf.constant(2)))
# The second time through, you won't see the side effect.
print(a_function_with_python_side_effect(tf.constant(3)))
```

```txt
Tracing!
tf.Tensor(6, shape=(), dtype=int32)
tf.Tensor(11, shape=(), dtype=int32)
```

```python
# This retraces each time the Python argument changes,
# as a Python argument could be an epoch count or other
# hyperparameter.
print(a_function_with_python_side_effect(2))
print(a_function_with_python_side_effect(3))
```

```txt
Tracing!
tf.Tensor(6, shape=(), dtype=int32)
Tracing!
tf.Tensor(11, shape=(), dtype=int32)
```

新的 Python 参数总是触发创建新的 graph，从而需要额外的 tracing。

## 参考

- https://www.tensorflow.org/guide/intro_to_graphs
