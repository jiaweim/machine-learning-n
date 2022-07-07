# Module: tf

- [Module: tf](#module-tf)
  - [模块](#模块)
  - [类](#类)
  - [函数](#函数)
  - [其它](#其它)
  - [参考](#参考)

2021-12-30, 15:23
***

## 模块

|模块|说明|
|---|---|
|audio|public API for tf.audio namespace|
|autodiff|Public API for tf.autodiff namespace|
|autograph|将 eager 风格 Python 转换为 TensorFlow graph|
|bitwise|对整数二进制形式的操作|
|compat|兼容函数|
|config|Public API for tf.config namespace|
|data|用于输入管道的 `tf.data.Dataset` API|
|debugging|Public API for tf.debugging namespace.|
|distribute|用于在多个设备上运行计算的库|
|dtypes | Public API for tf.dtypes namespace.|
|errors | TensorFlow 的异常类型|
|estimator | Estimator：用于处理模型的高级工具|
|experimental | Public API for tf.experimental namespace.|
|feature_column | Public API for tf.feature_column namespace.|
|graph_util | 在 Python 中操作张量图的辅助函数|
|image | 图像操作|
|io | Public API for tf.io namespace.|
|keras | TensorFlow 高级 API Keras 的实现|
|linalg | 线性代数操作|
|lite | Public API for tf.lite namespace.|
|lookup | Public API for tf.lookup namespace.|
|math | 数学操作|
|mlir | Public API for tf.mlir namespace.|
|nest | 结构相关函数 |
|nn | 神经网络基础操作|
|profiler | Public API for tf.profiler namespace.|
|quantization | Public API for tf.quantization namespace.|
|queue | Public API for tf.queue namespace.|
|ragged |参差张量|
|random | Public API for tf.random namespace.|
|raw_ops | Public API for tf.raw_ops namespace.|
|saved_model | Public API for tf.saved_model namespace.|
|sets | Tensorflow 集合操作|
|signal | 信号处理操作|
|sparse | 稀疏张量表示|
|strings | 字符串张量操作|
|summary | 输出摘要数据操作，用于分析和可视化|
|sysconfig | 系统配置库|
|test | Testing.|
|tpu | TPU 相关操作|
|train | 模型训练|
|types | TensorFlow 类型定义|
|version | Public API for tf.version namespace.|
|xla | Public API for tf.xla namespace.|

## 类

|类|说明|
|---|---|
|AggregationMethod|包含用于合并梯度的方法|
|CriticalSection|Critical section.|
|DType|Represents the type of the elements in a Tensor.|
|DeviceSpec|Represents a (possibly partial) specification for a TensorFlow device.|
|GradientTape|Record operations for automatic differentiation.|
|Graph|A TensorFlow computation, represented as a dataflow graph.|
|IndexedSlices|A sparse representation of a set of tensor slices at given indices.|
|IndexedSlicesSpec|Type specification for a tf.IndexedSlices.|
|Module|Base neural network module class.|
|Operation|Represents a graph node that performs computation on tensors.|
|OptionalSpec|Type specification for tf.experimental.Optional.|
|RaggedTensor|Represents a ragged tensor.|
|RaggedTensorSpec|Type specification for a tf.RaggedTensor.|
|RegisterGradient|A decorator for registering the gradient function for an op type.|
|SparseTensor|Represents a sparse tensor.|
|SparseTensorSpec|Type specification for a tf.sparse.SparseTensor.|
|Tensor|A tf.Tensor represents a multidimensional array of elements.|
|TensorArray|Class wrapping dynamic-sized, per-time-step, write-once Tensor arrays.|
|TensorArraySpec|Type specification for a tf.TensorArray.|
|TensorShape|Represents the shape of a Tensor.|
|TensorSpec|Describes a tf.Tensor.|
|TypeSpec|Specifies a TensorFlow value type.|
|UnconnectedGradients|Controls how gradient computation behaves when y does not depend on x.|
|Variable|See the variable guide.|
|VariableAggregation|Indicates how a distributed variable will be aggregated.|
|VariableSynchronization|Indicates when a distributed variable will be synced.|
|constant_initializer|Initializer that generates tensors with constant values.|
|name_scope|A context manager for use when defining a Python op.|
|ones_initializer|Initializer that generates tensors initialized to 1.|
|random_normal_initializer|Initializer that generates tensors with a normal distribution.|
|random_uniform_initializer|Initializer that generates tensors with a uniform distribution.|
|zeros_initializer|Initializer that generates tensors initialized to 0.|

## 函数

|函数|说明|
|---|---|
|Assert(...)|Asserts that the given condition is true.|
|abs(...)|Computes the absolute value of a tensor.|
|acos(...)|Computes acos of x element-wise.|
|acosh(...)|Computes inverse hyperbolic cosine of x element-wise.|
|add(...)|Returns x + y element-wise.|
|add_n(...)|Adds all input tensors element-wise.|
|argmax(...)|Returns the index with the largest value across axes of a tensor.|
|argmin(...)|Returns the index with the smallest value across axes of a tensor.|
|argsort(...)|Returns the indices of a tensor that give its sorted order along an axis.|
|as_dtype(...)|Converts the given type_value to a DType.|
|as_string(...)|Converts each entry in the given tensor to strings.|
|asin(...)|Computes the trignometric inverse sine of x element-wise.|
|asinh(...)|Computes inverse hyperbolic sine of x element-wise.|
|assert_equal(...)|Assert the condition x == y holds element-wise.|
|assert_greater(...)|Assert the condition x > y holds element-wise.|
|assert_less(...)|Assert the condition x < y holds element-wise.|
|assert_rank(...)|Assert that x has rank equal to rank.|
|atan(...)|Computes the trignometric inverse tangent of x element-wise.|
|atan2(...)|Computes arctangent of y/x element-wise, respecting signs of the arguments.|
|atanh(...)|Computes inverse hyperbolic tangent of x element-wise.|
|batch_to_space(...)|BatchToSpace for N-D tensors of type T.|
|bitcast(...)|Bitcasts a tensor from one type to another without copying data.|
|boolean_mask(...)|Apply boolean mask to tensor.|
|broadcast_dynamic_shape(...)|Computes the shape of a broadcast given symbolic shapes.|
|broadcast_static_shape(...)|Computes the shape of a broadcast given known shapes.|
|broadcast_to(...)|Broadcast an array for a compatible shape.|
|case(...)|Create a case operation.|
|cast(...)|Casts a tensor to a new type.|
|clip_by_global_norm(...)|Clips values of multiple tensors by the ratio of the sum of their norms.|
|clip_by_norm(...)|Clips tensor values to a maximum L2-norm.|
|clip_by_value(...)|Clips tensor values to a specified min and max.|
|complex(...)|Converts two real numbers to a complex number.|
|concat(...)|Concatenates tensors along one dimension.|
|cond(...)|Return true_fn() if the predicate pred is true else false_fn().|
|constant(...)|Creates a constant tensor from a tensor-like object.|
|control_dependencies(...)|Wrapper for Graph.control_dependencies() using the default graph.|
|convert_to_tensor(...)|Converts the given value to a Tensor.|
|cos(...)|Computes cos of x element-wise.|
|cosh(...)|Computes hyperbolic cosine of x element-wise.|
|cumsum(...)|Compute the cumulative sum of the tensor x along axis.|
|custom_gradient(...)|Decorator to define a function with a custom gradient.|
|device(...)|Specifies the device for ops created/executed in this context.|
|divide(...)|Computes Python style division of x by y.|
|dynamic_partition(...)|Partitions data into num_partitions tensors using indices from partitions.|
|dynamic_stitch(...)|Interleave the values from the data tensors into a single tensor.|
|edit_distance(...)|Computes the Levenshtein distance between sequences.|
|eig(...)|Computes the eigen decomposition of a batch of matrices.|
|eigvals(...)|Computes the eigenvalues of one or more matrices.|
|einsum(...)|Tensor contraction over specified indices and outer product.|
|ensure_shape(...)|Updates the shape of a tensor and checks at runtime that the shape holds.|
|equal(...)|Returns the truth value of (x == y) element-wise.|
|executing_eagerly(...)|Checks whether the current thread has eager execution enabled.|
|exp(...)|Computes exponential of x element-wise. .|
|expand_dims(...)|Returns a tensor with a length 1 axis inserted at index axis.|
|extract_volume_patches(...)|Extract patches from input and put them in the "depth" output dimension. 3D extension of extract_image_patches.|
|eye(...)|Construct an identity matrix, or a batch of matrices.|
|fill(...)|Creates a tensor filled with a scalar value.|
|fingerprint(...)|Generates fingerprint values.|
|floor(...)|Returns element-wise largest integer not greater than x.|
|foldl(...)|foldl on the list of tensors unpacked from elems on dimension 0. (deprecated argument values)|
|foldr(...)|foldr on the list of tensors unpacked from elems on dimension 0. (deprecated argument values)|
|function(...)|Compiles a function into a callable TensorFlow graph. (deprecated arguments)|
|gather(...)|Gather slices from params axis axis according to indices. (deprecated arguments)|
|gather_nd(...)|Gather slices from params into a Tensor with shape specified by indices.|
|get_current_name_scope(...)|Returns current full name scope specified by tf.name_scope(...)s.|
|get_logger(...)|Return TF logger instance.|
|get_static_value(...)|Returns the constant value of the given tensor, if efficiently calculable.|
|grad_pass_through(...)|Creates a grad-pass-through op with the forward behavior provided in f.|
|gradients(...)|Constructs symbolic derivatives of sum of ys w.r.t. x in xs.|
|greater(...)|Returns the truth value of (x > y) element-wise.|
|greater_equal(...)|Returns the truth value of (x >= y) element-wise.|
|group(...)|Create an op that groups multiple operations.|
|guarantee_const(...)|Promise to the TF runtime that the input tensor is a constant. (deprecated)|
|hessians(...)|Constructs the Hessian of sum of ys with respect to x in xs.|
|histogram_fixed_width(...)|Return histogram of values.|
|histogram_fixed_width_bins(...)|Bins the given values for use in a histogram.|
|identity(...)|Return a Tensor with the same shape and contents as input.|
|identity_n(...)|Returns a list of tensors with the same shapes and contents as the input|
|import_graph_def(...)|Imports the graph from graph_def into the current default Graph. (deprecated arguments)|
|init_scope(...)|A context manager that lifts ops out of control-flow scopes and function-building graphs.|
|inside_function(...)|Indicates whether the caller code is executing inside a tf.function.|
|is_tensor(...)|Checks whether x is a TF-native type that can be passed to many TF ops.|
|less(...)|Returns the truth value of (x < y) element-wise.|
|less_equal(...)|Returns the truth value of (x <= y) element-wise.|
|[linspace(...)](linspace.md)|Generates evenly-spaced values in an interval along a given axis.|
|load_library(...)|Loads a TensorFlow plugin.|
|load_op_library(...)|Loads a TensorFlow plugin, containing custom ops and kernels.|
|logical_and(...)|Returns the truth value of x AND y element-wise.|
|logical_not(...)|Returns the truth value of NOT x element-wise.|
|logical_or(...)|Returns the truth value of x OR y element-wise.|
|make_ndarray(...)|Create a numpy ndarray from a tensor.|
|make_tensor_proto(...)|Create a TensorProto.|
|map_fn(...)|Transforms elems by applying fn to each element unstacked on axis 0. (deprecated arguments)|
|matmul(...)|Multiplies matrix a by matrix b, producing a * b.|
|matrix_square_root(...)|Computes the matrix square root of one or more square matrices:|
|maximum(...)|Returns the max of x and y (i.e. x > y ? x |y) element-wise.|
|meshgrid(...)|Broadcasts parameters for evaluation on an N-D grid.|
|minimum(...)|Returns the min of x and y (i.e. x < y ? x |y) element-wise.|
|multiply(...)|Returns an element-wise x * y.|
|negative(...)|Computes numerical negative value element-wise.|
|no_gradient(...)|Specifies that ops of type op_type is not differentiable.|
|no_op(...)|Does nothing. Only useful as a placeholder for control edges.|
|nondifferentiable_batch_function(...)|Batches the computation done by the decorated function.|
|norm(...)|Computes the norm of vectors, matrices, and tensors.|
|not_equal(...)|Returns the truth value of (x != y) element-wise.|
|numpy_function(...)|Wraps a python function and uses it as a TensorFlow op.|
|one_hot(...)|Returns a one-hot tensor.|
|[ones(...)](ones.md)|Creates a tensor with all elements set to one (1).|
|ones_like(...)|Creates a tensor of all ones that has the same shape as the input.|
|pad(...)|Pads a tensor.|
|parallel_stack(...)|Stacks a list of rank-R tensors into one rank-(R+1) tensor in parallel.|
|pow(...)|Computes the power of one value to another.|
|print(...)|Print the specified inputs.|
|py_function(...)|Wraps a python function into a TensorFlow op that executes it eagerly.|
|[range(...)](range.md)|Creates a sequence of numbers.|
|rank(...)|Returns the rank of a tensor.|
|realdiv(...)|Returns x / y element-wise for real types.|
|recompute_grad(...)|An eager-compatible version of recompute_grad.|
|reduce_all(...)|Computes tf.math.logical_and of elements across dimensions of a tensor.|
|reduce_any(...)|Computes tf.math.logical_or of elements across dimensions of a tensor.|
|reduce_logsumexp(...)|Computes log(sum(exp(elements across dimensions of a tensor))).|
|reduce_max(...)|Computes tf.math.maximum of elements across dimensions of a tensor.|
|reduce_mean(...)|Computes the mean of elements across dimensions of a tensor.|
|reduce_min(...)|Computes the tf.math.minimum of elements across dimensions of a tensor.|
|reduce_prod(...)|Computes tf.math.multiply of elements across dimensions of a tensor.|
|reduce_sum(...)|Computes the sum of elements across dimensions of a tensor.|
|register_tensor_conversion_function(...)|Registers a function for converting objects of base_type to Tensor.|
|repeat(...)|Repeat elements of input.|
|required_space_to_batch_paddings(...)|Calculate padding required to make block_shape divide input_shape.|
|reshape(...)|Reshapes a tensor.|
|reverse(...)|Reverses specific dimensions of a tensor.|
|reverse_sequence(...)|Reverses variable length slices.|
|roll(...)|Rolls the elements of a tensor along an axis.|
|round(...)|Rounds the values of a tensor to the nearest integer, element-wise.|
|saturate_cast(...)|Performs a safe saturating cast of value to dtype.|
|scalar_mul(...)|Multiplies a scalar times a Tensor or IndexedSlices object.|
|scan(...)|scan on the list of tensors unpacked from elems on dimension 0. (deprecated argument values)|
|scatter_nd(...)|Scatters updates into a tensor of shape shape according to indices.|
|searchsorted(...)|Searches for where a value would go in a sorted sequence.|
|sequence_mask(...)|Returns a mask tensor representing the first N positions of each cell.|
|shape(...)|Returns a tensor containing the shape of the input tensor.|
|shape_n(...)|Returns shape of tensors.|
|sigmoid(...)|Computes sigmoid of x element-wise.|
|sign(...)|Returns an element-wise indication of the sign of a number.|
|sin(...)|Computes sine of x element-wise.|
|sinh(...)|Computes hyperbolic sine of x element-wise.|
|size(...)|Returns the size of a tensor.|
|slice(...)|Extracts a slice from a tensor.|
|[sort(...)](sort.md)|Sorts a tensor.|
|space_to_batch(...)|SpaceToBatch for N-D tensors of type T.|
|space_to_batch_nd(...)|SpaceToBatch for N-D tensors of type T.|
|split(...)|Splits a tensor value into a list of sub tensors.|
|sqrt(...)|Computes element-wise square root of the input tensor.|
|square(...)|Computes square of x element-wise.|
|squeeze(...)|Removes dimensions of size 1 from the shape of a tensor.|
|stack(...)|Stacks a list of rank-R tensors into one rank-(R+1) tensor.|
|stop_gradient(...)|Stops gradient computation.|
|strided_slice(...)|Extracts a strided slice of a tensor (generalized Python array indexing).|
|subtract(...)|Returns x - y element-wise.|
|switch_case(...)|Create a switch/case operation, i.e. an integer-indexed conditional.|
|tan(...)|Computes tan of x element-wise.|
|tanh(...)|Computes hyperbolic tangent of x element-wise.|
|tensor_scatter_nd_add(...)|Adds sparse updates to an existing tensor according to indices.|
|tensor_scatter_nd_max(...)|
|tensor_scatter_nd_min(...)|
|tensor_scatter_nd_sub(...)|Subtracts sparse updates from an existing tensor according to indices.|
|tensor_scatter_nd_update(...)|"Scatter updates into an existing tensor according to indices.|
|tensordot(...)|Tensor contraction of a and b along specified axes and outer product.|
|tile(...)|Constructs a tensor by tiling a given tensor.|
|timestamp(...)|Provides the time since epoch in seconds.|
|transpose(...)|Transposes a, where a is a Tensor.|
|truediv(...)|Divides x / y elementwise (using Python 3 division operator semantics).|
|truncatediv(...)|Returns x / y element-wise for integer types.|
|truncatemod(...)|Returns element-wise remainder of division. This emulates C semantics in that|
|tuple(...)|Groups tensors together.|
|type_spec_from_value(...)|Returns a tf.TypeSpec that represents the given value.|
|unique(...)|Finds unique elements in a 1-D tensor.|
|unique_with_counts(...)|Finds unique elements in a 1-D tensor.|
|unravel_index(...)|Converts an array of flat indices into a tuple of coordinate arrays.|
|unstack(...)|Unpacks the given dimension of a rank-R tensor into rank-(R-1) tensors.|
|variable_creator_scope(...)|Scope which defines a variable creation function to be used by variable().|
|vectorized_map(...)|Parallel map on the list of tensors unpacked from elems on dimension 0.|
|where(...)|Return the elements where condition is True (multiplexing x and y).|
|while_loop(...)|Repeat body while the condition cond is true. (deprecated argument values)|
|zeros(...)|Creates a tensor with all elements set to zero.|
|zeros_like(...)|Creates a tensor with all elements set to zero.|

## 其它

|成员|说明|其它|
|---|---|---|
|version|'2.8.0'||
|bfloat16|Instance of tf.dtypes.DType|16-bit bfloat (brain floating point)|
|bool|Instance of tf.dtypes.DType|Boolean|
|complex128|Instance of tf.dtypes.DType|128-bit complex.|
|complex64|Instance of tf.dtypes.DType|64-bit complex.|
|double|Instance of tf.dtypes.DType|64-bit (double precision) floating-point.|
|float16|Instance of tf.dtypes.DType|16-bit (half precision) floating-point.|
|float32|Instance of tf.dtypes.DType|32-bit (single precision) floating-point.|
|float64|Instance of tf.dtypes.DType|64-bit (double precision) floating-point.|
|half|Instance of tf.dtypes.DType|16-bit (half precision) floating-point.|
|int16|Instance of tf.dtypes.DType|Signed 16-bit integer.|
|int32|Instance of tf.dtypes.DType|Signed 32-bit integer.|
|int64|Instance of tf.dtypes.DType|Signed 64-bit integer.|
|int8|Instance of tf.dtypes.DType|Signed 8-bit integer.|
|newaxis|None|
|qint16|Instance of tf.dtypes.DType|Signed quantized 16-bit integer.|
|qint32|Instance of tf.dtypes.DType|signed quantized 32-bit integer.|
|qint8|Instance of tf.dtypes.DType|Signed quantized 8-bit integer.|
|quint16|Instance of tf.dtypes.DType|Unsigned quantized 16-bit integer.|
|quint8|Instance of tf.dtypes.DType|Unsigned quantized 8-bit integer.|
|resource|Instance of tf.dtypes.DType|Handle to a mutable, dynamically allocated resource.|
|string|Instance of tf.dtypes.DType|Variable-length string, represented as byte array.|
|uint16|Instance of tf.dtypes.DType|Unsigned 32-bit (dword) integer.|
|uint32|Instance of tf.dtypes.DType|
|uint64|Instance of tf.dtypes.DType|Unsigned 64-bit (qword) integer.|
|uint8|Instance of tf.dtypes.DType|Unsigned 8-bit (byte) integer.|
|variant|Instance of tf.dtypes.DType|Data of arbitrary type (known at runtime).|

## 参考

- https://www.tensorflow.org/api_docs/python/tf
