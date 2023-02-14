# Module: tf.dtypes

Last updated: 2022-10-28, 11:10
****

## Classes

|类|说明|
|---|---|
|`DType`|表示 `Tensor` 中元素的类型|

## Functions

|函数|说明|
|---|---|
|`as_dtype(...)`|将 `type_value` 转换为 `DType`|
|`cast(...)`|将张量转换为新的类型|
|`complex(...)`|将两个实数转换为复数|
|`saturate_cast(...)`|执行 `value` 到 `dtype` 的安全饱和强制转换|

## 其它成员

|成员|说明|
|----|---|
|QUANTIZED_DTYPES|```{tf.qint16,tf.qint16_ref,tf.qint32,tf.qint32_ref,tf.qint8, tf.qint8_ref,tf.quint16,tf.quint16_ref,tf.quint8,tf.quint8_ref}```|
|bfloat16|`tf.dtypes.DType` 实例，16-bit bfloat (brain floating point)|
|bool|`tf.dtypes.DType` 实例，Boolean|
|complex128|`tf.dtypes.DType` 实例，128-bit complex|
|complex64|`tf.dtypes.DType` 实例，64-bit complex|
|double|`tf.dtypes.DType` 实例，64-bit (double precision) floating-point|
|float16|`tf.dtypes.DType` 实例16-bit (half precision) floating-point.|
|float32|`tf.dtypes.DType` 实例32-bit (single precision) floating-point.|
|float64|`tf.dtypes.DType` 实例64-bit (double precision) floating-point.|
|half|`tf.dtypes.DType` 实例，16-bit (half precision) floating-point.|
|int16|`tf.dtypes.DType` 实例，Signed 16-bit integer.|
|int32|`tf.dtypes.DType` 实例，Signed 32-bit integer.|
|int64|`tf.dtypes.DType` 实例，Signed 64-bit integer.|
|int8|`tf.dtypes.DType` 实例，Signed 8-bit integer.|
|qint16|`tf.dtypes.DType` 实例，Signed quantized 16-bit integer.|
|qint32|`tf.dtypes.DType` 实例，signed quantized 32-bit integer.|
|qint8|`tf.dtypes.DType` 实例，Signed quantized 8-bit integer.|
|quint16|`tf.dtypes.DType` 实例，Unsigned quantized 16-bit integer.|
|quint8|`tf.dtypes.DType` 实例，Unsigned quantized 8-bit integer.|
|resource|`tf.dtypes.DType` 实例，可变的、动态分配的资源的句柄|
|string|`tf.dtypes.DType` 实例，变长 string, 以 byte array 表示|
|uint16|`tf.dtypes.DType` 实例，Unsigned 16-bit (word) integer.|
|uint32|`tf.dtypes.DType` 实例，Unsigned 32-bit (dword) integer.|
|uint64|`tf.dtypes.DType` 实例，Unsigned 64-bit (qword) integer.|
|uint8|`tf.dtypes.DType` 实例，Unsigned 8-bit (byte) integer|
|variant|`tf.dtypes.DType` 实例，Data of arbitrary type (known at runtime)|

## 参考

- https://tensorflow.google.cn/api_docs/python/tf/dtypes
