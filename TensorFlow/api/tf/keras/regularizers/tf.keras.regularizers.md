# Module: tf.keras.regularizers

- [Module: tf.keras.regularizers](#module-tfkerasregularizers)
  - [类](#类)
  - [函数](#函数)
  - [参考](#参考)

Last updated: 2022-07-05, 16:23
@author Jiawei Mao
***

TF 内置正则化器。

## 类

|类|说明|
|---|---|
|L1|A regularizer that applies a L1 regularization penalty.|
|L1L2|A regularizer that applies both L1 and L2 regularization penalties.|
|L2|A regularizer that applies a L2 regularization penalty.|
|OrthogonalRegularizer|A regularizer that encourages input vectors to be orthogonal to each other.|
|Regularizer|Regularizer base class.|
|l1|A regularizer that applies a L1 regularization penalty.|
|l2|A regularizer that applies a L2 regularization penalty.|
|orthogonal_regularizer|A regularizer that encourages input vectors to be orthogonal to each other.|

## 函数

|函数|说明|
|---|---|
|deserialize(...)|
|get(...)|Retrieve a regularizer instance from a config or identifier.|
|l1_l2(...)|Create a regularizer that applies both L1 and L2 penalties.|
|serialize(...)|

## 参考

- https://www.tensorflow.org/api_docs/python/tf/keras/regularizers
