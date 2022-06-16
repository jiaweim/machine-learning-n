# TensorFlow 2 快速入门（初学者）

- [TensorFlow 2 快速入门（初学者）](#tensorflow-2-快速入门初学者)
  - [1. 简介](#1-简介)
  - [2. 设置 TensorFlow](#2-设置-tensorflow)
  - [3. 加载数据集](#3-加载数据集)
  - [4. 构建机器学习模型](#4-构建机器学习模型)
  - [5. 训练和评估模型](#5-训练和评估模型)
  - [6. 参考](#6-参考)

Last updated: 2022-06-16, 14:53
@author Jiawei Mao
****

## 1. 简介

这篇入门教程介绍使用 Keras 执行如下操作：

1. 加载预构建数据集；
2. 构建一个图像分类神经网络机器学习模型；
3. 训练神经网络；
4. 评估模型的准确性。

## 2. 设置 TensorFlow

首先导入 TensorFlow：

```python
import tensorflow as tf

print("TensorFlow version: ", tf.__version__)
```

```sh
TensorFlow version:  2.9.1
```

## 3. 加载数据集

加载 MNIST 数据集，并将数据从整型转换为浮点型：

```python
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
```

## 4. 构建机器学习模型

叠加 layers 构建 `tf.keras.Sequential` 模型：

```python
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10)
])
```

对每个样本，该模型都返回一个 logit 向量，每个类别对应向量中的一个值。TensorFlow 模型是可调用对象，可以当作函数调用，接受输入参数，返回预测值。例如：

```python
predictions = model(x_train[:1]).numpy()
predictions
```

```sh
array([[-0.64067507, -0.21394847, -0.21010129,  0.45867267,  0.24329919,
         0.6170706 , -0.4533612 ,  0.5403824 ,  0.72258055,  0.83108985]],
      dtype=float32)
```

然后用 `tf.nn.softmax` 将上面的 logit 值转换为概率值，这样结果更直观：

```python
tf.nn.softmax(predictions).numpy()
```

```sh
array([[0.03884973, 0.05952687, 0.05975632, 0.11663494, 0.09403578,
        0.13665326, 0.046853  , 0.12656532, 0.15185966, 0.16926509]],
      dtype=float32)
```

> [!NOTE]
> 可以把 `tf.nn.softmax` 作为神经网络最后一层的激活函数，这样模型直接输出便于理解的概率值，但不推荐使用这种方法，因为 softmax 的输出无法为所有模型提供精确且数值稳定的损失值。

使用 `losses.SparseCategoricalCrossentropy` 定义损失函数，对每个样本，该函数使用 logit 向量计算标量损失：

```python
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
```

计算得到的损失值是真实类别概率的负对数，如果模型预测正确，损失值为 0。

未经训练的模型给出的概率接近随机值（每个类的概率为 1/10），因此初始损失应该接近于 `-tf.math.log(1/10) ~= 2.3`。

```python
loss_fn(y_train[:1], predictions).numpy()
```

```sh
1.9903085
```

在训练之前，使用 Keras 的 `Model.compile` 配置和编译模型：

- 将 `optimizer` 设置为 `adam`；
- 将 `loss` 设置为前面定义的 `loss_fn`；
- 将评估模型的指标 `metrics` 设置为 `accuracy`。

```python
model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])
```

## 5. 训练和评估模型

使用 `Model.fit` 调整模型参数、最小化损失值：

```python
model.fit(x_train, y_train, epochs=5)
```

```txt
Epoch 1/5
1875/1875 [==============================] - 6s 3ms/step - loss: 0.2918 - accuracy: 0.9153
Epoch 2/5
1875/1875 [==============================] - 5s 3ms/step - loss: 0.1402 - accuracy: 0.9582
Epoch 3/5
1875/1875 [==============================] - 5s 3ms/step - loss: 0.1058 - accuracy: 0.9684
Epoch 4/5
1875/1875 [==============================] - 5s 2ms/step - loss: 0.0842 - accuracy: 0.9738
Epoch 5/5
1875/1875 [==============================] - 5s 2ms/step - loss: 0.0721 - accuracy: 0.9772

<keras.callbacks.History at 0x1e7491b3ee0>
```

`Model.evaluate` 方法使用验证集或测试集评估模型性能：

```python
model.evaluate(x_test, y_test, verbose=2)
```

```sh
313/313 - 1s - loss: 0.0683 - accuracy: 0.9785 - 850ms/epoch - 3ms/step
[0.06833968311548233, 0.9785000085830688]
```

可以看出，分类器在这个数据集上的准确率约为 98%。

如果希望模型返回概率值，可以把上面训练好的模型和 softmax 函数组装到一起：

```python
probability_model = tf.keras.Sequential([
    model,
    tf.keras.layers.Softmax()
])
```

```python
probability_model(x_test[:5])
```

```txt
<tf.Tensor: shape=(5, 10), dtype=float32, numpy=
array([[4.50657422e-09, 6.61763311e-10, 1.79867754e-07, 2.13714098e-04,
        1.64469424e-10, 1.98828275e-07, 1.30286610e-13, 9.99783576e-01,
        1.88420444e-08, 2.19102685e-06],
       [3.46894069e-09, 5.84796544e-05, 9.99873400e-01, 6.63527026e-05,
        1.03661146e-14, 1.66455970e-06, 1.36476306e-08, 1.52059185e-14,
        4.14359995e-08, 4.01574667e-16],
       [7.43744522e-07, 9.99045074e-01, 1.97835994e-04, 5.12826136e-06,
        3.94731251e-05, 2.53630651e-06, 2.69190150e-06, 4.35092312e-04,
        2.70660530e-04, 7.97585187e-07],
       [9.99989152e-01, 7.82966192e-11, 2.38692974e-06, 6.17373930e-09,
        5.08395237e-09, 1.46661691e-07, 7.54723317e-07, 5.80910751e-07,
        1.20342021e-08, 7.00255578e-06],
       [2.03908858e-05, 1.20942314e-08, 5.76243110e-05, 1.68110205e-07,
        9.97464657e-01, 1.89277642e-07, 2.51607912e-06, 3.74163355e-05,
        9.43603936e-06, 2.40751449e-03]], dtype=float32)>
```

恭喜，到这里你已经成功使用 keras API 训练好了一个机器学习模型。

## 6. 参考

- https://www.tensorflow.org/tutorials/quickstart/beginner
