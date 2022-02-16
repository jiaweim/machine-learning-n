# 函数 API

- [函数 API](#函数-api)
  - [简介](#简介)
  - [训练、评估和推断](#训练评估和推断)
  - [模型的保存和序列化](#模型的保存和序列化)
  - [参考](#参考)

2021-11-10, 16:35
***

## 简介


![](images/2021-11-10-16-56-53.png)

还可以将每层的 input 和 output shape 绘制到图中：

```py
keras.utils.plot_model(model, "my_first_model_with_shape_info.png", show_shapes=True)
```

![](images/2021-11-10-16-57-40.png)

"graph of layers" 是深度学习模型直观的思维图，而函数API是创建模型的一种方式。

## 训练、评估和推断

使用函数 API 训练、评估和推断的工作方式和[序列模型](sequential_api.md)完全相同。

`Model` 类提供了内置的训练循环（`fit()` 方法）以及评估循环（`evaluate()`）。并且可以自定义这些循环，以实现监督学习以外的算法，如 GAN。

下面，我们载入 MNIST 数据集，训练上面创建的模型，并使用测试数据评估模型：

```py
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_train = x_train.reshape(60000, 784).astype("float32") / 255
x_test = x_test.reshape(10000, 784).astype("float32") / 255

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.RMSprop(),
    metrics=["accuracy"],
)

history = model.fit(x_train, y_train, batch_size=64, epochs=2, validation_split=0.2)

test_scores = model.evaluate(x_test, y_test, verbose=2)
print("Test loss:", test_scores[0])
print("Test accuracy:", test_scores[1])
```

```txt
Epoch 1/2
750/750 [==============================] - 3s 3ms/step - loss: 0.3430 - accuracy: 0.9035 - val_loss: 0.1851 - val_accuracy: 0.9463
Epoch 2/2
750/750 [==============================] - 2s 3ms/step - loss: 0.1585 - accuracy: 0.9527 - val_loss: 0.1366 - val_accuracy: 0.9597
313/313 - 0s - loss: 0.1341 - accuracy: 0.9592
Test loss: 0.13414572179317474
Test accuracy: 0.9592000246047974
```

## 模型的保存和序列化




## 参考

- https://tensorflow.google.cn/guide/keras/functional/
