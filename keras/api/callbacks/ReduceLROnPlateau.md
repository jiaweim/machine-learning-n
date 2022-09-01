# ReduceLROnPlateau

Last updated: 2022-09-01, 17:25
@author Jiawei Mao
****

## 简介

```python
tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.1,
    patience=10,
    verbose=0,
    mode='auto',
    min_delta=0.0001,
    cooldown=0,
    min_lr=0,
    **kwargs
)
```

当 metric 停止改善时，降低学习率（learning rate, LR）。

在学习停滞时，将学习率降低 2-10 倍往往对训练有利。该 callback 监视某个指标，如果在 `patience` 个 epochs 该指标都没有改善，就降低 LR。

## 参数

**monitor**

要监视的量值。

**factor**

学习率降低的因子，`new_lr = lr * factor`。

**patience**

如果在 `patience` 个 epochs 后没有改善，就降低学习率。

**verbose**

int. 0: quiet, 1: update messages.

**mode**

one of {'auto', 'min', 'max'}：

- In 'min' mode, the learning rate will be reduced when the quantity monitored has stopped decreasing; 
- in 'max' mode it will be reduced when the quantity monitored has stopped increasing;
- in 'auto' mode, the direction is automatically inferred from the name of the monitored quantity.

**min_delta**

threshold for measuring the new optimum, to only focus on significant changes.

**cooldown**

number of epochs to wait before resuming normal operation after lr has been reduced.

**min_lr**

学习率最小值。

## 示例

```python
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=5, min_lr=0.001)
model.fit(X_train, Y_train, callbacks=[reduce_lr])
```

## 参考

- https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/ReduceLROnPlateau
- https://keras.io/api/callbacks/reduce_lr_on_plateau/
