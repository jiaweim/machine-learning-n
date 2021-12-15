# 指标

- [指标](#指标)
  - [简介](#简介)
  - [精度指标（accuracy）](#精度指标accuracy)
    - [Accuracy class](#accuracy-class)
  - [AUC](#auc)

2021-11-26, 10:40
***

## 简介



## 精度指标（accuracy）

### Accuracy class

```py
tf.keras.metrics.Accuracy(name="accuracy", dtype=None)
```

计算预测值和标签纸相等的概率。

该指标创建两个局部变量 `total` 和 `count`，用于计算 `y_pred` 和 `y_true` 匹配的频率。这个频率最终以二进制精度返回：即 `count/total`。

`name`：metric 实例名称。

`dtype`，返回的指标值类型。

单独使用：

```py
>>> m = tf.keras.metrics.Accuracy()
>>> m.update_state([[1], [2], [3], [4]], [[0], [2], [3], [4]])
>>> m.result().numpy()
0.75
```

如果 `sample_weight`为 `None`，weight 默认为 1。将`sample_weight` 设置为 0 来屏蔽值。

```py
>>> m.reset_state()
>>> m.update_state([[1], [2], [3], [4]], [[0], [2], [3], [4]],
...                sample_weight=[1, 1, 0, 0])
>>> m.result().numpy()
0.5
```

或者和 `compile()` 使用：

```py
model.compile(optimizer='sgd',
              loss='mse',
              metrics=[tf.keras.metrics.Accuracy()])
```


## AUC

计算 ROC 或 PR 曲线的曲线下面积（Area under the curve, AUC）。

```py
tf.keras.metrics.AUC(
    num_thresholds=200, curve='ROC',
    summation_method='interpolation', name=None, dtype=None,
    thresholds=None, multi_label=False, num_labels=None, label_weights=None,
    from_logits=False
)
```

AUC 是一种用量度量分类模型好坏的标准。

ROC（Receiver operating characteristic, default）
