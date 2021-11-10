# tf.keras.metrics

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

## Accuracy

```py
tf.keras.metrics.Accuracy(
    name='accuracy', dtype=None
)
```

计算预测值和标签值相等的概率。

会计算两个本地变量，`total` 和 `count`，
