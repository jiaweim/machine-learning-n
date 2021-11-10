# tf.keras.losses

- [tf.keras.losses](#tfkeraslosses)
  - [CategoricalCrossentropy](#categoricalcrossentropy)
  - [SparseCategoricalCrossentropy](#sparsecategoricalcrossentropy)

## CategoricalCrossentropy

计算标签和预测之间的交叉熵损失。

```py
tf.keras.losses.CategoricalCrossentropy(
    from_logits=False, label_smoothing=0.0, axis=-1,
    reduction=losses_utils.ReductionV2.AUTO,
    name='categorical_crossentropy'
)
```

在有两个或多个标签类时使用该交叉熵损失函数。其中标签以 [one_hot](tf.math.md)

## SparseCategoricalCrossentropy

计算标签和预测之间的交叉损失。

```py
tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=False, reduction=losses_utils.ReductionV2.AUTO,
    name='sparse_categorical_crossentropy'
)
```

当有两个或多个标签类时，使用该交叉熵损失函数。其中标签要以 integer 形式提供。如果 label 以 `one-hot` 编码表示，则应该使用 `CategoricalCrossentropy`。

对 `y_pred` 
