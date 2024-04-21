# KerasTuner

KerasTuner 是一个易于使用、可扩展的超参数优化框架，用于解决超参数搜索的痛点。KerasTuner 内置有 Bayesian 优化、Hyperband和随机搜索算法，并且易于扩展，便于自定义新的搜索算法。

## 安装

KerasTuner 需要 Python 3.6+ 和 TensorFlow 2.0+。

安装最新版：

```powershell
pip install keras-tuner -U
```

## 简单介绍

导入 KerasTuner 和 TensorFlow:

```python
import keras_tuner
from tensorflow import keras
```

定义一个返回 Keras 模型的函数。使用 `hp` 参数定义模型的超参数：

```python
def build_model(hp):
    model = keras.Sequential()
    model.add(keras.layers.Dense(
        hp.Choice("units", [8, 16, 32]), 
        activation="relu"))
    model.add(keras.layers.Dense(1, activation="relu"))
    model.compile(loss="mse")
    return model
```

初始化 tuner (此处用 `RandomSearch`)。使用 `objective` 指定选择最佳模型所需的指标，并用 `max_trials` 指定要尝试的不同模型的数目。

```python
tuner = keras_tuner.RandomSearch(
    build_model, 
    objective="val_loss", 
    max_trials=5
)
```

开始搜索，获得最佳模型：

```python
tuner.search(x_train, y_train, epochs=5, validation_data=(x_val, y_val))
best_model = tuner.get_best_models()[0]
```

## 参考

- https://keras.io/keras_tuner/
