import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential()
# 添加 嵌入 层，期望输入 词汇量为 1000，输出嵌入维度为 64
model.add(layers.Embedding(input_dim=1000, output_dim=64))
# 添加 LSTM 层，包含 128 个内部单元
model.add(layers.LSTM(128))
# 添加包含 10 个 单元的 Dense 层
model.add(layers.Dense(10))
model.summary()
