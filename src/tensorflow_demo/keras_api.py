from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf

text_input_a = keras.Input(shape=(None,), dtype='int32')
text_input_b = keras.Input(shape=(None,), dtype='int32')

# 映射 1000 个 unique words 到 128  向量
shared_embedding = layers.Embedding(1000, 128)

# 使用上面的层同时编码两个输入
encoded_input_a = shared_embedding(text_input_a)
encoded_input_b = shared_embedding(text_input_b)

# 两个输出
prediction_a = layers.Dense(1, activation='sigmoid', name='prediction_a')(encoded_input_a)
prediction_b = layers.Dense(1, activation='sigmoid', name='prediction_b')(encoded_input_b)

# 这个模型包含两个输入，两个输出，中间使用共享层
model = tf.keras.Model(inputs=[text_input_a, text_input_b],
                       outputs=[prediction_a, prediction_b])

print(model.summary())
keras.utils.plot_model(model, to_file="shared_model.png")
