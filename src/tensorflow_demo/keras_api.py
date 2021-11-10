import tensorflow as tf
from tensorflow import keras


def build_model():
    text_input_a = tf.keras.Input(shape=(None,), dtype='int32')
