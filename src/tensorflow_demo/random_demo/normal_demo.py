import tensorflow as tf

tf.random.set_seed(5)
print(tf.random.normal([4], 0, 1, tf.float32))
print(tf.random.normal([2, 2], 0, 1, tf.float32, seed=1))
