import tensorflow as tf

x = tf.constant([-float("inf"), -9, -0.5, 1, 1.2, 200, 10, float("inf")])
print(tf.math.sin(x))
