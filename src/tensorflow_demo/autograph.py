import tensorflow as tf


@tf.function
def my_func(x):
    print('Tracing.\n')
    return tf.reduce_sum(x)


x = tf.constant([1, 2, 3])
my_func(x)
