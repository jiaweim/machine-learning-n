import tensorflow as tf

tf.compat.v1.disable_eager_execution()

v_1 = tf.compat.v1.constant([1, 2, 3, 4])
v_2 = tf.compat.v1.constant([2, 1, 5, 3])

v_add = tf.compat.v1.add(v_1, v_2)

with tf.compat.v1.Session() as sess:
    print(sess.run(v_add))
