import tensorflow as tf

print(tf.test.is_gpu_available())
print(tf.test.is_gpu_available(cuda_only=True))
print(tf.test.is_gpu_available(True, (3, 0)))

