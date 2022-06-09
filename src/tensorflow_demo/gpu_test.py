import tensorflow as tf

if tf.config.list_physical_devices('GPU'):
    print("TensorFlow IS using the GPU")
else:
    print("TensorFlow IS NOT using the GPU")

for device in tf.config.list_physical_devices("GPU"):
    print(device)
