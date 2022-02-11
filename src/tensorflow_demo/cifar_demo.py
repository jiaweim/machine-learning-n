import logging

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

# access CIFAR-10 dataset and display one of the images

tf.get_logger().setLevel(logging.ERROR)
cifar_dataset = keras.datasets.cifar10
(train_images, train_labels), (test_images, test_labels) = cifar_dataset.load_data()

print('Category: ', train_labels[100])

plt.figure(figsize=(1, 1))
plt.imshow(train_images[100])
plt.show()
