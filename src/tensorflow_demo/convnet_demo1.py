from sklearn.datasets import load_sample_image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Load sample images
china = load_sample_image("china.jpg") / 255
flower = load_sample_image("flower.jpg") / 255
images = np.array([china, flower])  # mini-batch size, height, width, channels
batch_size, height, width, channels = images.shape

# Create 2 filters

filters = np.zeros(shape=(7, 7, channels, 2), dtype=np.float32)  # 7x7 filter
# fh, fw, height and width of the receptive field
# fn' the number of feature maps in the previous layer
filters[:, 3, :, 0] = 1  # vertical line [fh, fw, fn', fn]
filters[3, :, :, 1] = 1  # horizontal line

outputs = tf.nn.conv2d(images, filters, strides=1, padding='SAME')

plt.imshow(outputs[0, :, :, 1], cmap='gray')  # plot 1st image's 2nd feature map
plt.show()
