from tensorflow.keras import models
from tensorflow.keras import layers

model = models.Sequential([
    # 7x7 filters, 因为输入图片不大，所以没有 stride
    layers.Conv2D(64, 7, activation='relu', padding='SAME', input_shape=[28, 28, 1]),
    layers.MaxPooling2D(2),

    # 2 个卷积层+1个池化层，如果图片太大，可以重复几次这个结构
    layers.Conv2D(128, 3, activation='relu', padding='SAME'),
    layers.Conv2D(128, 3, activation='relu', padding='SAME'),
    layers.MaxPooling2D(2),

    # filters 数目增大，因为低级特征比较有限，但是组合起来可以生成很多高级特征
    layers.Conv2D(256, 3, activation='relu', padding='SAME'),
    layers.Conv2D(256, 3, activation='relu', padding='SAME'),
    layers.MaxPooling2D(2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])
