import numpy as np
import matplotlib.pyplot as plt

x_data = [1., 2., 3.]
y_data = [2., 4., 6.]

w = 1.0


def forward(x):
    return x * w


def loss(x, y):
    y_pred = forward(x)
    return (y - y_pred) ** 2


def gradient(x, y):
    return 2 * x * (x * w - y)


lost_list = []
print('Predict (before training)', 4, forward(4))
for epoch in range(100):
    for x, y in zip(x_data, y_data):
        grad = gradient(x, y)
        w -= 0.01 * grad
        print('\tgrad: ', x, y, grad)
        l = loss(x, y)
        lost_list.append(l)

print("Predict (after trainnig)", 4, forward(4))

epoch_list = range(1, 101)
plt.plot(epoch_list, lost_list)
plt.ylabel('cost')
plt.xlabel('epoch')
plt.show()
