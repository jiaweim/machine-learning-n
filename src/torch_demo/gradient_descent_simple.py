import numpy as np
import matplotlib.pyplot as plt

x_data = [1., 2., 3.]
y_data = [2., 4., 6.]

w = 1.0


def forward(x):
    return x * w


def cost(xs, ys):
    cost = 0
    for x, y in zip(xs, ys):
        y_pred = forward(x)
        cost += (y_pred - y) ** 2
    return cost / len(xs)


def gradient(xs, ys):
    grad = 0
    for x, y in zip(xs, ys):
        grad += 2 * x * (x * w - y)
    return grad / len(xs)


cost_list = []
print('Predict (before training)', 4, forward(4))
for epoch in range(100):
    cost_val = cost(x_data, y_data)
    grad_val = gradient(x_data, y_data)
    w -= 0.01 * grad_val
    cost_list.append(cost_val)
    print('Epoch:', epoch, 'w=', w, 'loss=', cost_val)

print("Predict (after trainnig)", 4, forward(4))

epoch_list = range(1, 101)
plt.plot(epoch_list, cost_list)
plt.ylabel('cost')
plt.xlabel('epoch')
plt.show()
