# 核密度估计
2023-12-20, 22:32
****

下面使用核密度估计（KDE），一种强大非参数密度估计技术，来学习数据集的生成模型。有了这个生成模型，就可以绘制新的样本。这些新样本反应了数据的基本模型

<img src="images/image-20231220222952623.png" alt="image-20231220222952623" width="400"/>

```python
import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity

# 加载手写数字数据
digits = load_digits()

# 将 64 维数据降维到 15 维
pca = PCA(n_components=15, whiten=False)
data = pca.fit_transform(digits.data)

# 使用 grid-search 交叉验证优化 bandwidth
params = {"bandwidth": np.logspace(-1, 1, 20)}
grid = GridSearchCV(KernelDensity(), params)
grid.fit(data)

print("best bandwidth: {0}".format(grid.best_estimator_.bandwidth))

# 使用最佳估计器计算核密度估计
kde = grid.best_estimator_

# sample 44 new points from the data
new_data = kde.sample(44, random_state=0)
new_data = pca.inverse_transform(new_data)

# turn data into a 4x11 grid
new_data = new_data.reshape((4, 11, -1))
real_data = digits.data[:44].reshape((4, 11, -1))

# plot real digits and resampled digits
fig, ax = plt.subplots(9, 11, subplot_kw=dict(xticks=[], yticks=[]))
for j in range(11):
    ax[4, j].set_visible(False)
    for i in range(4):
        im = ax[i, j].imshow(
            real_data[i, j].reshape((8, 8)), cmap=plt.cm.binary, interpolation="nearest"
        )
        im.set_clim(0, 16)
        im = ax[i + 5, j].imshow(
            new_data[i, j].reshape((8, 8)), cmap=plt.cm.binary, interpolation="nearest"
        )
        im.set_clim(0, 16)

ax[0, 5].set_title("Selection from the input data")
ax[5, 5].set_title('"New" digits drawn from the kernel density model')

plt.show()
```

```
best bandwidth: 3.79269019073225
```
