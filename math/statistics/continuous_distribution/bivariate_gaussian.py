import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from sklearn import mixture

n_samples = 300

np.random.seed(0)

# 生成以 (20,20) 为中心的球面数据
shifted_gaussian = np.random.randn(n_samples, 2) + np.array([20, 20])

# 生成以 (0,0) 为中心的拉伸高斯数据
C = np.array(([[0.0, -0.7], [3.5, 0.7]]))
stretched_gaussian = np.dot(np.random.randn(n_samples, 2), C)

# 合并两个数据集
X_train = np.vstack([shifted_gaussian, stretched_gaussian])

# 拟合 2 组分混合高斯模型
clf = mixture.GaussianMixture(n_components=2, covariance_type="full")
clf.fit(X_train)

# 以等高线的形式显示模型预测的分数
x = np.linspace(-20.0, 30.0)
y = np.linspace(-20.0, 40.0)
X, Y = np.meshgrid(x, y)
XX = np.array([X.ravel(), Y.ravel()]).T
Z = -clf.score_samples(XX)
Z = Z.reshape(X.shape)

CS = plt.contour(X, Y, Z, norm=LogNorm(vmin=1.0, vmax=1000.0), levels=np.logspace(0, 3, 10))
CB = plt.colorbar(CS, shrink=0.8, extend="both")

plt.scatter(X_train[:, 0], X_train[:, 1], 0.8)
plt.title("Negative log-likelihood predicted by a GMM")
plt.axis("tight")
plt.show()
