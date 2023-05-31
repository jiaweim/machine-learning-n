# GMM 初始化方法

2023-05-31
****
演示高斯混合模型中不同的初始化方法。

这里用四个容易识别的 cluster 生成样本数据，展示初始化参数 `init_param` 支持的四种不同方法：`kmeans` (默认)、`random`、`random_from_data` 以及 `k-means++`。

```python
import matplotlib.pyplot as plt  
import numpy as np  
import sklearn.datasets  
from sklearn.mixture import GaussianMixture  
from sklearn.utils.extmath import row_norms  
from sklearn.datasets import make_blobs  
from timeit import default_timer as timer  
  
# 生成数据  
X, y_true = make_blobs(n_samples=4000, centers=4, cluster_std=0.60, random_state=0)  
X = X[:, ::-1]  
  
n_samples = 4000  
n_components = 4  
x_squared_norms = row_norms(X, squared=True)  
  
  
def get_initial_means(X, init_params, r):  
    # 将 GaussianMixture 的 max_iter=0，输出初始化均值
    gmm = GaussianMixture(  
        n_components=4, init_params=init_params, tol=1e-9, max_iter=0, random_state=r  
    ).fit(X)  
    return gmm.means_  
  
  
methods = ["kmeans", "random_from_data", "k-means++", "random"]  
colors = ["navy", "turquoise", "cornflowerblue", "darkorange"]  
times_init = {}  
relative_times = {}  
  
plt.figure(figsize=(4 * len(methods) // 2, 6))  
plt.subplots_adjust(  
    bottom=0.1, top=0.9, hspace=0.15, wspace=0.05, left=0.05, right=0.95  
)  
  
for n, method in enumerate(methods):  
    r = np.random.RandomState(seed=1234)  
    plt.subplot(2, len(methods) // 2, n + 1)  
  
    start = timer()  
    ini = get_initial_means(X, method, r)  
    end = timer()  
    init_time = end - start  
  
    gmm = GaussianMixture(  
        n_components=4, means_init=ini, tol=1e-9, max_iter=2000, random_state=r  
    ).fit(X)  
  
    times_init[method] = init_time  
    for i, color in enumerate(colors):  
        data = X[gmm.predict(X) == i]  
        plt.scatter(data[:, 0], data[:, 1], color=color, marker="x")  
  
    plt.scatter(  
        ini[:, 0], ini[:, 1], s=75, marker="D", c="orange", lw=1.5, edgecolors="black"  
    )  
    relative_times[method] = times_init[method] / times_init[methods[0]]  
  
    plt.xticks(())  
    plt.yticks(())  
    plt.title(method, loc="left", fontsize=12)  
    plt.title(  
        "Iter %i | Init Time %.2fx" % (gmm.n_iter_, relative_times[method]),  
        loc="right",  
        fontsize=10,  
    )  
  
plt.suptitle("GMM iterations and relative time taken to initialize")  
plt.show()
```

![[Pasted image 20230531205917.png]]
图中：橙色菱形表示 `init_param` 生成的 cluster 中心。余下数据用叉叉表示，颜色为 GMM 分类结果。

每个 subplot 右上方 "Iter" 后的数字表示 `GaussianMixture` 收敛所需的迭代次数，"Init Time" 是初始化运行的时间。一般来说，初始化时间越短，收敛所需的迭代次数就越多。

初始化时间是初始化方法所用时间与默认的 `kmeans` 所用时间比值。可以发现，其它三个方面都比 `kmeans` 快。

在本例中，使用 `random_from_data` 或 `random` 初始化，模型需要更多的迭代次数才能收敛。这里 `k-means++` 最合适，初始化时间短，收敛所需的迭代次数少。

## 参考

- https://scikit-learn.org/stable/auto_examples/mixture/plot_gmm_init.html