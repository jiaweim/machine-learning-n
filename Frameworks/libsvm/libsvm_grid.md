# 简介

grid.py 是一个使用 RBF kernel 进行 C-SVM **分类**参数选择工具。它使用交叉验证 CV 技术估计指定范围内每个参数组合的准确性，以确定问题的最佳参数。

grid.py 直接执行 libsvm 二进制文件进行交叉验证，然后使用 gnuplot 绘制 CV 精度的轮廓图。在使用之前，必须安装 libsvm 和 gnuplot。

使用：

```sh
grid.py [grid_options] [svm_options] dataset
```

