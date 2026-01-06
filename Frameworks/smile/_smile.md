# smile

- [Smile 快速入门](quickstart.md)
- [Smile 数据处理](./data.md)
- [模型验证](./validation.md)

## 安装

```xml
<dependency>
  <groupId>com.github.haifengl</groupId>
  <artifactId>smile-core</artifactId>
  <version>5.0.2</version>
</dependency>
```

对深度学习和 NLP，使用 `smile-deep` 和 `smile-nlp`。

有些算法依赖于 BLAS 和 LAPACK。在 SMILE v5.x 中使用这些算法，需要安装 OpenBLAS 和 ARPACK 以优化矩阵计算。对 Windows，可以在发布 packages 的 `bin` 目录找到预构建的 DLL 文件，将文件添加到 PATH 环境变量。



## 参考

- https://haifengl.github.io/
- https://github.com/haifengl/smile