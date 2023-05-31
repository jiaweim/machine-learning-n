# 从 TensorFlow 1.x 迁移到 TensorFlow 2

了解如何将 TensorFlow （TF）代码从 TF1.x 迁移到 TF2。转换工作需要一定工作量，但是 TF2 相对 TF1.x 增加了新的功能和模型，更清晰、简单，并且更容易调试。在迁移之前，请阅读 [TensorFlow 1.x vs TensorFlow 2 - Behaviors and APIs](https://www.tensorflow.org/guide/migrate/tf1_vs_tf2)。简而言之，迁移过程为：

1. 运行[自动化脚本](https://www.tensorflow.org/guide/migrate/upgrade)将 TF1.x API 使用转换为 `tf.compat.v1`。
2. 删除旧的 `tf.contrib.layers`，并用 [TF Slim](https://github.com/google-research/tf-slim) 符号替换。检查 [TF Addons](https://www.tensorflow.org/addons) 的其它 `tf.contrib` 符号。
3. 重写 [TF1.x 模型的前向传播](https://www.tensorflow.org/guide/migrate/model_mapping)，以在 TF2 中启动 eager 执行。
4. [验证迁移代码的准确性和数值正确性](https://www.tensorflow.org/guide/migrate/validate_correctness)。
5. 将模型的[训练、评估](https://www.tensorflow.org/guide/migrate/migrating_estimator)和[保存](https://www.tensorflow.org/guide/migrate/saved_model)升级到 TF2.
6. （可选）将包括 [TF Slim](https://github.com/google-research/tf-slim) 的 [TF2 兼容的 `tf.compat.v1` API](https://www.tensorflow.org/guide/migrate/model_mapping#incremental_migration_to_native_tf2) 迁移到惯用的 TF2 APIs。
