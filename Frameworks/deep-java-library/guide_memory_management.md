# 内存管理

2025-11-28⭐
@author Jiawei Mao
***
## 简介

内存是深度学习领域最大挑战之一，对 Java 尤其如此。Java 最大的问题是 Java GC 无法回收本地内存。它不知道用了多少内存，也不知道怎么释放。此外，Java GC 对高内存使用率（如 GPU 上训练）的应用太慢了。

没有自动内存管理，仍然可以手动释放内存。但是当深度学习创建很多 `NDArray` 时，手动释放不实用。

因此，DJL 使用 [NDManager](https://javadoc.io/doc/ai.djl/api/latest/ai/djl/ndarray/NDManager.html)。NDManager 即是 factory，也是 scope。由 manager 创建的所有数组都属于其 scope。通过释放 manager，可以释放与其绑定的所有数组。包括通过 `NDArray.add(other)` 和 `Activation.relu(array)` 等操作创建的数组。它们附加到输入参数所属 manager。如果操作包含多个 `NDArray` 参数，则与第一个参数所属 manager 绑定。

NDManager 为分层结构，代表不同的抽象层次和数据释放节点。下面看一些常见用法。

### 推理用例

典型推理用例 NDManager 的结构：

<img src="./images/ndmanager_structure_for_inference.png" width="120" />

结构的顶层是 system-manager，它包含全局内存，无法关闭。调用 `NDManager.newBaseManager()` 创建的 manager 为 system-manager 的子节点。

在 system-manager 下y有一个用于为 model 和 predictor 的 manager。model 和 predictor 都可能持久存在，包含预测所需参数。model 包含参数的标准副本，predictor 可能包含额外的参数副本，具体取决于使用的引擎以及是否使用多个设备。

传递给 `predict()` 的数据被添加到 `PredictorContext` manager。这些数据是临时的，仅在 predict 期间有效，确保在 `predict()` 调用期间创建的所有临时数组能被迅速释放。只要 `Translator` 的输入和输出都是标准 Java 类，而不是 `NDArray` 或 `NDList`，它都能自动处理内存。如果输入或输出使用 `NDResource`，用户必须确保与特定 manager 绑定，并在不需要时释放。

确保在预处理、后处理和模型中，仅在 `PredictorContext` 中创建数据。在其它 manager 中创建的内存（特别是模型和 predictor）无法释放，会导致内存泄漏。

### 训练用例

典型训练用例中 NDManager 的结构：

<img src="./images/ndmanager_structure_for_training.png" width="120" />

有推理有i样，它包含相同的系统 manager 和模型。`Trainer` 与 `Predictor` 类似，但是只创建一个 `Trainer`，而不是每个设备创建一个。

此外，还有一个 `Batch`。它包含一个训练步骤的内存，训练后立刻释放。这里使用 `NDManager` 使每个 batch 可以占据整个 GPU 而不使 GC 受损。**注意**：`Batch` 必须在训练步骤结束时手动关闭。



确保在训练步骤、loss、model 和 data 是在 `Batch` manager 中创建。在其它 manager (特别是 model 或 trainer)创建的内存不会释放，从而导致内存泄露。如果你关闭了 `Batch`，但在训练过程中内存持续增长，那么很可能是部分内存绑定到了错误的 manager 上。

## 经验法则



- 输出 `NDarray` 的操作应与输入 `NDArray` 绑定到同一个 manager。顺序很重要，因为默认使第一个 `NDArray` manager 作为操作的manager
- 如果中间 `NDArray` 进入了更上层的 `NDManager` (例如从 `Trainer` 到 `Model`)，这就是内存泄漏
- 可以使用 `NDManager.debugDump()` 查看 `NDManagr` 的资源数量是否持续增长
- 如果需要大量中间 `NDArray`，建议自定义 `NDManager` 或手动关闭

## 高级内存管理

还有一些工具用于更复杂的内存管理，第一种是使用 `NDManager.tempAttachAll()`。这使得 manager 可以从其它 manager 借用一些资源（`NDManager` 或 `NDList`）。当借用完成，资源回到原 manager。

临时 attach 最常见的用途是常见执行某些计算的 scope。为计算创建创建专用 manager，输入会临时附加到该 manager。计算完成后，计算结果通过手动或 `computationManager.ret(resource)` 连接到更高层的 manager。然后关闭计算 manager，释放所有中间计算内存，并将输入数组返回原 manager。

## 排除故障

如果你发现内存持续上升，以下是一些值得注意的因素：

- 如果你在训练，要确保关闭 `Batch`
- 如果是在推理，确保 translator 要么使用标准 Java 类（不是 `NDManager` 或 `NDList`），要么手动释放推理的输入、输出

如果不是以上问题，尝试调用 `ndManager.cap()`。调用 `cap` 会阻止 manager 附加额外数据。在训练 batch 或预测前，限制 Model 或 Trainer/Predictor 的 manager，如果不小心将内存附加到这些高级 manager，会抛出异常。

也可以使用 `BaseNDManager.debugDump(...)` 查看 manager 绑定的资源。反复调用可以确定 manager 的资源是否在增加。设置数组和 manager 名称有助于识别这些 dump。

##  参考

- https://docs.djl.ai/master/docs/development/memory_management.html
- https://github.com/deepjavalibrary/djl/blob/master/docs/development/memory_management.md