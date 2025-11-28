# 模型

## 加载模型

模型是通过训练创建的 artifact 集合。在深度学习中，使用模型推理通常涉及预处理和后处理。djl 提供 `ZooModel` 类，可以轻松地将数据处理与模型结合起来。

下面介绍如何在各种场景中加载预训练模型。

### 使用 ModelZoo 加载 Model

djl **推荐**使用 `ModelZoo` API 加载模型。

`ModelZoo` 提供一种统一的模型加载方式。其 API 声明式的性质允许将模型信息存储在配置文件中，为测试和部署模型提供了很大的灵活性。

#### Criteria 类

可以使用 `Criteria` 类指定搜索条件来查找要加载的模型。`Criteria` 类遵循 djl 的 Builder 惯例：以 `set` 开头的方法为必填字段，以 `opt` 开头的为可选字段，创建 `Criteria` 必需调用  `setType()` 方法：

```java
Criteria<Image, Classifications> criteria =
        Criteria.builder()
                .setTypes(Image.class, Classifications.class)
                .build();
```

该 criteria 接受以下可选信息：

- Engine: 指定希望模型加载到哪个引擎
- Device: 指定希望模型加载到哪个设备（GPU/CPU）
- Application: 定义模型应用
- Input/Output 数据类型：定义所需的输入和输出数据类型
- artifact id: 定义要加载的模型的 artifact id，可以使用包含 group-id 的完全限定名
- group id: 定义模型所属预加载 `ModelZoo` 的 group-id
- ModelZoo: 指定用于搜索模型的 ModelZoo
- model urls: 逗号分隔字符串，定义模型存储位置
- Translator: 定义用于 ZooModel 的自定义数据处理功能
- Progress: 指定模型加载进度
- filters: 定义与模型属性匹配的搜索过滤条件
- options: 定义 engine/model 的加载选项
- arguments: 定义模型的参数以定制 Translator 的行为
