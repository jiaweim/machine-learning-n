# torch.utils.tensorboard

- [torch.utils.tensorboard](#torchutilstensorboard)
  - [简介](#简介)
  - [SummaryWriter](#summarywriter)
    - [add\_graph](#add_graph)
    - [add\_scalars](#add_scalars)
  - [参考](#参考)

***

## 简介

## SummaryWriter

添加数据方法：

- add_graph()
- add_histogram()
- add_figure()
- add_text()
- add_custom_scalars()
- add_scalars()
- add_images()
- add_video()
- add_embedding()
- add_mesh()
- add_scalar()
- add_image()
- add_audio()
- add_pr_curve()
- add_hparams())

以及写入数据到 disk 的方法：

- flush()
- close()

### add_graph

```python
add_graph(model, 
    input_to_model=None, 
    verbose=False, 
    use_strict_trace=True)
```

将 graph 数据添加到 summary。

**参数：**

- **model** (`torch.nn.Module`) – 待绘制的 Model
- **input_to_model** (`torch.Tensor` or `list` of `torch.Tensor`) – 输入模型的一个变量或变量 tuple
- **verbose** (`bool`) – 是否在 console 输出 graph 结构
- **use_strict_trace** (`bool`) – 是否将 `strict`  关键字参数传递给 `torch.jit.trace`。如果需要记录 mutable 容器类型（list, dict），应设置为 `False`。

### add_scalars

```python
add_scalars(main_tag, tag_scalar_dict, global_step=None, walltime=None)
```

添加标量数据到 summary。

**参数**：

- **main_tag** (`str`) – The parent name for the tags
- **tag_scalar_dict** (`dict`) – Key-value pair storing the tag and corresponding values
- **global_step** (`int`)

与 dict 中发送的值相关的索引，如 epoch。

- **walltime** (`float`) – Optional override default walltime (time.time()) seconds after epoch of event

**示例 1：**

```python
writer.add_scalars(
    main_tag='loss',
    tag_scalar_dict={'training': loss,
                    'validation': val_loss},
    global_step=epoch
)
```

**示例 2：**

```python
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()
r = 5
for i in range(100):
    writer.add_scalars('run_14h', {'xsinx':i*np.sin(i/r),
                                    'xcosx':i*np.cos(i/r),
                                    'tanx': np.tan(i/r)}, i)
writer.close()
# This call adds three values to the same scalar plot with the tag
# 'run_14h' in TensorBoard's scalar section.
```

![](2023-02-10-19-13-35.png)

## 参考

- https://pytorch.org/docs/stable/tensorboard.html
