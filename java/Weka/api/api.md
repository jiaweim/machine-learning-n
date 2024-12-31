# Weka API 总结

## Filter

### Remove

`Remove` 用于删除特征（属性）。有两个选项：

- `-R`，指定删除的特征，例如 `-R 1` 表示删除第一列
- `-V`，是否反转，反转时表示保留 `-R` 指定特征

使用示例：

```java
Remove remove = new Remove();
remove.setOptions(new String[]{"-R", "1"});
remove.setInputFormat(data);
data = Filter.useFilter(data, remove);
```

