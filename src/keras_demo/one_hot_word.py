import numpy as np

"""
单词级的 one-hot 编码
初始数据：每个样本是列表中的一个元素
本例中的样本是一个句子
"""

samples = ['The cat sat on the mat.', 'The dog ate my homework.']

token_index = {}  # 构建数据中所有标记的索引
for sample in samples:
    # 利用 split 方法对样本进行分词。在实际应用中，还需要从样本中去掉标点和特殊字符
    for word in sample.split():
        if word not in token_index:
            token_index[word] = len(token_index) + 1  # 为每个唯一单词指定一个唯一索引，注意，没有为索引编号 0 指定单词

max_length = 10  # 对样本进行分词，只考虑每个样本前 max_length 个单词

# 结果保存在 results 中，第 i 个样本的第 j 个单词的 index
results = np.zeros(shape=(len(samples),
                          max_length,
                          max(token_index.values()) + 1))
for i, sample in enumerate(samples):
    for j, word in list(enumerate(sample.split()))[:max_length]:
        index = token_index.get(word)
        results[i, j, index] = 1.
