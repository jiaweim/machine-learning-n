import numpy as np

samples = ['The cat sat on the mat.', 'The dog ate my homework.']

# 将单词保存为长度为 1000 的向量，如果单词数量接近 1000 个（或更多），那么会遇到很多散列冲突，这回降低这种编码方法的准确性
dimensionality = 1000
max_length = 10

results = np.zeros((len(samples),
                    max_length,
                    dimensionality))
for i, sample in enumerate(samples):
    for j, word in list(enumerate(sample.split()))[:max_length]:
        index = abs(hash(word)) % dimensionality  # 将单词散列为 0~1000 范围内的一个随机整数索引
        results[i, j, index] = 1.
