import numpy as np


def preprocess(text):
    """

    Parameters
    ----------
    text

    Returns
    -------

    """
    text = text.lower()
    text = text.replace('.', ' .')
    words = text.split(' ')

    word_to_id = {}
    id_to_word = {}
    for word in words:
        if word not in word_to_id:
            new_id = len(word_to_id)
            word_to_id[word] = new_id
            id_to_word[new_id] = word
    corpus = np.array([word_to_id[w] for w in words])
    return corpus, word_to_id, id_to_word


def create_co_matrix(corpus, vocab_size, window_size=1):
    """

    Parameters
    ----------
    corpus 单词 ID 列表
    vocab_size 词汇个数
    window_size 窗口大小

    Returns 共现矩阵
    -------

    """
    corpus_size = len(corpus)
    co_matrix = np.zeros((vocab_size, vocab_size), dtype=np.int32)

    for idx, word_id in enumerate(corpus):
        for i in range(1, window_size + 1):
            left_idx = idx - i
            right_idx = idx + i
            if left_idx >= 0:
                left_word_id = corpus[left_idx]
                co_matrix[word_id, left_word_id] += 1
            if right_idx < corpus_size:
                right_word_id = corpus[right_idx]
                co_matrix[word_id, right_word_id] += 1
    return co_matrix


def cos_similarity(x, y, eps=1e-8):
    """
    计算余弦相似度
    Parameters
    ----------
    x 向量 x
    y 向量 y
    eps 为了避免出现除以 0添加的量

    Returns 两个向量的余弦相似度
    -------

    """
    nx = x / (np.sqrt(np.sum(x ** 2)) + eps)  # x 的正规化
    ny = y / (np.sqrt(np.sum(y ** 2)) + eps)  # y 的正规化
    return np.dot(nx, ny)


def most_similar(query, word_to_id, id_to_word, word_matrix, top=5):
    """
    将 query 作为查询词，将与这个查询词相似的单词按降序显示出来
    Parameters
    ----------
    query 查询词
    word_to_id 单词到单词ID的 dict
    id_to_word 单词ID到单词的 dict
    word_matrix 汇总了单词向量的矩阵，假定保存了与各行对应的单词向量
    top 显示到前几位

    Returns
    -------

    """
    if query not in word_to_id:
        print('%s is not found' % query)
        return
    print('\n[query] ' + query)
    query_id = word_to_id[query]
    query_vec = word_matrix[query_id]

    vocab_size = len(id_to_word)
    similarity = np.zeros(vocab_size)
    for i in range(vocab_size):
        similarity[i] = cos_similarity(word_matrix[i], query_vec)

    count = 0
    for i in (-1 * similarity).argsort():
        if id_to_word[i] == query:
            continue
        print(" %s: %s" % (id_to_word[i], similarity[i]))
        count += 1
        if count >= top:
            return


def ppmi(C, verbose=False, eps=1e-8):
    """
    通过共现矩阵计算 PPMI 矩阵
    Parameters
    ----------
    C 共现矩阵
    verbose 是否输出运行情况的标志。当处理大语料库时，设置 verbose=True 可用于确认运行情况
    eps

    Returns PPMI 矩阵
    -------

    """
    M = np.zeros_like(C, dtype=np.float32)
    N = np.sum(C)
    S = np.sum(C, axis=0)
    total = C.shape[0] * C.shape[1]
    cnt = 0

    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            pmi = np.log2(C[i, j] * N / (S[j] * S[i]) + eps)
            M[i, j] = max(0, pmi)

            if verbose:
                cnt += 1
                if cnt % (total // 100 + 1) == 0:
                    print('%.1f%% done' % (100 * cnt / total))

    return M
