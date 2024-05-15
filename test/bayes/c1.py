def P(S, A):
    if set(A).issubset(set(S)):
        return len(A) / len(S)
    else:
        return 0
