"""
Linear positional embeddings  used in paper `Dynamic Memory Networks for Visual Question Answering`
"""

text = "Hello my name is John and I very like cookies"
D = 10
M = len(text.split())

import numpy as np

# According to original paper
# we shall use 1-based indexing

l = np.zeros(shape=(D, M))
for j in range(l.shape[1]):
    for d in range(l.shape[0]):
        l[j][d] = (1 - (j + 1) / M) - ((d + 1) / D) * (1 - 2 * (j + 1) / M)

print(l)
print("\n## Weights for word embedding no. 1 ##")
print(l[:, 1])
