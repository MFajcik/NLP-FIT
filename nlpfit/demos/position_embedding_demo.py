
text = "Hello my name is John and I very like cookies"
D = 10
M = len(text.split())

import numpy as np

l = np.zeros(shape=(D, M))
for j in range(l.shape[1]):
    for d in range(l.shape[0]):
        l[j][d]=(1-j/M) - (d/D)*(1-2*j/M)

print(l)
print("##")
print(l[:,1])