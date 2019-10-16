import numpy as np
import matplotlib.pyplot as plt

def shapes(*x):
    for idx, item in enumerate(x): print(f"arg_{idx}: {item.shape}") if hasattr(item, "shape") else print(f"arg_{idx}: {len(item)}")

def pltlen(x, label_=None): plt.plot(list(range(len(x))), x, label = label_)

def stats(x): print(x.mean().item(), x.std().item())

def flatten(x): return x.reshape(np.multiply(*x.shape))

def get_idx(X):
    return np.random.permutation(X.shape[0]) if hasattr(X, 'shape') else np.random.permutation(len(X))

def get_layers(start, hs, end, step):
    lse = [*list(range(hs, end, -step)), end]
    return list(zip([start,*lse[:]], [*lse[:], end]))[:-1]
