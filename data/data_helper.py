from utils.base import *

from torch.utils.data import DataLoader, TensorDataset
from functools import partial

loadnp = partial(np.load, allow_pickle=True)

def getTensors(path):
    xn = np.array([])
    yn = np.array([])
    for p in path.ls():
        xn, yn = np.append(xn,loadnp(p/'x.npy')), np.append(yn,loadnp(p/'y.npy'))
    xt, yt = nptotorch(xn), torch.Tensor(yn)
    return xt, yt

def nptotorch(x):
    return ([torch.stack([torch.Tensor(img).transpose(0,2) for img in vid]) for vid in x])

def ttodl(x,y, bs=16, sffl= True):
    if not x.device.type == "cpu": x=x.cpu()
    if not y.device.type == "cpu": y=y.cpu()
    return DataLoader(TensorDataset(x, y), batch_size=bs, shuffle=sffl)

def shuffle(X,Y, idx):
    try: return X[idx], Y[idx]
    except: return [X[i] for i in idx], Y[idx]

def sliceseq(x, y, sl):
    xs, ys = None, None
    for idx, vid in enumerate(x):
        sframes = torch.stack(torch.split(vid, sl), dim=0) if not vid.shape[0]%sl else torch.stack(torch.split(vid[:vid.shape[0]//sl*sl], sl), dim=0)
        xs = sframes if xs is None else torch.cat((xs, sframes))
        ys = torch.stack([y[idx] for _ in range(sframes.shape[0])]) if ys  is None else torch.cat((ys, torch.stack([y[idx] for _ in range(sframes.shape[0])])))
    return xs,ys

def split_slice(x, y, idx, split = 0.6, sl = 3):
    x_train, x_test, y_train, y_test = train_valid(x, y, split, idx)
    # assert(set([i.item() for i in y_train])==set([i.item() for i in y_test]))
    x_train, y_train = sliceseq(x_train, y_train, sl)
    x_test, y_test = sliceseq(x_test, y_test, sl)
    ix = get_idx(x_test)
    x_test, y_test = shuffle(x_test, y_test, ix)
    return x_train, x_test, y_train, y_test
    
def train_valid(X,Y, split, idx):
    train = idx[:round(len(X)*split)]
    valid = idx[round(len(X)*split):]
    try: return X[train], X[valid], Y[train], Y[valid]
    except: return [X[i] for i in train], [X[i] for i in valid], Y[train], Y[valid]