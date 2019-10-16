import torch
import torch.nn as nn

class Module(nn.Module):
    def __pre_init__(self): super().__init__()
    def __init__(self): pass

def init_cnn(m, *layers):
    if getattr(m, 'bias', None) is not None: nn.init.constant_(m.bias, 0)
    if isinstance(m, (layers)): nn.init.kaiming_normal_(m.weight)
    for l in m.children(): init_cnn(l)

def cc(x, y=None):
    if torch.cuda.device_count():
        if not y: return x.cuda()
        return x.cuda().long() if y =="long" else x.cuda().float()
    if not y: return x
    return x.long() if y =="long" else x.float()

class Flatten(Module):
    def __call__(self,x): return torch.flatten(x, start_dim=1)


class noop(Module):
    def __call__(self, x): return x

class FlattenDim(Module):
    def __init__(self, dim): self.dim=dim
    def __call__(self,x): return torch.flatten(x, start_dim=self.dim)