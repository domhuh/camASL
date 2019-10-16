from utils.base import *
from fastai.layers import conv_layer

def P3Dconv(ni,nf,stride=1,padding=1, s=True, t=False):
    return nn.Conv3d(ni,nf,kernel_size=(3,1,1) if not s or t else (1,3,3),stride=1,
                     padding=padding,bias=False)

class ResBlockA(Module):
    def __call__(self, x): return self.bn(self.act(self.layer(x)+self.idlayer(x)))

class P3DaBlock(ResBlockA):
    def __init__(self, ni, nf, pad=True,**kwargs):
        super().__init__()
        self.layer = nn.Sequential(nn.Conv3d(ni,nf,1),
                                   P3Dconv(nf,nf,padding=(0,1,1) if pad else 0, s=True),
                                   P3Dconv(nf,nf,padding=(1,0,0) if pad else 0, t=True),
                                   nn.Conv3d(nf,nf,1),)
        self.idlayer = nn.Conv3d(ni, nf, 3, **kwargs)
        self.act = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm3d(nf)
