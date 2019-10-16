from utils.base import *

class rC3DBlock(nn.Module):
    def __init__(self, in_c, out_c, ks=3,**kwargs):
        super().__init__(**kwargs)
        self.rconv = ResConv(in_c, ks, ks//2)
        self.bn = nn.BatchNorm3d(in_c)
        self.conv = ConvRelu(in_c, out_c, ks, 1, ks//2)
    def __call__(self, X):
        fm = self.bn(F.leaky_relu(self.rconv(X) + X).sub(0.5))
        return self.conv(fm)

class rC3DBlockMP(nn.Module):
    def __init__(self, in_c, out_c, pad=False, **kwargs):
        super().__init__(**kwargs)
        self.rconv = ResConv(in_c, 3, 1)
        self.conv = ConvRelu(in_c, out_c, 3, 1, 1)
        self.bn = nn.BatchNorm3d(in_c)
        self.mp = nn.MaxPool2d((2,2), 2, 1) if pad else nn.MaxPool2d((2,2), 2)
    def __call__(self, X):
        fm = self.bn(F.leaky_relu(self.rconv(X) + X))
        init = True
        for img in fm1:
            pfm = self.mp(img).unsqueeze(0) if init else torch.cat([pfm1, self.mp(img).unsqueeze(0)], dim = 0)
            init = False
        return F.leaky_relu(self.conv(pfm))

class StackPool(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.max = nn.AdaptiveMaxPool3d(size)
        self.avg = nn.AdaptiveAvgPool3d(size)
    def __call__(self, X):
        return torch.cat((self.max(X), self.avg(X)), dim=1)

def init_cnn(m):
    if getattr(m, 'bias', None) is not None: nn.init.constant_(m.bias, 0)
    if isinstance(m, (nn.Conv3d,nn.Conv2d,nn.Linear)): nn.init.kaiming_normal_(m.weight)
    for l in m.children(): init_cnn(l)

def ConvRelu(in_c, out_c, ks, stride, padding=0):
    conv1 = nn.Conv3d(in_c, out_c, ks, stride, padding)
    return nn.Sequential(conv1,nn.LeakyReLU())

def ResConv(in_c, kernel_size, stride):
    return nn.Sequential(ConvRelu(in_c, in_c//2, 1, 1),
                         ConvRelu(in_c//2, in_c//2, kernel_size, stride, padding=kernel_size//2),
                         ConvRelu(in_c//2, in_c, 1, 1))