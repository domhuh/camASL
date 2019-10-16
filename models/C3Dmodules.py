from rC3Dmodules import ConvRelu, init_cnn
from utils.base import *

class C3DBlock(BasicTrainableClassifier):
    def __init__(self,nc,nf,padding=0):
        super().__init__()
        self.block = nn.Sequential(ConvRelu(nc,nf,3,1,padding),
                                   ConvRelu(nf,nf,3,1,padding),
                                   nn.MaxPool3d(2,2, padding, ceil_mode=True))
    def forward(self,x): return self.block(x)