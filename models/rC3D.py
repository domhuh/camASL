from rC3Dmodules import *

class rC3D(BasicTrainableClassifier):
    def __init__(self, in_c, num_classes, **kwargs):
        super().__init__(**kwargs)
        self.ls = nn.Sequential(rC3DBlockMP(in_c, 64, True),
                                rC3DBlockMP(64, 128, True),
                                rC3DBlockMP(128, 256, True),
                                rC3DBlock(256, 512),
                                rC3DBlock(512, 512))
        self.pool = StackPool(1)
        self.fc = nn.Linear(256,num_classes)
        init_cnn(self)
    def __call__(self, X):
        fm = self.ls(X)
        pfm = self.pool(fm)
        ft = self.fc(torch.flatten(pfm,start_dim=1))
        return ft