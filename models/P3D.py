from P3Dmodules import *

class P3D(BasicTrainableClassifier):
    def __init__(self, ni, no, **kwargs):
        super().__init__(**kwargs)
        self.model = nn.Sequential(P3DaBlock(ni, 64, padding=1),
                                   nn.MaxPool3d((1,2,2),(1,2,2)),
                                   P3DaBlock(64, 128, padding=1),
                                   nn.MaxPool3d(2,2, padding=(1,0,0)),
                                   P3DaBlock(128, 256, padding=1),
                                   P3DaBlock(256, 512, padding=1),
                                   nn.MaxPool3d(2,2),
                                   P3DaBlock(512, 512, padding=1),
                                   P3DaBlock(512, 512, padding=1),
                                   nn.MaxPool3d(2,2),
                                   P3DaBlock(512, 512, padding=1),
                                   P3DaBlock(512, 512, padding=1),
                                   nn.MaxPool3d(2,2),
                                   FlattenDim(1),
                                   nn.Linear(8192,4096),
                                   nn.Linear(4096,no)
                                  )
        init_cnn(self)
    def __call__(self,x): return self.model(x)