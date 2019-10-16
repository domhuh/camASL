from C3Dmodules import *

class C3D(BasicTrainableClassifier):
    def __init__(self, nc, no, **kwargs):
        super().__init__(**kwargs)
        self.cp_model = nn.Sequential(ConvRelu(nc, 64, 3, 1, 1),
                                      nn.MaxPool3d((1,2,2),(1,2,2)),
                                      ConvRelu(64, 128, 3, 1, 1),
                                      nn.MaxPool3d((2,2,2),2,(1,0,0)),
                                      C3DBlock(128,256,padding=(1,0,0)),
                                      C3DBlock(256,512,padding=1),
                                      C3DBlock(512,512,padding=(1,0,0)),
                                      C3DBlock(512,512,padding=1))
        self.fc = nn.Sequential(nn.Linear(4096,4096),
                                nn.Linear(4096,no))
        init_cnn(self)

    def forward(self, x):
        return self.fc(torch.flatten(self.cp_model(x), start_dim=1))