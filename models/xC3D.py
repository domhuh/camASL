from utils.base import *
from rC3D import rC3D, init_cnn
import cv2 as cv
import pickle

class genDepthC3D(BasicTrainableClassifier):
    def __init__(self, in_c, num_classes, path, **kwargs):
        super().__init__(**kwargs)
        with open(path, "rb") as mfile:
            self.gan = pickle.load(mfile)
        self.model = rC3D(in_c, num_classes)
        init_cnn(self)
    def __call__(self, x):
        gX = torch.cat([self.gan.generator.forward(img)[None,:] for img in x], dim=0) 
        return self.model(gX)

class OpticFlowC3D(BasicTrainableClassifier):
    def __init__(self, in_c, num_classes,**kwargs):
        super().__init__(**kwargs)
        self.model = rC3D(in_c, num_classes)
        
    def __call__(self,x):
        out = None
        with torch.no_grad(): 
            for b in x:
                first = self.tonp(b[0])
                second = self.tonp(b[-1])
                ro = self.denseOF(first, second)
                out = self.tot(ro)[None,None,] if out is None else torch.cat((out,self.tot(ro)[None,None,]),
                                                                        dim=0)
        return self.model(out)
    
    def denseOF(self, first, second):
        prev = cv.cvtColor(first, cv.COLOR_BGR2GRAY)
        mask = np.zeros_like(first)
        mask[..., 1] = 255
        last = cv.cvtColor(second, cv.COLOR_BGR2GRAY)
        flow = cv.calcOpticalFlowFarneback(prev, last,
                                           None, 0.5, 3, 15, 3, 5, 1.2, 0)
        magnitude, angle = cv.cartToPolar(flow[..., 0], flow[..., 1])
        mask[..., 0] = angle * 180 / np.pi / 2
        mask[..., 2] = cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX)
        return cv.cvtColor(mask, cv.COLOR_HSV2BGR)
    
    def tonp(self,x): return x.transpose(0,2).cpu().numpy()
    def tot(self,x): return torch.Tensor(x).transpose(0,2).cuda().data.detach()
