import numpy as np 
import pandas as pd
from cv2 import *
from fastai import *
from fastai.vision import *
import string

path_str = "./data/asl_alphabet_train"
path = Path(path_str)

classes = list(string.ascii_uppercase)
classes.append(['del','space','nothing'])

data = ImageDataBunch.from_folder(path, train=".", valid_pct=0.2,
        ds_tfms=get_transforms(), size=224, num_workers=1).normalize(imagenet_stats)

learn = create_cnn(data, models.resnet50, metrics= [error_rate,accuracy])

learn.fit_one_cycle(4)
learn.save('stage-1')
learn.unfreeze()
learn.fit_one_cycle(2, max_lr=slice(7e-6,8e-6))
learn.save('stage-2')
learn.path = Path("../model")

learn.export()