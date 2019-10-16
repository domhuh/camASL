from fastai import *
from fastai.vision import *
import numpy as np
import torch
import dill 

def convert2torch(src, dest):
	torch.save(src.model, "{}/{}".format(dest,"torch_model.pkl"), pickle_module=dill)

def convert2fastai(data, model):
	learn = Learner(data, model)
	return learn

def combineModels(model_1, model_2):
	list_of_layers = list(model_1.children())
	list_of_layers.extend(list(model_2.children()))
	model_3 = nn.Sequential (*list_of_layers)
	return model_3