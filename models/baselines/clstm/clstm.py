import torch
import torchvision
from torch.autograd import Variable
import torch.utils.data as D
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.utils.rnn as R
import dill 
from tqdm import tqdm
import time

def log_time(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        print('%s ran for %.1f s' %(method.__name__, (te - ts)))
        return result
    return timed

def fpool(input_c):
    return nn.Conv2d(in_channels = input_c, out_channels=1, kernel_size=1)

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class CLSTM_Cell(nn.Module):
    def __init__(self, input_size,
                 hidden_dim, kernel_size,
                 output_size, bias=True, use_cuda=False):
        super().__init__()
        
        self.input_dim, self.height, self.width = input_size
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.bias = bias
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.use_cuda = use_cuda
        self.conv = self.localize(nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels= 4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias))
        
    def forward(self, x, chs):
        x = torch.cat([x, torch.sigmoid(chs[1])], dim=1)
        x = F.relu(self.conv(x))
        f,c,i,o = torch.split(x, self.hidden_dim, dim=1) 
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * chs[0] + i * g
        h_next = o * torch.tanh(c_next)
        
        return c_next, h_next


    def init_hidden(self, batch_size):
        return (Variable(torch.zeros(batch_size, self.hidden_dim, self.height, self.width)),
                Variable(torch.zeros(batch_size, self.hidden_dim, self.height, self.width)))
    
    def localize(self, x):
        if self.use_cuda:
            return x.cuda()
        else:
            return x

    @log_time
    def train(self, data, labels, epochs, lr, path=None):
        optimizer=optim.Adam(self.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        for e in range(epochs):
            optimizer.zero_grad()
            running_loss = 0
            for idx, seq in enumerate(data):
                (c_next, h_next) = self.init_hidden(1)
                for frame in tqdm(seq):
                    c_next, h_next = self.forward(frame.transpose(0,2).reshape(1,3,224,224),
                        (c_next,h_next))
                prediction = F.relu(self.fc(self.flatten(self.combine_chs(c_next, h_next))))
                loss = criterion(prediction,
                                torch.argmax(torch.tensor(labels[idx], dtype=torch.long).reshape(1,5)).reshape(1))
                loss.backward(retain_graph = True)
                optimizer.step()
                running_loss += loss.item()

            print("Epoch {}/{}: Loss ({:0.2f})\n".format(e+1,epochs,running_loss))

            if e%10 == 0 and path != None:
                torch.save(src.model, path+"".format(running_loss), pickle_module=dill)


if __name__ == "__main__":
	img_size = (3, 224,224)
	lstm_hd = 128
	cnn_ks = [3,3]
	num_classes = 5
	test = CLSTM_Cell(img_size, lstm_hd, cnn_ks, num_classes)
	for param in test.parameters():
	    print(type(param.data), param.size())