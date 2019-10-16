from helpers import *
from data import main as getData

class WideResLSTM(nn.Module):
    def __init__(self, num_classes, hidden_dim = 1, num_layers =1,
                 dropout = 0 , vgg_pretrained = True):
        super().__init__()
        self.num_classes = num_classes
        self.vision_model = WideResNet(num_groups = 2, N = 3, num_classes = 5, n_in_channels =3).features[:6]
        self.lstm = nn.LSTM(131072, hidden_dim, num_layers, dropout = dropout).cuda()
        self.fc = nn.Linear(65,self.num_classes).cuda()
        self.bn = nn.BatchNorm2d(32).cuda()
        
    def forward(self, X, verbose = False):
        init = True
        for seq in X:
            fm = self.bn(self.vision_model.forward(seq))
            enc = torch.flatten(fm, start_dim = 1).unsqueeze(0) if init else torch.cat([enc, torch.flatten(fm, start_dim = 1).unsqueeze(0)], dim = 0)
            init = False
        d = F.leaky_relu(self.lstm(enc)[0])
        out = F.softmax(self.fc(torch.flatten(d, start_dim = 1)))
        return out


if __name__ = "__main__":
    X_train_t, y_train_t, X_test_t, y_test_t = getData()
    wr_model = nn.DataParallel(WideResLSTM(5,num_layers=1, dropout=0.2).cuda())
    wr_model.train = train
    wr_model.test = test_acc
    losses=[]; valid_acc = []; train_acc = []
    num_epochs = 20
    wr_model.train(vgg_model,X_train_t, y_train_t, X_test_t, y_test_t, losses, valid_acc, train_acc, bs=2, epochs= num_epochs)

    # fig = plt.figure(figsize=(15,7.5), dpi= 80)
    # plt.subplot(1, 2, 1)
    # plt.plot(list(range(len(valid_acc))), valid_acc, label = "validation")
    # plt.plot(list(range(len(train_acc))), train_acc, label = "training")
    # plt.legend()
    # plt.title(f"Accuracy: (V:{valid_acc[-1]}, T:{train_acc[-1]})")
    # plt.subplot(1, 2, 2)
    # plt.plot(list(range(len(losses))), losses)
    # plt.title(f"Loss: {losses[-1]}")
    # plt.show()

    # import time
    # from IPython.display import clear_output
    # import matplotlib.pyplot as plt
    # import random
    # for _ in range(5):
    # seq=random.randint(0,9)+10*_
    # for idx in range(len(X[seq])):
    #     plt.imshow(X[seq][idx])
    #     plt.title('''Actual:{}
    #                 \nPrediction:{}'''.format(vocab[Y[seq]],
    #                                             vocab[torch.argmax(torch.flatten(wr_model.forward(X_t[seq].unsqueeze(0)))).item()]))
    #     plt.show()
    #     time.sleep(0.01)
    #     plt.close()
    #     clear_output(wait=True)


        