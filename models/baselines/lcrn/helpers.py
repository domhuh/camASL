from sklearn.metrics import accuracy_score
from _imports import *

torch.autograd.set_detect_anomaly(True)

def train(self, X_train, Y_train, X_test, Y_test,
          losses= None, valid_acc=None, train_acc=None, bs=16, epochs=1, verbose = False):
    crit = nn.CrossEntropyLoss()
    if not isinstance(self,nn.DataParallel):
        op = torch.optim.Adam([*self.lstm.parameters(), *self.fc.parameters(), *self.bn.parameters()], lr=1e-3)
    else:
        op = torch.optim.Adam([*self.module.lstm.parameters(), *self.module.fc.parameters(), *self.module.bn.parameters()], lr=1e-3)

    eps = tnrange(epochs)
    for e in eps:
        for idx in range(0,X_train.shape[0], bs):
            op.zero_grad()
            pred = self.forward(X_train[idx:idx+bs], verbose = verbose)
            loss = crit(pred, Y_train[idx:idx+bs])
            loss.backward(retain_graph = True)
            op.step()
        losses.append(loss.item())
        if not e%2:
            valid_acc.append(self.test(self, X_test, Y_test, bs)) if valid_acc is not None else None
            train_acc.append(self.test(self, X_train, Y_train, bs)) if train_acc is not None else None

        eps.set_description("Epoch %d, Loss:( %.4f), Valid:(%.2f)" %(e+1, loss.item(),valid_acc[-1]))
        
def test_acc(self, X, Y, bs = 16):
    preds = []
    for idx in range(0,X.shape[0], bs):
        preds.extend([torch.argmax(p) for p in torch.flatten(self.forward(X[idx:idx+bs]),start_dim=1)])
    y_pred = list(map(lambda p: p.cpu().numpy().item(), preds))
    Y = list(map(lambda y: y.cpu().numpy().item(), Y))
    return accuracy_score(y_pred, Y)