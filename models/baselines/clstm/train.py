from clstm import CLSTM_Cell
import numpy as np

def load_data(x_path, y_path):
	X = np.load(x_path,allow_pickle=True)
	Y = np.load(y_path,allow_pickle=True)
	for i in range(Y.shape[0]):
		Y[i] = Y[i][0]
	return X,Y


if __name__ == "__main__":
	X,Y = load_data("./data/X_torch.npy", "./data/Y.npy")
	img_size = (3, 224,224)
	lstm_hd = 1
	cnn_ks = [3,3]
	num_classes = 16

	baseline_model = CLSTM_Cell(img_size, lstm_hd, cnn_ks, num_classes)

	baseline_model.train(data=X, labels=Y, epochs = 20, lr=1e-6)