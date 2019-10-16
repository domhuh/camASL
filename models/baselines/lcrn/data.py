from _imports import *

def main():
    path_data = Path("../input/als-vid-img/data/data")

    vocab = ['alarm' , 'lock', 'movie', 'rain', 'weather']
    dim = 128
    X = np.array([])
    for cat in tqdm(vocab):
        path_tmp = path_data/cat
        imgs = path_tmp.ls()
        cat_imgs = []
        for im in imgs:
    #         cat_imgs.append(np.array(
    #             [np.array(cv2.resize(cv2.imread(str(i)),(dim,dim)))/255 for i in im.ls()]))
            cat_imgs.append(np.array(
                 [np.array(PIL.Image.open(i).resize((dim,dim)))/255 for i in im.ls()]))
        cat_imgs = np.array(cat_imgs)
        X = np.append(X,cat_imgs)

    pad_length = max([seq.shape[0] for seq in X])
    X_padded = pad_sequences(X, pad_length, 'float64', 'post')

    X_train_t = torch.stack(list(
            map(lambda seq: torch.stack(list(
                map(lambda img: torch.cuda.FloatTensor(img).transpose(0,2)
                    ,seq))).cuda()
                ,X_train)))
    X_test_t = torch.stack(list(
            map(lambda seq: torch.stack(list(
                map(lambda img: torch.cuda.FloatTensor(img).transpose(0,2)
                    ,seq))).cuda()
                ,X_test)))
    X_t = torch.cat((X_train_t,X_test_t))


    y_train_t = torch.cuda.LongTensor(y_train)
    y_test_t = torch.cuda.LongTensor(y_test)

    # path_save = Path("../working")
    # os.mkdir(path_save/"out") if not os.path.exists(path_save/"out") else None
    # torch.save(X_train_t, path_save/"out"/'x_train.pt'), torch.save(y_train_t, path_save/"out"/'y_train.pt')
    # torch.save(X_test_t, path_save/"out"/'x_test.pt'), torch.save(y_test_t, path_save/"out"/'y_test.pt')


    return X_train_t, y_train_t, X_test_t, y_test_t