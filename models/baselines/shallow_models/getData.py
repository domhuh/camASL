import numpy as np 
import pandas as pd
import os
from fastai.vision import Path
from tqdm import tqdm
import PIL
import gc

def main():
    path_data = Path("../input/data/data")

    vocab = ['alarm' , 'lock', 'movie', 'rain', 'weather']
    dim = 224
    X = np.array([])
    for cat in tqdm(vocab):
        path_tmp = path_data/cat
        imgs = path_tmp.ls()
        cat_imgs = []
        cat_imgs_ten = []
        for im in imgs:
            seq = []
            for i in im.ls():
                img = np.array(PIL.Image.open(i).resize((dim,dim)))/255
                seq.append(img)
            cat_imgs.append(np.array(seq))
        cat_imgs = np.array(cat_imgs)
        X = np.append(X,cat_imgs)

    X_pca = []
    pca = PCA(224)
    max_idx = 0
    X_pca_tmp = []
    for seq in tqdm(X):
        seq_t = []
        for im in seq:
            pca.fit(im.reshape(224,224*3))
            tmp_idx = np.where(np.cumsum(pca.explained_variance_ratio_)>0.98)[0][0]
            if max_idx < tmp_idx:
                max_idx = tmp_idx
            seq_t.append(np.array(pca.singular_values_))
        X_pca_tmp.append(np.array(seq_t))
    min_seq = min([i.shape[0] for i in X])
    pca = PCA(max_idx if max_idx<min_seq else min_seq)
    for seq in tqdm(X_pca_tmp):
        pca.fit(seq)
        X_pca.append(np.array(pca.singular_values_))
    X_pca = np.array(X_pca)

    Y =  np.array([np.argmax(np.array(pd.get_dummies(vocab).iloc[u//10])) for u in range(50)])

    X_train, X_test, y_train, y_test = train_test_split(X_pca, Y, test_size=0.30, shuffle=True)

    return X_train, X_test, y_train, y_test