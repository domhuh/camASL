
# from fastai.vision import Path
# paths = [i.ls()[0] for i in Path("../data/s1").ls()]
# paths += [i.ls()[0] for i in Path("../data/s2").ls()]
# paths

# def getData(path, **kwarg):
#     xt_rgb, yt_rgb = getTensors(path)
#     idx = get_idx(xt_rgb)
#     x_train_rgb, x_test_rgb, y_train_rgb, y_test_rgb = split_slice(xt_rgb, yt_rgb, idx)
#     del xt_rgb, yt_rgb; gc.collect()
#     train_dl_rgb = ttodl(x_train_rgb, y_train_rgb, **kwarg)
#     valid_dl_rgb = ttodl(x_test_rgb, y_test_rgb, **kwarg)
#     return train_dl_rgb, valid_dl_rgb


# c3d = C3D(3,16).cuda()

# for e in range(10):
#     for p in paths:
#         train_dl_rgb, valid_dl_rgb = getData(p, bs=8)
#         gc.collect()
#         c3d.fit(train_dl_rgb, valid_ds=valid_dl_rgb, epochs=1,
#                     cbs=True, learning_rate=1e-5)
#         train_dl_rgb, valid_dl_rgb = None, None
#         gc.collect()