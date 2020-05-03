import numpy as np
from keras.optimizers import SGD, Adam
from keras.callbacks import ModelCheckpoint
from Model.SSR_net import SSR_net
from preprocess.Face_seg_more import face_seg_transform
import os

x, y = face_seg_transform()

print(x.shape)
print(y.shape)

ssrnet = SSR_net(image_size=200, stage_num=[3, 3, 3], lambda_local=0.25, lambda_d=0.25)()

ssrnet.compile(optimizer=Adam(), loss=["mae"], metrics={'pred_a': 'mae'})

ssrnet.summary()

# validation_split = 0.333
# data_num = len(x)
# indexes = np.arange(data_num)
# np.random.shuffle(indexes)
# x = x[indexes]
# y = y[indexes]
# train_num = int(data_num * (1 - validation_split))
#
# x_train = x[:train_num]
# x_test = x[train_num:]
# y_train = y[:train_num]
# y_test = y[train_num:]
#
# x = 0
# y = 0
output_path = '../Output/output_4'
try:
    os.makedirs(output_path)
except Exception:
    pass

filepath = output_path + "/weights-improvement-{epoch:02d}-{val_loss:.2f}.h5"
checkpoint = ModelCheckpoint(filepath=filepath, monitor='val_loss', verbose=1, save_best_only=True)
callbacks_list = [checkpoint]
hist = ssrnet.fit(x, y, batch_size=50,
                  epochs=50, verbose=1, callbacks=callbacks_list,
                  shuffle=True, validation_split=0.3)

np.save(output_path+'/hist.npy', hist.history, allow_pickle=True)
