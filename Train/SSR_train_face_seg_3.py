import numpy as np
from keras.optimizers import SGD, Adam
from keras.callbacks import ModelCheckpoint
from Model.SSR_net import SSR_net
from preprocess.Face_segmentation_trans import face_seg_transform

x, y = face_seg_transform()

print(x.shape)
print(y.shape)

ssrnet = SSR_net(image_size=200, stage_num=[3, 3, 3], lambda_local=0.25, lambda_d=0.25)()

ssrnet.compile(optimizer=Adam(), loss=["mae"], metrics={'pred_a': 'mae'})

ssrnet.summary()

validation_split = 0.333
data_num = len(x)
indexes = np.arange(data_num)
np.random.shuffle(indexes)
x = x[indexes]
y = y[indexes]
train_num = int(data_num * (1 - validation_split))

x_train = x[:train_num]
x_test = x[train_num:]
y_train = y[:train_num]
y_test = y[train_num:]

x = 0
y = 0

filepath = "../Output/output_3/weights-improvement-{epoch:02d}-{val_loss:.2f}.h5"
checkpoint = ModelCheckpoint(filepath=filepath, monitor='val_loss', verbose=1, save_best_only=True)
callbacks_list = [checkpoint]
hist = ssrnet.fit(x_train, y_train, batch_size=50,
                  validation_data=(x_test, y_test),
                  epochs=50, verbose=1, callbacks=callbacks_list)

np.save('../Output/output_3/hist.npy', hist.history, allow_pickle=True)
