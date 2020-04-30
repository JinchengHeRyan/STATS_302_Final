import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# read in the data
import matplotlib.image as mpimg
# load the data
import os

path = "../input/facial-age/face_age"
age_list = os.listdir(path)
# set up the output of y
y = list()
# set up the output of x
x = list()
for age in age_list:
    # find the path of file names for ages
    fig_list = os.listdir(path + "/" + age)
    # a wrong folder of the figures should be removed
    if age == "face_age":
        continue
    for fig in fig_list:
        path_new = path + "/" + age + "/" + fig
        # read in the fig
        data = mpimg.imread(path_new)
        # save it in x and y
        x.append(data)
        y.append(int(age))

# transform x and y into array
print(np.array(x).shape)
print(np.array(y).shape)

import numpy as np
from keras.optimizers import SGD, Adam
from keras.callbacks import ModelCheckpoint
from Model.SSR_net import SSR_net

x = np.array(x)
y = np.array(y)

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

filepath = "../output/weights-improvement-{epoch:02d}-{val_loss:.2f}.h5"
checkpoint = ModelCheckpoint(filepath=filepath, monitor='val_loss', verbose=1, save_best_only=True)
callbacks_list = [checkpoint]
hist = ssrnet.fit(x_train, y_train, batch_size=50,
                  validation_data=(x_test, y_test),
                  epochs=50, verbose=1, callbacks=callbacks_list)
