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
    fig_list = os.listdir(path+"/"+age)
    # a wrong folder of the figures should be removed
    if age == "face_age":
        continue
    for fig in fig_list:
        path_new = path+"/"+age+"/"+fig
        # read in the fig
        data = mpimg.imread(path_new)
        # save it in x and y
        x.append(data)
        y.append(int(age))
        
# transform x and y into array
print(np.array(x).shape)
print(np.array(y).shape)

x = np.array(x)
y = np.array(y)
np.save("x_train",x)
np.save("y_train",y)