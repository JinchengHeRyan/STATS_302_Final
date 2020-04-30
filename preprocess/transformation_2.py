import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# read in the data
import matplotlib.image as mpimg
# load the data
import os
import random


def data_loader_2():
    path = "../input/age"
    age_list = os.listdir(path)
    num_each_folder = 200
    y = list()
    x = list()
    for age in age_list:
        list_fig = os.listdir(path + "/" + age)
        if len(list_fig) <= num_each_folder:
            for i in range(1, num_each_folder + 1):
                try:
                    data = mpimg.imread(path + "/" + age + "/" + str(i) + ".png")
                    x.append(data)
                    y.append(int(age))
                except:
                    continue
        else:
            n = len(list_fig)
            order = random.sample([a for a in range(n)], num_each_folder)
            for i in order:
                i += 1
                data = mpimg.imread(path + "/" + age + "/" + str(i) + ".png")
                x.append(data)
                y.append(int(age))
    x = np.array(x)
    y = np.array(y)
    return x, y
