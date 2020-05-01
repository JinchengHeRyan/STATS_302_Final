import os
import numpy as np
import cv2
import time
import matplotlib.image as mpimg

face_cascade = cv2.CascadeClassifier('../Age_Rcg/lbpcascade_frontalface_improved.xml')


def face_seg_save(filepath: str, temp_file_path: str):
    img_size = (200, 200)
    try:
        os.mkdir('../temp_files/')
        os.mkdir(temp_file_path)
    except Exception:
        pass
    age_list = os.listdir(filepath)
    for age in age_list:
        fig_list = os.listdir(filepath + '/' + age)
        for fig in fig_list:
            fig_path = filepath + '/' + age + '/' + fig
            input_image = cv2.imread(fig_path)
            gray_img = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
            detected = face_cascade.detectMultiScale(gray_img, 1.1)
            if len(detected) > 0:
                detected = detected[0]
                x, y, w, h = detected
                x_1 = x
                y_1 = y
                x_2 = x + w
                y_2 = y + h
                face = cv2.resize(input_image[y_1: y_2 + 1, x_1: x_2 + 1, :], img_size)
                temp_path = temp_file_path + '/' + age + '/' + fig
                try:
                    os.mkdir(temp_file_path + '/' + age)
                except Exception:
                    pass
                cv2.imwrite(temp_path, img=face)


def face_seg_transform():
    filepath = '../data/age'
    temp_file_path = '../temp_files/age_face_seg' + str(int(time.time()))
    face_seg_save(filepath=filepath, temp_file_path=temp_file_path)

    path = temp_file_path
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
    x = np.array(x)
    y = np.array(y)
    return x, y
