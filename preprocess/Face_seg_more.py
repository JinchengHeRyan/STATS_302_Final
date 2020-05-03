import os
import numpy as np
import cv2
import time
# import matplotlib.image as mpimg

face_cascade = cv2.CascadeClassifier('../Age_Rcg/lbpcascade_frontalface_improved.xml')


def face_seg_save(filepath: str, temp_file_path: str):
    img_size = (200, 200)
    ad = 0.25

    if not os.path.exists(temp_file_path):
        os.makedirs(temp_file_path)

    age_list = os.listdir(filepath)
    for age in age_list:
        try:
            temp = int(age)  # Check whether is valid directory
        except Exception:
            continue
        fig_list = os.listdir(filepath + '/' + age)

        count = 0
        for fig in fig_list:
            fig_path = filepath + '/' + age + '/' + fig
            input_image = cv2.imread(fig_path)

            img_h, img_w, _ = np.shape(input_image)

            gray_img = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
            detected = face_cascade.detectMultiScale(gray_img, 1.1)
            if len(detected) == 1:
                detected = detected[0]
                x, y, w, h = detected
                x_1 = x
                y_1 = y
                x_2 = x + w
                y_2 = y + h

                xw1 = max(int(x_1 - ad * w), 0)
                yw1 = max(int(y_1 - ad * h), 0)
                xw2 = min(int(x_2 + ad * w), img_w - 1)
                yw2 = min(int(y_2 + ad * h), img_h - 1)

                face = cv2.resize(input_image[yw1: yw2 + 1, xw1: xw2 + 1, :], img_size)
                temp_path = temp_file_path + '/' + age + '/' + fig
                try:
                    os.makedirs(temp_file_path + '/' + age)
                except Exception:
                    pass
                cv2.imwrite(temp_path, img=face)
                count += 1
            if count == 200:
                break


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

        for fig in fig_list:
            path_new = path + "/" + age + "/" + fig
            # read in the fig
            data = cv2.imread(path_new) / 255
            # save it in x and y
            x.append(data)
            y.append(int(age))

            if len(fig_list) < 100:
                x.append(np.flip(data, axis=1))
                y.append(int(age))

    # transform x and y into array
    x = np.array(x)
    y = np.array(y)
    return x, y
