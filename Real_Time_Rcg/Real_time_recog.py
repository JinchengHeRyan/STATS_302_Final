import cv2
import os
import timeit
import numpy as np
from Model.SSR_net import SSR_net
import keras
import matplotlib.image as mpimg

model = SSR_net(image_size=200, stage_num=[3, 3, 3], lambda_local=0.25, lambda_d=0.25)()
model.load_weights('../output/weights-improvement-44-6.48.h5')

face_cascade = cv2.CascadeClassifier('lbpcascade_frontalface_improved.xml')

try:
    os.mkdir('./img')
except OSError:
    pass

# load model and weights
img_size = (200, 200)
stage_num = [3, 3, 3]
cap = cv2.VideoCapture(0)
img_idx = 0
# detected = ''  # make this not local variable
skip_frame = 5  # every 5 frame do 1 detection and network forward propagation
ad = 0.5


def draw_label(input_img, loc, label):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    thickness = 2
    text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
    x, y = loc
    cv2.rectangle(input_img, (x, y - text_size[1]), (x + text_size[0], y), (255, 0, 0), cv2.FILLED)
    cv2.putText(input_img, label, loc, font, font_scale, (255, 255, 255), thickness)


def draw_faces(detected, input_img, ad, img_size, model: keras.Model):
    for i, (x, y, w, h) in enumerate(detected):
        x_1 = x
        y_1 = y
        x_2 = x + w
        y_2 = y + h

        xw1 = max(int(x_1 - ad * w), 0)
        yw1 = max(int(y_1 - ad * h), 0)
        xw2 = min(int(x_2 + ad * w), img_w - 1)
        yw2 = min(int(y_2 + ad * h), img_h - 1)

        faces = np.zeros(shape=(len(detected), img_size[0], img_size[1], 3))
        faces[i, :, :, :] = cv2.resize(input_img[yw1: yw2 + 1, xw1: xw2 + 1, :], img_size) / 255
        # faces[i, :, :, :] = mpimg.imread(temp)
        # faces[i, :, :, :] = cv2.normalize(faces[i, :, :, :], None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

        cv2.rectangle(input_img, (x_1, y_1), (x_2, y_2), (255, 0, 0), 2)
        cv2.rectangle(input_img, (xw1, yw1), (xw2, yw2), (0, 0, 255), 2)

    if len(detected) > 0:
        ages_pred = model.predict(faces)

    for i, (x, y, w, h) in enumerate(detected):
        x_1 = x
        y_1 = y
        x_2 = x + w
        y_2 = y + h

        age_label = str(int(ages_pred[i]))
        draw_label(input_img=input_img, loc=(x_1, y_1), label=age_label)

    # cv2.imshow('result', input_img)
    return input_img


if __name__ == '__main__':
    while True:
        # get video frame
        ret, input_img = cap.read()
        img_idx += 1
        img_h, img_w, _ = np.shape(input_img)

        if img_idx == 1 or img_idx % skip_frame == 0:
            # detect faces using LBP detector
            gray_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
            start_time = timeit.default_timer()
            detected = face_cascade.detectMultiScale(gray_img, 1.1)

            print(detected)

            input_img = draw_faces(detected=detected, input_img=input_img, ad=ad, img_size=img_size, model=model)
            cv2.imwrite('img/' + str(img_idx) + '.png', input_img)
