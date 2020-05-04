import cv2
import keras
import numpy as np
from Model.mtcnn.mtcnn_model import mtcnn
from Model.SSR_net import SSR_net


def face_count(detected):
    return len(detected)


def draw_label(input_img, loc, label):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    thickness = 2
    text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
    x, y = loc
    cv2.rectangle(input_img, (x, y - text_size[1]), (x + text_size[0], y), (255, 0, 0), cv2.FILLED)
    cv2.putText(input_img, label, loc, font, font_scale, (255, 255, 255), thickness)


def draw_faces(detected, input_img, ad, img_size, model: keras.Model):
    img_h, img_w, _ = np.shape(input_img)

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
        # faces[i, :, :, :] = cv2.resize(input_img[yw1: yw2 + 1, xw1: xw2 + 1, :], img_size) / 255
        faces[i, :, :, :] = cv2.resize(input_img[y_1: y_2 + 1, x_1: x_2 + 1, :], img_size) / 255

        cv2.rectangle(input_img, (x_1, y_1), (x_2, y_2), (255, 0, 0), 2)
        cv2.rectangle(input_img, (xw1, yw1), (xw2, yw2), (0, 0, 255), 2)

    if len(detected) > 0:
        ages_pred = model.predict(faces)
        ages_output = str()
        for i in range(len(ages_pred)):
            ages_output += str(int(ages_pred[i])) + ' '
    else:
        ages_output = None

    for i, (x, y, w, h) in enumerate(detected):
        x_1 = x
        y_1 = y
        x_2 = x + w
        y_2 = y + h

        age_label = str(int(ages_pred[i]))
        draw_label(input_img=input_img, loc=(x_1, y_1), label=age_label)

    # cv2.imshow('result', input_img)

    return input_img, ages_output


def mtcnn_detect(img, img_size):
    model = mtcnn()
    threshold = [0.5, 0.6, 0.7]
    rectangles = model.detectFace(img, threshold)

    faces = []
    detected = []
    for i in range(len(rectangles)):
        rectangle = rectangles[i]
        print('rectangle = ', rectangle)
        if rectangle is not None:
            W = -int(rectangle[0]) + int(rectangle[2])
            H = -int(rectangle[1]) + int(rectangle[3])
            paddingH = 0.01 * W
            paddingW = 0.02 * H
            crop_img = img[int(rectangle[1] + paddingH):int(rectangle[3] - paddingH),
                       int(rectangle[0] - paddingW):int(rectangle[2] + paddingW)]
            if crop_img is None:
                continue
            if crop_img.shape[0] < 0 or crop_img.shape[1] < 0:
                continue

            x_1 = int(rectangle[0])
            y_1 = int(rectangle[1])
            x_2 = int(rectangle[2])
            y_2 = int(rectangle[3])

            detected.append((x_1, y_1, x_2, y_2))

            faces.append(cv2.resize(img[y_1: y_2 + 1, x_1: x_2 + 1, :], img_size) / 255)

            cv2.rectangle(img, (int(rectangle[0]), int(rectangle[1])), (int(rectangle[2]), int(rectangle[3])),
                          (255, 0, 0), 2)

            for i in range(5, 15, 2):
                cv2.circle(img, (int(rectangle[i + 0]), int(rectangle[i + 1])), 2, (0, 255, 0))

    faces = np.array(faces)

    if len(detected) > 0:
        model_2 = SSR_net(image_size=200, stage_num=[3, 3, 3], lambda_local=0.25, lambda_d=0.25)()
        model_2.load_weights('../Output/output_3/weights-improvement-24-8.21.h5')
        ages_pred = model_2.predict(faces)
        ages_output = str()
        for i in range(len(ages_pred)):
            ages_output += str(int(ages_pred[i])) + ' '
    else:
        ages_output = None

    for i, (x_1, y_1, x_2, y_2) in enumerate(detected):
        age_label = str(int(ages_pred[i]))
        draw_label(input_img=img, loc=(x_1, y_1), label=age_label)

    return detected, ages_output
