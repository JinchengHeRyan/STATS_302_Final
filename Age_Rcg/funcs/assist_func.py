import cv2
import keras
import numpy as np
from Model.migration_model.migration import find_mig


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
    #     ages_output = str()
    #     for i in range(len(ages_pred)):
    #         ages_output += str(int(ages_pred[i])) + ' '
    # else:
    #     ages_output = None

    ages_output = str()

    for i, (x, y, w, h) in enumerate(detected):
        x_1 = x
        y_1 = y
        x_2 = x + w
        y_2 = y + h

        age_label = str(int(ages_pred[i])) + find_mig(prediction=int(ages_pred[i]), fig=faces[i].reshape(1, img_size[0], img_size[1], 3))
        draw_label(input_img=input_img, loc=(x_1, y_1), label=age_label)
        ages_output += age_label + ' '

    # cv2.imshow('result', input_img)

    return input_img, ages_output


def mtcnn_detect(img, img_size, model_mtcnn, model_SSR):
    threshold = [0.5, 0.6, 0.7]
    rectangles = model_mtcnn.detectFace(img, threshold)

    faces = []
    face_num = 0
    ages_output = str()

    for i in range(len(rectangles)):
        rectangle = rectangles[i]

        if rectangle is not None:

            x_1 = int(rectangle[0])
            y_1 = int(rectangle[1])
            x_2 = int(rectangle[2])
            y_2 = int(rectangle[3])

            face_num += 1

            face = cv2.resize(img[y_1: y_2 + 1, x_1: x_2 + 1, :], img_size) / 255
            face = face.reshape(1, img_size[0], img_size[1], 3)

            age = int(model_SSR.predict(face))

            migration_output = find_mig(prediction=age, fig=face)
            ages_output += str(age) + migration_output + ' '

            cv2.rectangle(img, (int(rectangle[0]), int(rectangle[1])), (int(rectangle[2]), int(rectangle[3])),
                          (255, 0, 0), 2)

            draw_label(input_img=img, loc=(x_1, y_1), label=str(age)+migration_output)

            for i in range(5, 15, 2):
                cv2.circle(img, (int(rectangle[i + 0]), int(rectangle[i + 1])), 2, (0, 255, 0))

    return face_num, ages_output
