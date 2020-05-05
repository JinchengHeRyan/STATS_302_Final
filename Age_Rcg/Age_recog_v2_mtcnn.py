import sys
sys.path.append('../')

import cv2
import os
from Model.SSR_net import SSR_net
from Model.mtcnn.mtcnn_model import mtcnn
from Age_Rcg.funcs.assist_func import draw_faces, face_count, mtcnn_detect
import time

model_SSR = SSR_net(image_size=200, stage_num=[3, 3, 3], lambda_local=0.25, lambda_d=0.25)()
model_SSR.load_weights('../Output/output_3/weights-improvement-24-8.21.h5')

model_mtcnn = mtcnn()


# face_cascade = cv2.CascadeClassifier('lbpcascade_frontalface_improved.xml')


def realtime_recog():
    img_out_path = './img_' + str(int(time.time()))
    try:
        os.mkdir(img_out_path)
    except OSError:
        pass

    img_size = (200, 200)
    cap = cv2.VideoCapture(0)

    # detected = ''  # make this not local variable
    skip_frame = 5  # every 5 frame do 1 detection and network forward propagation
    ad = 0.25

    img_idx = 0
    while True:
        # get video frame
        ret, input_img = cap.read()
        img_idx += 1

        if img_idx == 1 or img_idx % skip_frame == 0:

            face_num, ages_output = mtcnn_detect(input_img, img_size, model_mtcnn=model_mtcnn, model_SSR=model_SSR)

            print("Detected {} faces! ".format(face_num) if face_num > 0 else "Detect No Faces! ")

            if face_num > 0:
                print('\t' + ages_output + 'years old')

            cv2.imwrite(img_out_path + '/' + str(img_idx) + '.png', input_img)
            cv2.imshow('result', input_img)
            cv2.waitKey(1)


def static_recog(input_img_path: str):
    img_out_path = './img_static'
    try:
        os.mkdir(img_out_path)
    except OSError:
        pass

    ad = 0.25
    img_size = (200, 200)

    input_img = cv2.imread(input_img_path)

    face_num, ages_output = mtcnn_detect(input_img, img_size, model_mtcnn=model_mtcnn, model_SSR=model_SSR)

    print("Detected {} faces! ".format(face_num) if face_num > 0 else "Detect No Faces! ")

    if face_num > 0:
        print('\t' + ages_output + 'years old')

    cv2.imwrite(img_out_path + '/' + str(int(time.time())) + '.png', input_img)


if __name__ == '__main__':
    Mode = 0        # 0 is real time mode, 1 is static mode

    if Mode == 0:
        realtime_recog()
    else:
        input_file_path = ['1.png', '2.png']
        for path in input_file_path:
            static_recog(path)
