import cv2
import os
from Model.SSR_net import SSR_net
from Age_Rcg.funcs.assist_func import draw_faces, face_count
import time

model = SSR_net(image_size=200, stage_num=[3, 3, 3], lambda_local=0.25, lambda_d=0.25)()
model.load_weights('../Output/output_3/weights-improvement-24-8.21.h5')


face_cascade = cv2.CascadeClassifier('lbpcascade_frontalface_improved.xml')


def realtime_recog():
    img_out_path = './img_' + str(int(time.time()))
    try:
        os.mkdir(img_out_path)
    except OSError:
        pass

    img_size = (200, 200)
    stage_num = [3, 3, 3]
    cap = cv2.VideoCapture(0)

    # detected = ''  # make this not local variable
    skip_frame = 5  # every 5 frame do 1 detection and network forward propagation
    ad = 0.5

    img_idx = 0
    while True:
        # get video frame
        ret, input_img = cap.read()
        img_idx += 1

        if img_idx == 1 or img_idx % skip_frame == 0:
            # detect faces using LBP detector
            gray_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
            detected = face_cascade.detectMultiScale(gray_img, 1.1)

            print("Detected {} faces! ".format(face_count(detected)) if face_count(detected) > 0 else "Detect No Faces! ")

            input_img, ages_output = draw_faces(detected=detected, input_img=input_img, ad=ad, img_size=img_size, model=model)

            if face_count(detected)> 0:
                print('\t' + ages_output + 'years old')

            cv2.imwrite(img_out_path + '/' + str(img_idx) + '.png', input_img)


def static_recog(input_img_path: str):
    img_out_path = './img_static'
    try:
        os.mkdir(img_out_path)
    except OSError:
        pass
    input_img = cv2.imread(input_img_path)
    gray_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
    detected = face_cascade.detectMultiScale(gray_img, 1.1)
    print("Detected {} faces! ".format(face_count(detected)) if face_count(detected) > 0 else "Detect No Faces! ")

    ad = 0.5
    img_size = (200, 200)
    input_img, ages_output = draw_faces(detected=detected, input_img=input_img, ad=ad, img_size=img_size, model=model)

    if face_count(detected) > 0:
        print('\t' + ages_output + 'years old')

    cv2.imwrite(img_out_path + '/' + 'static' + '.png', input_img)


if __name__ == '__main__':
    Mode = 0        # 0 is real time mode, 1 is static mode

    if Mode == 0:
        realtime_recog()
    else:
        input_file_path = '1.jpg'
        static_recog(input_file_path)
