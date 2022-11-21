import cv2
import os
import mediapipe
import numpy as np
import matplotlib.pyplot as plt

def resize_imgs():
    origin = 'downloaded_imgs'
    directory = 'background_imgs'
    for file in os.listdir(origin):
        f = os.path.join(origin, file)
        img = cv2.imread(f)
        img = cv2.resize(img, (640, 480))
        cv2.imwrite(directory + '/' + file, img)
        print("Imagem " + file + " com tamanho 640x480")

def video_bg():
    change_bg = mediapipe.solutions.selfie_segmentation
    segmentate = change_bg.SelfieSegmentation()

    vcap = cv2.VideoCapture('videos/example1.mp4') #captura da webcam imbutida no laptop
    # width = vcap.get(3)
    # height = vcap.get(4)
    # fps = vcap.get(5)
    # print(width, height, fps)
    # framecount = vcap.get(7)
    # print('frames', framecount)
    vcap.set(3, 640)
    vcap.set(4, 480)

    while True:
        success, img = vcap.read()
        if success == True:
            new_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            result = segmentate.process(new_img)
            mask = result.segmentation_mask > 0.5
            mask_3 = np.dstack((mask, mask, mask))
            output = np.where(mask_3, new_img, 255)

            cv2.imshow("Image", img)
            cv2.imshow("segment", output)

            k = cv2.waitKey(10)
            #se ESC for apertado, break
            if k == 27:
                break

#resize_imgs()
video_bg()