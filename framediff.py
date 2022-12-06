import cv2
import numpy as np
from car_counters import CentroidCarCounter, LineCarCounter

COUNTER = CentroidCarCounter
FILENAME = 'videos/rodovia2'
FILE = f'{FILENAME}.mp4'
ROI = f'{FILENAME}_roi.json'

def main():
    counter = COUNTER(ROI)
    kernel = np.ones((5,5), np.uint8)
    file = 'rodovia2'
    #abrir o arquivo
    vcap = cv2.VideoCapture('videos/' + file +'.mp4')
    #bool se arquivo abriu e frame atual
    ret, frame = vcap.read()
    prev_frame = frame

    frame_diff = cv2.absdiff(frame,prev_frame)

    while True:
        if not ret:
            break
        frame = cv2.GaussianBlur(frame,(5,5), cv2.BORDER_DEFAULT)
        curr_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        prev_frame_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

        frame_diff = cv2.absdiff(curr_frame_gray,prev_frame_gray)

        frame_diff = cv2.morphologyEx(frame_diff, cv2.MORPH_OPEN, kernel, iterations=3)
        frame_diff = cv2.morphologyEx(frame_diff, cv2.MORPH_CLOSE, kernel, iterations=1)
        #frame_diff = cv2.dilate(frame_diff, kernel, iterations=1)

        _, fd_th = cv2.threshold(frame_diff, 0, 1, cv2.THRESH_BINARY)

        cv2.imshow('frame diff',fd_th*255)

        fd_th = np.uint8(fd_th)

        counter.update(fd_th)

        cv2.imshow('og', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        prev_frame = frame.copy()
        ret, frame = vcap.read()

    cv2.imwrite("bg/fd-frame-" +file + ".png", frame)
    cv2.imwrite("bg/framediff-" +file +".png", frame_diff)


main()