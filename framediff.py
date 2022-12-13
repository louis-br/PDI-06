import cv2
import numpy as np
from car_counters import TrackingCarCounter, LineCarCounter

COUNTER = LineCarCounter
FILENAME = 'videos/rodovia2'
FILE = f'{FILENAME}.mp4'
ROI = f'{FILENAME}_roi.json'

def main():
    counter = COUNTER(ROI)
    kernel = np.ones((5,5), np.uint8)
    vcap = cv2.VideoCapture(FILE)
    width = int(vcap.get(3))
    height = int(vcap.get(4))
    size = (width, height)
    mask = cv2.VideoWriter('bg/fd.avi', cv2.VideoWriter_fourcc(*'MJPG'), 30.0, size)
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

        #frame_diff = cv2.morphologyEx(frame_diff, cv2.MORPH_OPEN, kernel, iterations=3)
        #frame_diff = cv2.morphologyEx(frame_diff, cv2.MORPH_CLOSE, kernel, iterations=1)
        #frame_diff = cv2.dilate(frame_diff, kernel, iterations=1)

        _, fd_th = cv2.threshold(frame_diff, 10, 1, cv2.THRESH_BINARY)

        cv2.imshow('frame diff',fd_th*255)

        fd_th = np.uint8(fd_th)

        counter.update(fd_th)

        fd_convert = cv2.cvtColor(fd_th, cv2.COLOR_GRAY2BGR)
        mask.write(fd_convert*255)

        cv2.imshow('og', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        prev_frame = frame.copy()
        ret, frame = vcap.read()

    #cv2.imwrite("bg/fd-frame-" +FILE + ".png", frame)
    #cv2.imwrite("bg/framediff-" +FILE +".png", frame_diff)


main()