import numpy as np
import cv2
from time import sleep
from car_counters import CentroidCarCounter, LineCarCounter

COUNTER = CentroidCarCounter
FILENAME = 'videos/rodovia2'
FILE = f'{FILENAME}.mp4'
ROI = f'{FILENAME}_roi.json'
THRESH = 32
MHI_DURATION = 8


def mhi():
    counter = COUNTER(ROI)

    kernel = np.ones((3,3), np.uint8)
    vcap = cv2.VideoCapture(FILE)
    ret, frame = vcap.read()
    h, w = frame.shape[:2]
    print(h, w)
    prev_frame = frame.copy()
    motion_history = np.zeros((h, w), np.float32)
    timestamp = 0
    while vcap.isOpened():
        ret, frame = vcap.read()
        time = float(1/90)
        sleep(time)
        if not ret:
            break
        frame_diff = cv2.absdiff(frame, prev_frame)
        gray_diff = cv2.cvtColor(frame_diff, cv2.COLOR_BGR2GRAY)
        gauss = cv2.GaussianBlur(gray_diff,(3,3), cv2.BORDER_DEFAULT)
        ret, fgmask = cv2.threshold(gauss, THRESH, 1, cv2.THRESH_BINARY)
        timestamp += 1

        # update motion history
        cv2.motempl.updateMotionHistory(fgmask, motion_history, timestamp, MHI_DURATION)

        # normalize motion history
        mh = np.clip((motion_history - (timestamp - MHI_DURATION)) / MHI_DURATION, 0, 1)
        _, mh2 = cv2.threshold(mh, 0, 1, cv2.THRESH_BINARY)
        cv2.imshow('motion-history', mh2*255)

        mh2 = np.uint8(mh2)

        counter.update(mh2)

        prev_frame = frame.copy()
        #frame = cv2.line(frame, (1, 200), (599, 200), (255,0,0), 1)
        cv2.imshow('og', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

mhi()