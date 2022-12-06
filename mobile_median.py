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
    vcap = cv2.VideoCapture('videos/' + file + '.mp4')
    ret, frame = vcap.read()

    #selecionar 25 frames aleatorios no video -> resultado mais "liso"(smooth)
    while True:
        if not ret:
            break
        frame = cv2.GaussianBlur(frame,(5,5), cv2.BORDER_DEFAULT)
        selectframe = vcap.get(cv2.CAP_PROP_FRAME_COUNT) * np.random.uniform(size=15)
        frames = []
        for fr in selectframe:
            vcap.set(cv2.CAP_PROP_POS_FRAMES, fr)
            ret, frame = vcap.read()
            frames.append(frame)

        median = np.median(frames, axis=0).astype(dtype=np.uint8)
        gray_md = cv2.cvtColor(median, cv2.COLOR_RGB2GRAY)
        gray_fr = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        diff = cv2.absdiff(gray_fr, gray_md)

        med_open = cv2.morphologyEx(diff, cv2.MORPH_ERODE, kernel, iterations=1)
        med_open = cv2.morphologyEx(diff, cv2.MORPH_OPEN, kernel, iterations=1)

        _, median_th = cv2.threshold(med_open, 10, 1, cv2.THRESH_BINARY)
        #cv2.imshow('frame', median_th)
        cv2.imshow('frame', diff)

        median_th = np.uint8(median_th)

        counter.update(median_th)

        cv2.imshow('og', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cv2.imwrite('bg/median-' + file + '.png', median)
main()

