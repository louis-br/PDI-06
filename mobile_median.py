import cv2
import numpy as np
from car_counters import TrackingCarCounter, LineCarCounter

COUNTER = TrackingCarCounter
FILENAME = 'videos/rodovia2'
FILE = f'{FILENAME}.mp4'
ROI = f'{FILENAME}_roi.json'

def main():
    counter = COUNTER(ROI)
    kernel = np.ones((5,5), np.uint8)
    file = 'rodovia2'
    vcap = cv2.VideoCapture('videos/' + file + '.mp4')
    ret, frame = vcap.read()
    width = int(vcap.get(3))
    height = int(vcap.get(4))
    size = (width, height)
    mask = cv2.VideoWriter('bg/median.avi', cv2.VideoWriter_fourcc(*'MJPG'), 30.0, size)
    atual = 0
    frames = []
    frame = cv2.GaussianBlur(frame,(5,5), cv2.BORDER_DEFAULT)
    while True:
        if not ret:
            break
        prev_frame = frame[:]
        atual = atual + 1
        frames.append(prev_frame)
        ret, frame = vcap.read()
        frame = cv2.GaussianBlur(frame,(5,5), cv2.BORDER_DEFAULT)
        if atual < 5:
            continue
        else:
            frames.pop(0)

        median = np.median(frames, axis=0).astype(dtype=np.uint8)
        gray_md = cv2.cvtColor(median, cv2.COLOR_RGB2GRAY)
        gray_fr = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        diff = cv2.absdiff(gray_fr, gray_md)

        med_morph = cv2.morphologyEx(diff, cv2.MORPH_CLOSE, kernel, iterations=1)
        med_morph = cv2.morphologyEx(med_morph, cv2.MORPH_ERODE, kernel, iterations=1)

        _, median_th = cv2.threshold(med_morph, 10, 1, cv2.THRESH_BINARY)
        cv2.imshow('frame', median_th*255)
        #cv2.imshow('frame', diff)

        median_th = np.uint8(median_th)

        counter.update(median_th)

        median_convert = cv2.cvtColor(median_th, cv2.COLOR_GRAY2RGB)
        mask.write(median_convert*255)
        cv2.imshow('og', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cv2.imwrite('bg/median-' + file + '.png', median)
    mask.release()
    vcap.release()
main()

