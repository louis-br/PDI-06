import cv2
import numpy as np
from car_counters import TrackingCarCounter, LineCarCounter


COUNTER = TrackingCarCounter
FILENAME = 'videos/rodovia2'
FILE = f'{FILENAME}.mp4'
ROI = f'{FILENAME}_roi.json'

def main():
    counter = COUNTER(ROI)
    file = 'rodovia2'
    vcap = cv2.VideoCapture('videos/' + file + '.mp4')
    ret, frame = vcap.read()

    #selecionar 25 frames aleatorios no video -> resultado mais "liso"(smooth)
    while True:
        kernel = np.ones((5,5), np.uint8)
        frame = cv2.GaussianBlur(frame,(5,5), 0)
        selectframe = vcap.get(cv2.CAP_PROP_FRAME_COUNT) * np.random.uniform(size=25)
        frames = []
        for fr in selectframe:
            vcap.set(cv2.CAP_PROP_POS_FRAMES, fr)
            ret, frame = vcap.read()
            frames.append(frame)


        mean = np.mean(frames, axis=0).astype(dtype=np.uint8)
        gray_avg = cv2.cvtColor(mean, cv2.COLOR_RGB2GRAY)
        gray_fr = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        diff = cv2.absdiff(gray_fr, gray_avg)

        avg_morph = cv2.morphologyEx(diff, cv2.MORPH_ERODE, kernel, iterations=1)
        avg_morph = cv2.morphologyEx(diff, cv2.MORPH_OPEN, kernel, iterations=1)

        _, mean_th = cv2.threshold(avg_morph, 20, 1, cv2.THRESH_BINARY)
        cv2.imshow('frame', diff)
        #cv2.imshow('frame', mean_th)

        mean_th = np.uint8(mean_th)

        counter.update(mean_th)

        cv2.imshow('og', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cv2.imwrite('bg/mean-' + file + '.png', mean)
main()

