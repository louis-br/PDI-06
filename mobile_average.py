import cv2
import numpy as np
from car_counters import TrackingCarCounter, LineCarCounter


COUNTER = TrackingCarCounter
FILENAME = 'videos/rodovia2'
FILE = f'{FILENAME}.mp4'
ROI = f'{FILENAME}_roi.json'

def main():
    counter = COUNTER(ROI)
    vcap = cv2.VideoCapture(FILE)
    ret, frame = vcap.read()
    width = int(vcap.get(3))
    height = int(vcap.get(4))
    size = (width, height)
    mask = cv2.VideoWriter('bg/avg.avi', cv2.VideoWriter_fourcc(*'MJPG'), 30.0, size)
    frames=[]
    frame = cv2.GaussianBlur(frame,(5,5), 0)
    kernel = np.ones((5,5), np.uint8)
    atual = 0

    #selecionar 25 frames aleatorios no video -> resultado mais "liso"(smooth)
    while True:
        if not ret:
            break
        prev_frame = frame[:]
        atual = atual + 1
        frames.append(prev_frame)
        ret, frame = vcap.read()
        frame = cv2.GaussianBlur(frame,(5,5), 0)
        if atual < 5:
            continue
        else:
            frames.pop(0)

        mean = np.mean(frames, axis=0).astype(dtype=np.uint8)
        gray_avg = cv2.cvtColor(mean, cv2.COLOR_RGB2GRAY)
        gray_fr = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        diff = cv2.absdiff(gray_fr, gray_avg)

        avg_morph = cv2.morphologyEx(diff, cv2.MORPH_CLOSE, kernel, iterations=1)
        avg_morph = cv2.morphologyEx(avg_morph, cv2.MORPH_ERODE, kernel, iterations=1)

        _, mean_th = cv2.threshold(avg_morph, 20, 1, cv2.THRESH_BINARY)
        #cv2.imshow('frame', diff)
        cv2.imshow('frame', mean_th*255)

        mean_th = np.uint8(mean_th)

        counter.update(mean_th)

        avg_convert = cv2.cvtColor(mean_th, cv2.COLOR_GRAY2BGR)
        mask.write(avg_convert*255)

        cv2.imshow('og', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cv2.imwrite('bg/mean-' + file + '.png', mean)
    mask.release()
    vcap.release()
main()

