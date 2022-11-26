import cv2
import numpy as np

def main():
    file = 'rodovia2'
    vcap = cv2.VideoCapture('videos/' + file + '.mp4')
    ret, frame = vcap.read()

    #selecionar 25 frames aleatorios no video -> resultado mais "liso"(smooth)
    while True:
        if not ret:
            break
        selectframe = vcap.get(cv2.CAP_PROP_FRAME_COUNT) * np.random.uniform(size=25)
        frames = []
        for fr in selectframe:
            vcap.set(cv2.CAP_PROP_POS_FRAMES, fr)
            ret, frame = vcap.read()
            frames.append(frame)

    # while True:
    #     selectframe = vcap.get(cv2.CAP_PROP_FRAME_COUNT) * np.random.uniform(size=cv2.CAP_PROP_FRAME_COUNT)
    #     frames = []
    #     for fr in selectframe:
    #         vcap.set(cv2.CAP_PROP_POS_FRAMES, fr)
    #         ret, frame = vcap.read()
    #         frames.append(frame)


        median = np.median(frames, axis=0).astype(dtype=np.uint8)
        cv2.imshow('frame', median)
        cv2.imshow('og', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cv2.imwrite('bg/median-' + file + '.png', median)
main()

