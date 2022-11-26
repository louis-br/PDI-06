import numpy as np
import cv2

MHI_DURATION = 50


def main():
    file = 'rodovia2'
    thresh = 32
    vcap = cv2.VideoCapture('videos/' + file +'.mp4')
    ret, frame = vcap.read()
    h, w = frame.shape[:2]
    prev_frame = frame.copy()
    motion_history = np.zeros((h, w), np.float32)
    timestamp = 0
    while True:
        ret, frame = vcap.read()
        if not ret:
            break
        frame_diff = cv2.absdiff(frame, prev_frame)
        gray_diff = cv2.cvtColor(frame_diff, cv2.COLOR_BGR2GRAY)
        ret, fgmask = cv2.threshold(gray_diff, thresh, 1, cv2.THRESH_BINARY)
        timestamp += 1

        # update motion history
        cv2.motempl.updateMotionHistory(fgmask, motion_history, timestamp, MHI_DURATION)

        # normalize motion history
        mh = np.uint8(np.clip((motion_history - (timestamp - MHI_DURATION)) / MHI_DURATION, 0, 1) * 255)
        cv2.imshow('motion-history', mh)
        cv2.imshow('og', frame)

        prev_frame = frame.copy()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
main()