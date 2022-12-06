import cv2
import json

FILENAME = 'videos/rodovia1'
FILE = f'{FILENAME}.mp4'
ROI = f'{FILENAME}_roi.json'

def main():
    vcap = cv2.VideoCapture(FILE)
    ret, frame = vcap.read()
    rect = cv2.selectROI("ROI", frame)
    print(rect)
    with open(ROI, 'w') as file:
        json.dump({
            'x': rect[0],
            'y': rect[1],
            'width': rect[2],
            'height': rect[3]
        }, file)

main()