import json
import cv2
import numpy as np

class Rectangle:
    def __init__(self, y=0, x=0, height=0, width=0):
        self.y = y
        self.x = x
        self.height = height
        self.width = width

    def from_dict(self, dict):
        self.y = dict['y']
        self.x = dict['x']
        self.height = dict['height']
        self.width = dict['width']

def get_roi_from_json(path):
    with open(path, 'r') as file:
        rect = Rectangle()
        rect.from_dict(json.load(file))
        return rect

#uso da linha para contagem
class LineCarCounter:
    def __init__(self):
        #self.lastFrame
        pass

    def update(self, frame):
        pass

#uso dos contornos para pegar o centroide de cada carro e o retangulo ao redor deles
class CentroidCarCounter:
    def __init__(self, ROI_path):
        self.ROI = get_roi_from_json(ROI_path)
        self.lastFrame = np.zeros((self.ROI.height, self.ROI.width))

    def update(self, frame):
        frame = frame[self.ROI.y:self.ROI.y + self.ROI.height, self.ROI.x:self.ROI.x + self.ROI.width]
        # print(frame.type)
        cv2.imshow("crop", frame*255)
        contours, _ = cv2.findContours(frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        img = cv2.drawContours(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)*255, contours, -1, (0,255,0), 3)
        print(img.shape)
        for contour in contours:
            x,y,w,h = cv2.boundingRect(contour)
            img = cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
        cv2.imshow("contours", img)
        return 0