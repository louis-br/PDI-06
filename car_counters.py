import json
import cv2
import numpy as np

class Rectangle:
    def __init__(self, y=0, x=0, y2=None, x2=None, height=0, width=0):
        self.y = y
        self.x = x
        self.y2 = y2 if y2 else y + height
        self.x2 = x2 if x2 else x + width

    def height(self):
        return self.y2 - self.y

    def width(self):
        return self.x2 - self.x

    def from_dict(self, dict):
        self.__init__(y=dict['y'], x=dict['x'], height=dict['height'], width=dict['width'])

    def intersection(self, other):
        ax1, ay1, ax2, ay2 = self.x, self.y, self.x + self.width, self.y + self.height
        bx1, by1, bx2, by2 = other.x, other.y, other.x + other.width, other.y + other.height
        x1 = max(min(ax1, ax2), min(bx1, bx2))
        y1 = max(min(ay1, ay2), min(by1, by2))
        x2 = min(max(ax1, ax2), max(bx1, bx2))
        y2 = min(max(ay1, ay2), max(by1, by2))
        return Rectangle(y=y1, x=x1, y2=y2, x2=x2) if x1 < x2 and y1 < y2 else None


def get_roi_from_json(path):
    with open(path, 'r') as file:
        rect = Rectangle()
        rect.from_dict(json.load(file))
        return rect


class LineCarCounter:
    def __init__(self, ROI_path, min_height=50, min_width=50): #, line_size=25, line_y_percent=0.5):
        self.ROI = get_roi_from_json(ROI_path)
        self.min_height = min_height
        self.min_width = min_width
        self.last_contour_count = 0
        self.cars = 0
        #self.ROI.y = int((self.ROI.y + self.ROI.height) * line_y_percent - line_size/2)
        #self.ROI.height = line_size//2

    def update(self, frame):
        frame = frame[self.ROI.y:self.ROI.y + self.ROI.height, self.ROI.x:self.ROI.x + self.ROI.width]
        contours, _ = cv2.findContours(frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        img = cv2.drawContours(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)*255, contours, -1, (255,0,0), 3)
        count = 0
        for contour in contours:
            x,y,w,h = cv2.boundingRect(contour)
            if w < self.min_width and h < self.min_height:
                continue
            img = cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
            count = count + 1
        
        new = max(self.last_contour_count - count, 0)
        self.last_contour_count = count
        self.cars += new

        cv2.imshow("contours", img)

        print(f'Cars: {self.cars} New: {self.last_contour_count - count}')
        return new


class TrackingCarCounter:
    def __init__(self, ROI_path, min_height=50, min_width=50):
        self.ROI = get_roi_from_json(ROI_path)
        self.last_contours = []
        self.min_height = min_height
        self.min_width = min_width

    def update(self, frame):
        frame = frame[self.ROI.y:self.ROI.y + self.ROI.height, self.ROI.x:self.ROI.x + self.ROI.width]
        # print(frame.type)
        #cv2.imshow("crop", frame*255)
        contours, _ = cv2.findContours(frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        img = cv2.drawContours(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)*255, contours, -1, (255,0,0), 3)
        for contour in contours:
            x,y,w,h = cv2.boundingRect(contour)
            if w < self.min_width and h < self.min_height:
                continue
            img = cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)

        cv2.imshow("contours", img)
        return 0